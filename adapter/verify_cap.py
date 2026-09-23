"""Per-step verify length for DSpark without changing the verify layout. Default OFF.

The target verify runs the fixed [bs, 6] rectangle (anchor + 5 drafts). With DSV41_VERIFY_CAP set,
each request gets a live length L = 1 + k (k = drafts actually verified):

  * acceptance is capped at k drafts through the engine's own cutoff (CapCorrectLen + bonus from
    row k), so the committed tokens are exactly those of a verify with k drafts;
  * the router ids of the dead rows (k+1..5) are replaced by the anchor row's ids, so the dead rows
    add no experts to the per-layer set the MoE streams (the saving; MoE is bandwidth-bound);
  * dead rows are never committed (commit <= k + 1), and live rows only attend to earlier rows, so
    their wrong MoE output cannot reach a committed token.

k comes from DSV41_VERIFY_CAP:
  "1".."5"     fixed k for every step (5 = all live, measures the machinery's own cost)
  "conf:T"     k = number of leading drafts whose running product of the draft confidence head's
               per-position survival stays >= T (at least DSV41_VERIFY_CAP_MIN, default 1).
               The head's value for position i is computed from drafts before i only, so the
               cut is a stopping rule: it never looks at the token it drops (exact for sampling).
Sampled rows go through block_verify, which treats dead positions as q = 0, p(X) = 0.
"""
import os

import torch

SPEC = os.environ.get("DSV41_VERIFY_CAP", "").strip()
ENABLED = SPEC not in ("", "0", "off")
STRIDE = 6
MAX_BS = 64
E_TARGET, K_TARGET = 384, 6

_state = {"live": None, "in_verify": False, "conf": None, "fixed": None, "thr": None,
          "kmin": int(os.environ.get("DSV41_VERIFY_CAP_MIN", "1")), "logged": False}
if ENABLED:
    if SPEC.startswith("conf:"):
        _state["thr"] = float(SPEC.split(":", 1)[1])
    else:
        _state["fixed"] = int(SPEC)


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def live_buf(device=None):
    buf = _state["live"]
    if buf is None:
        assert not _capturing(), "verify_cap: live buffer must exist before capture"
        buf = torch.full((MAX_BS,), STRIDE, dtype=torch.int64,
                         device=device if device is not None else torch.cuda.current_device())
        if _state["fixed"] is not None:
            buf.fill_(1 + _state["fixed"])
        _state["live"] = buf
    return buf


def cutoff(bs):
    return live_buf()[:bs] if ENABLED else None


def set_live_from_confidence(confidence, bs):
    """confidence [bs, 5] per-position survival (position i given drafts < i)."""
    if _state["thr"] is None or confidence is None:
        return
    buf = live_buf(confidence.device)
    cum = torch.cumprod(confidence.float().clamp(0, 1), dim=1)
    k = (cum >= _state["thr"]).to(torch.int64).cumprod(dim=1).sum(dim=1).clamp(min=_state["kmin"], max=STRIDE - 1)
    buf[:bs].copy_(k + 1)


def _src_rows(m, device):
    """Row each verify row takes its router output from (itself if live, else its anchor).
    Computed once per forward and reused by all 43 layers (inside the graph: once per replay)."""
    src = _state.get("src")
    if src is not None and src.shape[0] == m:
        return src
    live = live_buf(device)
    rows = torch.arange(m, device=device)
    req, pos = rows // STRIDE, rows % STRIDE
    dead = pos >= live[req.clamp(max=MAX_BS - 1)]
    src = torch.where(dead, req * STRIDE, rows)
    _state["src"] = src
    if _capturing() and m not in _state.setdefault("captured", set()):
        _state["captured"].add(m)
        print(f"[verify_cap] remap captured into the verify graph for M={m}", flush=True)
    return src


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _remap_kernel(W, I, LIVE, M, SW: tl.constexpr, SI: tl.constexpr, STRIDE_R: tl.constexpr,
                      K: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        rmask = r < M
        req = r // STRIDE_R
        pos = r % STRIDE_R
        live = tl.load(LIVE + req, rmask, STRIDE_R)
        dead = rmask & (pos >= live)
        anchor = req * STRIDE_R
        for k in tl.static_range(K):
            w = tl.load(W + anchor * SW + k, dead, 0.0)
            i = tl.load(I + anchor * SI + k, dead, 0)
            tl.store(W + r * SW + k, w, dead)
            tl.store(I + r * SI + k, i, dead)
except Exception:                                   # no triton: fall back to index_select
    _remap_kernel = None


def remap_dead_rows(weights, indices):
    """weights/indices [M, 6] for M = bs * 6 verify rows (request-major). In place when possible:
    one kernel per layer copies the anchor row's ids and weights over the dead rows."""
    m = indices.shape[0]
    if (_remap_kernel is not None and weights.is_contiguous() and indices.is_contiguous()
            and weights.shape[1] == K_TARGET and m <= 1024):
        if _capturing() and m not in _state.setdefault("captured", set()):
            _state["captured"].add(m)
            print(f"[verify_cap] in-place remap captured into the verify graph for M={m}", flush=True)
        _remap_kernel[(1,)](weights, indices, live_buf(indices.device), m, weights.stride(0),
                            indices.stride(0), STRIDE, K_TARGET, triton.next_power_of_2(m))
        return weights, indices
    src = _src_rows(m, indices.device)
    return weights.index_select(0, src), indices.index_select(0, src)


def install_gate(module):
    """sglang.kernels.ops.moe.moe_fused_gate"""
    if not ENABLED or getattr(module, "_dsv41_verify_cap", False):
        return
    module._dsv41_verify_cap = True
    original = module.moe_fused_gate

    def wrapped(scores, bias, topk, *a, **kw):
        weights, indices = original(scores, bias, topk, *a, **kw)
        m = scores.shape[0] if scores.dim() == 2 else 0
        if (_state["in_verify"] and scores.shape[-1] == E_TARGET and topk == K_TARGET
                and m and m % STRIDE == 0 and m // STRIDE <= MAX_BS):
            return remap_dead_rows(weights, indices)
        return weights, indices

    module.moe_fused_gate = wrapped
    print(f"[verify_cap] armed ({SPEC}): dead verify rows route to the anchor's experts", flush=True)


def install_model(dsv4_module):
    """sglang.srt.models.deepseek_v4: mark target-verify forwards (eager and graph capture)."""
    if not ENABLED:
        return
    cls = dsv4_module.DeepseekV4ForCausalLM
    if getattr(cls, "_dsv41_verify_cap", False):
        return
    cls._dsv41_verify_cap = True
    orig = cls.forward

    def forward(self, input_ids, positions, forward_batch, *a, **kw):
        if _state["live"] is None and not _capturing():
            live_buf(input_ids.device)
        prev = _state["in_verify"]
        _state["src"] = None
        _state["in_verify"] = (type(self).__name__ == "DeepseekV4ForCausalLM"
                               and forward_batch.forward_mode.is_target_verify())
        try:
            return orig(self, input_ids, positions, forward_batch, *a, **kw)
        finally:
            _state["in_verify"] = prev

    cls.forward = forward


class _Cut:
    def __init__(self, verify_lens):
        self.verify_lens = verify_lens


def install_verify(verify_module):
    """sglang.srt.speculative.dspark_components.dspark_verify"""
    if not ENABLED:
        return
    ep = verify_module.DsparkVerifyEpilogue
    if getattr(ep, "_dsv41_verify_cap", False):
        return
    ep._dsv41_verify_cap = True
    orig_accept = ep._accept

    def _accept(self, *, candidates, logits, draft_tokens, seq_lens, cutoff_verify_lens=None):
        if cutoff_verify_lens is None:
            cutoff_verify_lens = cutoff(candidates.shape[0])
        return orig_accept(self, candidates=candidates, logits=logits, draft_tokens=draft_tokens,
                           seq_lens=seq_lens, cutoff_verify_lens=cutoff_verify_lens)

    ep._accept = _accept

    orig_adt = verify_module.accept_draft_tokens

    def accept_draft_tokens(*, candidates, cutoff_layout=None, **kw):
        if cutoff_layout is None:
            cutoff_layout = _Cut(cutoff(candidates.shape[0]))
        return orig_adt(candidates=candidates, cutoff_layout=cutoff_layout, **kw)

    verify_module.accept_draft_tokens = accept_draft_tokens

    ex = verify_module.TargetVerifyExecutor
    orig_run = ex.run_non_compact

    def run_non_compact(self, *, batch, draft_input, verify_ids_2d, **kw):
        bs = verify_ids_2d.shape[0]
        if _state["thr"] is not None:
            set_live_from_confidence(_state["conf"], bs)
            if not _state["logged"]:
                _state["logged"] = True
                print(f"[verify_cap] confidence {'present' if _state['conf'] is not None else 'MISSING'}",
                      flush=True)
        _state["conf"] = None
        return orig_run(self, batch=batch, draft_input=draft_input, verify_ids_2d=verify_ids_2d, **kw)

    ex.run_non_compact = run_non_compact


def install_draft(draft_module):
    """sglang.srt.speculative.dspark_components.dspark_draft: keep a folded proposal's confidence."""
    if not ENABLED or _state["thr"] is None:
        return
    cls = draft_module.DraftBlockProposer
    orig = cls.propose

    def propose(self, *a, **kw):
        out = orig(self, *a, **kw)
        if getattr(out, "confidence", None) is not None:
            _state["conf"] = out.confidence
        return out

    cls.propose = propose


def install_planner(planner_module):
    """sglang.srt.speculative.dspark_components.dspark_planner: keep the step's confidence."""
    if not ENABLED or _state["thr"] is None:
        return
    cls = planner_module.DSparkVerifyPlanner
    orig = cls.compute_confidence_tensor

    def compute_confidence_tensor(self, **kw):
        out = orig(self, **kw)
        _state["conf"] = out
        return out

    cls.compute_confidence_tensor = compute_confidence_tensor


def install_dspark(dspark_module):
    """sglang.srt.models.deepseek_v4_dspark: the engine builds the draft's confidence head only in
    the ragged-verify modes; build it in static mode too when the conf policy needs it."""
    if not ENABLED or _state["thr"] is None:
        return
    orig = dspark_module.build_dspark_v4_confidence_head
    mode_fn = dspark_module.read_ragged_verify_mode
    from sglang.srt.speculative.ragged_verify import RaggedVerifyMode

    def build(*a, **kw):
        dspark_module.read_ragged_verify_mode = lambda: RaggedVerifyMode.CAP_ACCEPT
        try:
            head = orig(*a, **kw)
        finally:
            dspark_module.read_ragged_verify_mode = mode_fn
        print(f"[verify_cap] draft confidence head built: {head is not None}", flush=True)
        return head

    dspark_module.build_dspark_v4_confidence_head = build
