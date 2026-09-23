"""Losslessness test for block_verify.block_accept_probs on a toy language model.

Target and draft are context-dependent distributions over V=5 tokens (random but fixed per
prefix). Speculative steps with gamma=3 generate the first 3 tokens of 400k independent samples;
the empirical joint of those 3 tokens must match the target's exact joint (chi-square), and block
verification must accept at least as many tokens per step as token verification.
"""
import itertools
import math

import torch

import block_verify as bv

V, G = 5, 3
N = int(__import__("os").environ.get("BV_TEST_N", "400000"))
torch.manual_seed(0)
gen = torch.Generator().manual_seed(1)
MAXLEN = 8
table_p, table_q = {}, {}


def dist(table, prefix, sharp):
    key = tuple(prefix)
    if key not in table:
        logits = torch.randn(V, generator=gen) * sharp
        table[key] = torch.softmax(logits, -1)
    return table[key]


def p_of(prefix):
    return dist(table_p, prefix, 1.5)


def q_of(prefix):
    # a draft that is correlated with the target but wrong in places
    base = torch.log(p_of(prefix)) + 0.8 * torch.randn(V, generator=gen)
    key = tuple(prefix)
    if key not in table_q:
        table_q[key] = torch.softmax(base, -1)
    return table_q[key]


def run(rule):
    out = torch.zeros(N, 3, dtype=torch.long)
    produced = torch.zeros(N, dtype=torch.long)
    seqs = [[] for _ in range(N)]
    accepted_total, steps_total = 0, 0
    while (produced < 3).any():
        act = [i for i in range(N) if produced[i] < 3]
        bs = len(act)
        X = torch.zeros(bs, G, dtype=torch.long)
        TP = torch.zeros(bs, G + 1, V)
        DP = torch.zeros(bs, G, V)
        for r, i in enumerate(act):
            pre = list(seqs[i])
            for j in range(G):
                q = q_of(pre)
                DP[r, j] = q
                TP[r, j] = p_of(pre)
                x = int(torch.multinomial(q, 1))
                X[r, j] = x
                pre.append(x)
            TP[r, G] = p_of(pre)
        if rule == "block":
            tau, corr = bv.block_accept_probs(X, TP, DP)
        else:                                            # token rule, for comparison
            ptok = TP[:, :G].gather(-1, X.unsqueeze(-1)).squeeze(-1)
            qtok = DP.gather(-1, X.unsqueeze(-1)).squeeze(-1)
            acc = torch.rand(bs, G) <= (ptok / qtok).clamp(max=1)
            tau = torch.where(acc.all(1), torch.full((bs,), G), (~acc).float().argmax(1))
            corr = torch.zeros(bs, dtype=torch.long)
            for r in range(bs):
                t = int(tau[r])
                if t == G:
                    corr[r] = int(torch.multinomial(TP[r, G], 1))
                else:
                    res = (TP[r, t] - DP[r, t]).clamp_min(0)
                    corr[r] = int(torch.multinomial(res / res.sum(), 1)) if res.sum() > 0 else int(torch.multinomial(TP[r, t], 1))
        accepted_total += int(tau.sum())
        steps_total += bs
        for r, i in enumerate(act):
            emit = X[r, :int(tau[r])].tolist() + [int(corr[r])]
            seqs[i].extend(emit)
            for t in emit:
                if produced[i] < 3:
                    out[i, produced[i]] = t
                    produced[i] += 1
    return out, accepted_total / steps_total


def chi2(out):
    counts = torch.zeros(V, V, V)
    for a, b, c in out.tolist():
        counts[a, b, c] += 1
    stat, dof = 0.0, 0
    for a, b, c in itertools.product(range(V), repeat=3):
        e = N * float(p_of([])[a] * p_of([a])[b] * p_of([a, b])[c])
        if e >= 5:
            stat += (counts[a, b, c] - e) ** 2 / e
            dof += 1
    return stat, dof - 1


if __name__ == "__main__":
    accs = {}
    for rule in ("token", "block"):
        out, acc = run(rule)
        stat, dof = chi2(out)
        z = (stat - dof) / math.sqrt(2 * dof)
        print(f"{rule:5s}: accepted drafts/step {acc:.4f}  chi2 {stat:.1f} on {dof} dof (z = {z:+.2f})", flush=True)
        assert abs(z) < 4.5, f"{rule}: output distribution differs from the target (z={z:.2f})"
        accs[rule] = acc
    assert accs["block"] > accs["token"], "block verification accepted fewer tokens than token verification"
    print("test_block_verify: ok")
