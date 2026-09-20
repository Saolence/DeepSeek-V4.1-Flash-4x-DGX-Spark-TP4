"""Install storage adapter in every serving worker, only when explicitly enabled."""
import importlib.abc
import importlib.machinery
import os
import sys

class EngramLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__ == 'sglang.srt.layers.engram':
            from engram_backend import install
            install(module)
            # After engram_backend: Engram rows fetched on a side stream right after the
            # hasher. Gated on DSV41_ENGRAM_PREFETCH, inactive by default.
            from engram_prefetch import install as install_engram_prefetch
            install_engram_prefetch(module)
        elif module.__name__ == 'sglang.srt.layers.quantization.fp8_utils':
            from mxfp8_b12x import install
            install(module)
            # AFTER b12x, so this wraps its wrapper rather than the original.
            # Gated on DSV41_SHARED_PAD_K, inactive by default.
            from shared_pad_k import install as install_shared_pad
            install_shared_pad(module)
        elif module.__name__ == 'sglang.srt.layers.quantization.fp8':
            from mxfp8_b12x import install_fp8
            install_fp8(module)
            from shared_pad_k import install_fp8 as install_shared_pad_fp8
            install_shared_pad_fp8(module)
        elif module.__name__ == 'sglang.srt.model_executor.model_runner':
            from prefill_empty_cache import install
            install(module)
        elif module.__name__ == 'sglang.srt.layers.attention.deepseek_v4_backend':
            # sglang#39187 backport: dense prefill indexer scored in bounded row chunks.
            # Gate checked BEFORE the import so a disabled flag imports nothing. v1 targets
            # the dev-dsv41 image backend (self.candidate_masks); v2 is the PR verbatim for
            # candidate_metadata backends (dsv4.1 branch >= f80c91a4b). Each refuses the other.
            if os.environ.get('DSV41_INDEXER_CHUNKED', '0').strip() not in ('0', 'off', 'false', ''):
                import inspect as _inspect
                _src = _inspect.getsource(module.DeepseekV4AttnBackend._low_ratio_index_topk_dense)
                if 'self.candidate_masks' in _src:
                    from indexer_chunked import install as install_indexer_chunked
                else:
                    from indexer_chunked_v3 import install as install_indexer_chunked
                install_indexer_chunked(module)
        elif module.__name__ == 'sglang.srt.entrypoints.openai.encoding_dsv41':
            from encoding_compat import install_encoder
            install_encoder(module)
        elif module.__name__ == 'sglang.srt.entrypoints.openai.serving_chat':
            from encoding_compat import install_serving_chat
            install_serving_chat(module)
        elif module.__name__ == 'sglang.srt.model_loader.weight_utils':
            # Gated on DSV41_FAST_LOAD: this rank's tensors are read eagerly by a thread
            # pool instead of page-faulted through the loader's mmap (adapter/fast_load.py).
            from fast_load import install_weight_utils
            install_weight_utils(module)
        elif module.__name__ == 'sglang.srt.models.deepseek_v4':
            from fast_load import install_deepseek_v4
            install_deepseek_v4(module)
        elif module.__name__ == 'sglang.srt.models.deepseek_v4_dspark':
            from fast_load import install_dspark
            install_dspark(module)
        elif module.__name__ == 'sglang.srt.speculative.dspark_components.dspark_verify':
            # Tap for offline draft training data. Gate checked BEFORE the import, so a disabled
            # flag imports nothing. Wraps TargetVerifyExecutor.commit_hidden; capture is switched
            # at runtime by the presence of DSV41_DRAFT_CAPTURE_TRIGGER, no restart needed.
            if os.environ.get('DSV41_DRAFT_CAPTURE', '0').strip() not in ('0', 'off', 'false', ''):
                from draft_capture import install as install_draft_capture
                install_draft_capture(module)
        elif module.__name__ == 'sglang.srt.managers.schedule_batch':
            from loop_abort import install as install_loop_abort
            install_loop_abort(module)
        else:
            # V4.1 ratio-1/2 indexers always call the FP4 DeepGEMM kernel.
            # SM120 needs its split-128 planner even when the legacy FP8
            # indexer uses the torch path. The upstream guard misses this case.
            cls = module.PagedIndexerMetadata
            original = cls.__post_init__
            def post_init(self):
                sm12 = bool(getattr(module, '_IS_SM120', False) or
                            getattr(module, '_IS_SM121', False))
                if sm12 and self.compress_ratio in (1, 2):
                    self.force_deep_gemm_metadata = True
                original(self)
            cls.__post_init__ = post_init

class EngramFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ('sglang.srt.layers.engram',
                            'sglang.srt.layers.quantization.fp8_utils',
                            'sglang.srt.layers.quantization.fp8',
                            'sglang.srt.model_executor.model_runner',
                            'sglang.srt.layers.attention.deepseek_v4_backend',
                            'sglang.srt.entrypoints.openai.encoding_dsv41',
                            'sglang.srt.entrypoints.openai.serving_chat',
                            'sglang.srt.model_loader.weight_utils',
                            'sglang.srt.models.deepseek_v4',
                            'sglang.srt.models.deepseek_v4_dspark',
                            'sglang.srt.managers.schedule_batch',
                            'sglang.srt.layers.attention.dsv4.metadata'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = EngramLoader(spec.loader)
        return spec

if os.environ.get('DSV41_SOURCE'):
    sys.meta_path.insert(0, EngramFinder())
    try:
        import tp3_pad
        tp3_pad.install()
    except Exception as exc:
        print(f'DSV41 TP pad not installed: {exc}', flush=True)
