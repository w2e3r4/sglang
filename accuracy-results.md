## Accuracy and reproduction results

**What was tested:** a port of this PR onto the `v0.5.23` release line, with the `forward_decode` change now included in this PR (details below). I did **not** run commit `3b59b08f` itself on GPU. All numbers below are for the ported patch.

### Setup

- 8x H800 80GB, driver 550.127.08, image `lmsysorg/sglang:v0.5.23` (torch 2.14.1+cu130, sglang-kernel 0.4.10)
- Base: `297e54fd` (release/v0.5.23 head). Patched = base + one commit (diff below).
- Target `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (rev `c5f5f263`), draft `lmsys/SGLang-EAGLE3-Qwen3-Next-80B-A3B-Instruct-FP8-SpecForge-Meituan`
- Problem config:
  ```
  python -m sglang.launch_server --model-path Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --attention-backend fa3 --kv-cache-dtype fp8_e4m3 \
    --tp 8 --dp 8 --enable-dp-attention --ep-size 8 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/SGLang-EAGLE3-Qwen3-Next-80B-A3B-Instruct-FP8-SpecForge-Meituan \
    --speculative-draft-model-quantization fp8 \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --cuda-graph-max-bs-decode 8 --max-running-requests 512 --mem-fraction-static 0.8
  ```
  `--speculative-draft-model-quantization fp8` gives the bf16 draft's attention a `k_scale`, which the FP8 descale path needs. `--cuda-graph-max-bs-decode 8` makes larger per-rank batches run eager, where the metadata is built before DP padding.

### Reproduction

- **Base:** `bench_serving --dataset-name random-ids --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.05 --num-prompts 400 --max-concurrency 200 --seed 1`. The scheduler died about 53 s after the server was ready:
  ```
  File ".../speculative/eagle_worker_v2.py", line 847, in draft_forward
  File ".../layers/attention/flashattention_backend.py", in forward_decode
  File ".../sgl_kernel/flash_attn.py", line 230, in flash_attn_with_kvcache
  RuntimeError: k_descale must have shape (batch_size, num_heads_k)
  ```
  399/400 requests failed.
- **Base + the original extend-only patch (`3b59b08f`, ported onto release; no decode change):** same config, same crash, same stack. A temporary log at the decode FA call showed `forward_batch.batch_size=25`, `cu_seqlens_q.numel()-1=24`. On this branch the first failure is in the EAGLE3 draft's multi-step **decode**, which the original extend-only patch did not touch. (This run applied the patch to the image's tree directly. I'm reporting it as supporting evidence, not as a pinned-commit run.)
- **Patched:** same config, all requests completed and no descale error appeared:

  | workload | prompts | concurrency | in/out len | succeeded |
  |---|---|---|---|---|
  | same as repro | 400 | 200 | 1024/1024 | 400 |
  | short | 800 | 256 | 128/512 | 800 |
  | long | 96 | 32 | 8000/1024 | 96 |
  | mixed | 400 | 128 | 4000/1024 | 400 |
  | GSM8K (x2 runs) | 1319 each | 256 | - | 1319 each |

### Path coverage

I ran one patched server with a temporary rate-limited log (not part of the patch). It records, at the FA call, the `forward_mode`, `forward_batch.batch_size`, `cu_seqlens_q.numel()-1` and the resulting descale shape. The log fires only when `layer.k_scale` is set (FP8 KV, `float8_e4m3fn`). Summary over the first 20000 FP8-descale FA calls on each DP rank (8 ranks, DP padding mode `MAX_LEN`):

- `DRAFT_EXTEND_V2` in `forward_extend`: 586 to 1059 calls per rank had `batch_size != segments`, e.g. 26 vs 25, 27 vs 23, 28 vs 25. The PR's re-derived descale used the segment count.
- Draft `DECODE` in `forward_decode`: 1172 to 2118 calls per rank were mismatched, e.g. 26 vs 25 and 27 vs 24.
- `TARGET_VERIFY` and normal `EXTEND`: no mismatches seen.

The accuracy runs below used the clean patched commit with no diagnostics. They used the same server flags.

### Accuracy (GSM8K, full test split)

Harness: `sgl-eval==0.1.3`, built-in gsm8k prompt, 0-shot chat, 1319 problems.
```
sgl-eval run gsm8k --base-url http://127.0.0.1:30123/v1 \
  --temperature 0.0 --top-p 1.0 --seed 0 --max-tokens 2048 --num-threads 256
```
Every run used identical eval settings. Only the server code and flags changed.

| run | code | server config | correct / total | accuracy | truncated (2048) | request errors |
|---|---|---|---|---|---|---|
| reference | base | DP8 attn, **no spec** | 1261 / 1319 | 95.60% | 7 | 0 |
| patched | patched | DP8 attn + EAGLE3 (problem config) | 1264 / 1319 | 95.83% | 5 | 0 |
| patched, rerun | patched | same | 1263 / 1319 | 95.75% | 7 | 0 |
| patched | patched | DP8 attn, no spec | 1264 / 1319 | 95.83% | 10 | 0 |
| base | base | EAGLE3, attn TP8 (no DP attn) | 1263 / 1319 | 95.75% | 8 | 0 |
| patched | patched | EAGLE3, attn TP8 (no DP attn) | 1263 / 1319 | 95.75% | 5 | 0 |

Base cannot serve the problem config, because it crashes. So the problem-config row is compared against base with speculative decoding turned off, and all other flags stay the same.

Per-sample comparisons:

| comparison | delta | ref right / cmp wrong | ref wrong / cmp right |
|---|---|---|---|
| reference vs patched (problem config) | +0.23 pp (+3) | 5 | 8 |
| base vs patched, DP8 attn, no spec | +0.23 pp (+3) | 4 | 7 |
| base vs patched, EAGLE3 without DP attn | 0.00 pp | 6 | 6 |
| patched problem config, run 1 vs run 2 (identical) | -0.08 pp (-1) | 5 | 4 |

The identical patched configuration was run twice: 9 problems changed correctness (5 right-to-wrong, 4 wrong-to-right), for a net difference of one correct answer. These runs demonstrate output variability; two repeats do not establish a noise distribution or prove equivalence. I report the measured counts and deltas without claiming a statistically established absence of regression.

### Decode change added in this PR update

Same `fa_descale_batch_size()` helper, same `forward_extend` re-derivation, same test file. One addition in `forward_decode`:

```diff
         q, q_rope, k_rope, fa_k_descale, fa_v_descale = self.prepare_paged_mha_query(
             q,
             q_rope,
             k_rope,
             layer,
-            logical_batch_size=forward_batch.batch_size,
+            # DP padding can grow forward_batch.batch_size past the segments in
+            # metadata.cu_seqlens_q, which is what FA validates the descale against.
+            logical_batch_size=fa_descale_batch_size(
+                metadata.cu_seqlens_q, forward_batch.batch_size
+            ),
             kv_head_num=layer.tp_k_head_num,
             is_prefill=False,
         )
```

The test also gets one more case: `([1] * 24, 25)`, the DP-padded draft decode shape I observed. PR commit `67a44c8639026fb35ebb79eb7bc0e9994263d7e0` adds this decode change and a mocked decode-call regression test. The GPU runs above used the same production-code change on the v0.5.23 baseline; commit `67a44c8` itself has not been rerun on GPU.

### Not verified

- Commit `3b59b08f` as-is, and current `main`. Only the v0.5.23 port was tested.
- Other models or attention variants: MLA, topk > 1 cascade, local attention, FA4, CUDA graph with per-rank batch > 8.
- Other accuracy benchmarks, and multiple seeds per config.

The accompanying reproduction bundle contains commands, logs, the tested patch, and original and normalized per-sample results. The tested release patch is `public_fix.diff` (SHA-256 `2ba3b2d6737b108f80c2c129c740e9f6fc1b92db204221ab381e7f3d88f7b928`).
