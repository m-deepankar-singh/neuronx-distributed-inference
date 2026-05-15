Use `experimental` as the base branch, but make cold prefill a **separate patch stack** on top of the hybrid APC work. The branch already has the right scaffolding: dynamic CTE profiles, `TEXT_ONLY_CTE`, `COMPACT_CTE_ATTENTION_MASK`, `COLD_ZERO_CONV_FAST_PATH`, tunable `kernel_q_tile_size` / `kernel_kv_tile_size`, block size default `128`, and hybrid APC knobs in `start_vllm_server.sh`.

The cold-prefill plan should be:

```text
experimental
  └── qwen36-cold-prefill-perf
        01 instrumentation
        02 dynamic CTE buckets
        03 text-only CTE graph
        04 compact CTE mask hardening
        05 fused GDN CTE as default
        06 safe cold-zero conv fast path
        07 attention tile sweep
        08 benchmark + acceptance gates
```

## Architecture goal

Cold prefill cannot skip the prompt. So the goal is to reduce:

```text
padded token work
dummy tensor work
dense mask work
GDN kernel overhead
conv/state overhead
attention tiling overhead
KV/block-layout overhead
```

For Qwen3.6 this is especially important because the model is dominated by GDN: the model description says 48 of 64 layers are Gated DeltaNet and 16 are standard attention layers.

---

# 1. Patch 01: cold-prefill instrumentation

Add this first. Do not optimize blind.

## Add metrics

For every cold prefill request, log:

```text
actual_prompt_len
selected_cte_bucket
padding_tokens = selected_cte_bucket - actual_prompt_len
padding_ratio
ctx_batch_size
block_size
kernel_q_tile_size
kernel_kv_tile_size
text_only_cte_enabled
compact_mask_enabled
cold_zero_conv_fast_path_enabled
use_nki_fused
prefill_latency_ms
actual_tok_per_s
bucket_tok_per_s
HBM usage if available
```

## Files

Add to:

```text
contrib/models/Qwen3.6-27B/vllm/run_offline_inference.py
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh
validation_scripts/qwen36_hybrid_apc_validation.py
```

The experimental branch already has launcher wiring for CTE buckets, ctx batch size, tile sizes, text-only CTE, compact CTE mask, and cold-zero conv fast path, so instrumentation should report those values directly from config.

## Acceptance gate

Before any optimization:

```text
Reproduce current cold prefill:
~420 tok/s cold baseline
same prompt set
same max_model_len
same compiled artifact
```

Then all later changes must compare against this baseline.

---

# 2. Patch 02: dynamic CTE bucket profiles

This is the highest-ROI cold-prefill lever.

The experimental branch already supports:

```bash
--cte-buckets
--cte-bucket-profile
```

and validates that CTE buckets are 128-aligned because DeltaNet CTE uses 128-token chunks.

## Use these profiles

### Short latency profile

```text
[128, 256, 512, 1024]
```

### General production profile

```text
[256, 512, 1024, 2048]
```

### Long context profile

```text
[4096, 8192, 16384, 32768]
```

### 262K recovery profile

```text
[256]
```

The `experimental` launcher already defines these profiles.

## Immediate experiment

Run:

```bash
git checkout experimental
git checkout -b qwen36-cold-prefill-perf

contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /path/to/Qwen3.6-27B \
  --compiled-artifacts /path/to/artifacts \
  --max-model-len 2048 \
  --seq-len 2048 \
  --cte-bucket-profile short \
  --tensor-parallel-size 4 \
  --logical-nc-config 2 \
  --max-num-seqs 1 \
  --ctx-batch-size 1 \
  --block-size 128 \
  --enable-vllm-chunked-prefill \
  --text-only-cte \
  --compact-cte-attention-mask
```

## Acceptance gate

For prompts under 512 tokens:

```text
p50 cold prefill latency improves ≥1.5x
actual tok/s improves
bucket tok/s does not regress badly
outputs match baseline
```

---

# 3. Patch 03: text-only CTE graph hardening

The experimental branch already passes empty vision tensors for text-only CTE when `use_text_only_cte_inputs` is enabled. That avoids allocating full dummy `[batch, seq, hidden]` vision embeddings in text-only serving.

But the model still has this fallback behavior:

```python
elif is_for_context_encoding and vision_embeddings.numel() > 0:
    inputs_embeds = inputs_embeds + vision_embeddings.sum() * 0
    inputs_embeds = inputs_embeds + vision_mask.sum().to(inputs_embeds.dtype) * 0
```

So the plan is:

## Keep

```python
vision_embeddings = torch.zeros((0,), dtype=torch_dtype)
vision_mask = torch.zeros((0,), dtype=torch.int32)
```

for text-only CTE.

## Add explicit assert in text-only mode

```python
if is_for_context_encoding and self.config.use_text_only_cte_inputs:
    assert vision_embeddings.numel() == 0
    assert vision_mask.numel() == 0
```

Only allow dense dummy vision tensors in a separate multimodal artifact.

## Acceptance gate

Compare:

```text
text_only_cte = 1
text_only_cte = 0
```

Expected:

```text
same output token IDs
lower cold prefill latency
lower HBM traffic / lower input staging cost
```

---

# 4. Patch 04: compact CTE attention mask hardening

The experimental branch already avoids converting 2D masks into dense 4D `[B, 1, S, S]` masks on Neuron CTE paths. It only builds dense masks when compact mode is disabled and the model is not using the Neuron CTE/block-KV path.

That is good. Harden it.

## Add guard

```python
if is_for_context_encoding and seq_length > 2048:
    assert use_compact_cte_attention_mask or use_neuron_cte_attention
```

Do not allow accidental dense mask construction for long context.

## Add test

```text
S = 4096
compact mask enabled
assert no dense causal SxS allocation path
```

## Acceptance gate

```text
2K / 8K / 32K prefill does not allocate dense SxS mask
outputs match dense fallback at small S, e.g. 256/512
```

---

# 5. Patch 05: make fused GDN CTE the default everywhere

This is already partly done on `experimental`.

The branch added `initial_state` support to `_fused_chunked_forward`, and the fused NKI kernel now accepts `initial_state` as a recurrent checkpoint or zeros.

The kernel copies `initial_state` into SBUF and keeps state there across chunks.

The model path now uses `_fused_chunked_forward(..., initial_state=initial_state)` for the chunked-prefill recurrent-state path unless explicitly forced to the older NKI chunked or PyTorch paths.

## Next work

Make this explicit and testable:

```text
USE_NKI_FUSED=1 default
USE_NKI_CHUNKED=0 default
USE_PYTORCH_CHUNK=0 default
```

Add a startup log:

```text
GDN_CTE_KERNEL=fused_initial_state
```

Add benchmark toggles:

```bash
USE_NKI_FUSED=1
USE_NKI_FUSED=0 USE_NKI_CHUNKED=1
USE_PYTORCH_CHUNK=1
```

## Acceptance gate

For cold prefill:

```text
fused_initial_state >= old chunked NKI
fused_initial_state output token IDs match baseline
GDN final_state numerical diff within tolerance
```

---

# 6. Patch 06: make cold-zero conv fast path safe

The experimental branch has `COLD_ZERO_CONV_FAST_PATH`, and the GDN layer uses a fast `F.conv1d` path when the flag is enabled.

But the current logic is too broad:

```python
cold_prefill_from_zero = getattr(self.config, "use_cold_zero_conv_fast_path", False)
```

That only checks the flag. It should also verify the prefill is truly starting from zero prefix. Otherwise a partial-prefix restore could incorrectly ignore `conv_state_cache`.

## Change it to

```python
cold_prefill_from_zero = (
    getattr(self.config, "use_cold_zero_conv_fast_path", False)
    and position_ids is not None
    and bool((position_ids[:, :1].long() == 0).all())
    and recurrent_state_cache is not None
    and conv_state_cache is not None
    and not getattr(self.config, "use_hybrid_apc_manager", False)
)
```

For hybrid APC partial-prefix restore, only use fast path when:

```python
hybrid_restore_mask is empty/zero
position_ids[:, 0] == 0
```

Do **not** use it when:

```text
position_ids[:, 0] > 0
restore_prefix_len > 0
partial prefix hit exists
decode path
```

## Add tests

```text
cold prefill position 0:
  fast conv == stateful conv

partial prefix position > 0:
  fast conv disabled

deliberately force fast conv on partial prefix:
  test should fail or output should diverge
```

## Acceptance gate

```text
cold-zero fast path improves latency
partial-prefix exactness remains intact
```

---

# 7. Patch 07: attention tile and block-size sweep

The experimental launcher exposes:

```bash
--kernel-q-tile-size
--kernel-kv-tile-size
--block-size
```

and passes the tile sizes into `chunked_prefill_config`.

## Sweep

For 2K / 8K cold prefill:

```text
q_tile=128, kv_tile=512
q_tile=128, kv_tile=1024
q_tile=128, kv_tile=2048
q_tile=256, kv_tile=1024
```

For block size:

```text
block_size=128
block_size=256
```

Do not start with 64 for cold prefill unless APC reuse is the main goal. For pure cold prefill, smaller block size can increase block metadata/layout overhead.

## Acceptance gate

Pick two production profiles:

```text
latency profile:
  block_size=128
  cte_profile=short
  ctx_batch_size=1

long-context profile:
  block_size=256 or 128, whichever loads/runs faster
  cte_profile=262k or long
  ctx_batch_size=1
```

---

# 8. Patch 08: benchmark matrix

Use a fixed prompt suite:

```text
128 tokens
256 tokens
384 tokens
512 tokens
1K tokens
2K tokens
8K tokens
32K tokens
128K tokens if artifact available
262K tokens if artifact loads
```

Run each with:

```text
temperature=0
top_k=1
max_tokens=1 for pure prefill timing
max_tokens=32 for end-to-end timing
```

## Matrix

```text
A. baseline experimental, single CTE bucket 512
B. short CTE profile [128,256,512,1024]
C. B + text-only CTE
D. C + compact CTE mask
E. D + fused GDN CTE
F. E + cold-zero conv fast path
G. F + tile sweep
```

Track:

```text
cold prefill p50/p95 latency
actual tok/s
bucket tok/s
decode tok/s
first token latency
HBM usage
artifact load success
token exactness
GDN state diff
```

---

# Exact launch commands

## 2K short-prompt cold-prefill candidate

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /path/to/Qwen3.6-27B \
  --compiled-artifacts /path/to/2k/artifacts \
  --max-model-len 2048 \
  --seq-len 2048 \
  --cte-bucket-profile short \
  --tensor-parallel-size 4 \
  --logical-nc-config 2 \
  --max-num-seqs 1 \
  --ctx-batch-size 1 \
  --block-size 128 \
  --kernel-q-tile-size 128 \
  --kernel-kv-tile-size 1024 \
  --enable-vllm-chunked-prefill \
  --text-only-cte \
  --compact-cte-attention-mask \
  --cold-zero-conv-fast-path
```

## 128K candidate

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /path/to/Qwen3.6-27B \
  --compiled-artifacts /path/to/128k/artifacts \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-buckets 256,512,1024,2048 \
  --tensor-parallel-size 4 \
  --logical-nc-config 2 \
  --max-num-seqs 1 \
  --ctx-batch-size 1 \
  --block-size 128 \
  --kernel-q-tile-size 128 \
  --kernel-kv-tile-size 1024 \
  --enable-vllm-chunked-prefill \
  --text-only-cte \
  --compact-cte-attention-mask
```

## 262K candidate

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /path/to/Qwen3.6-27B \
  --compiled-artifacts /path/to/262k/artifacts \
  --max-model-len 262144 \
  --seq-len 262144 \
  --cte-bucket-profile 262k \
  --tensor-parallel-size 4 \
  --logical-nc-config 2 \
  --max-num-seqs 1 \
  --ctx-batch-size 1 \
  --block-size 256 \
  --kernel-q-tile-size 128 \
  --kernel-kv-tile-size 1024 \
  --enable-vllm-chunked-prefill \
  --text-only-cte \
  --compact-cte-attention-mask
```

For 262K, start with `block_size=256` and `[256]` only. After it loads and runs, try `block_size=128`.

---

# What to add to `experimental` immediately

## Must add/fix

```text
1. Cold-prefill benchmark logging.
2. Safer cold-zero conv fast-path guard.
3. Unit test: fast conv equals stateful conv only at position 0.
4. Unit test: compact mask avoids dense SxS for long CTE.
5. Unit test: text-only CTE passes empty vision tensors.
6. Runtime log of selected GDN CTE kernel.
7. Benchmark script for bucket/tile/block-size matrix.
```

## Already present and should be kept

```text
dynamic CTE bucket profiles
128-aligned bucket validation
text-only CTE flag
compact CTE attention mask flag
fused GDN CTE with initial_state
tunable q/kv tile sizes
block size default 128
hybrid APC config separation
```

The launcher already wires most of those into `additional_config`, including `use_text_only_cte_inputs`, `use_compact_cte_attention_mask`, `use_cold_zero_conv_fast_path`, GDN cache dtype knobs, and Neuron `chunked_prefill_config`.

---

# Acceptance criteria

I would not call the cold-prefill patch stack successful unless it hits these:

```text
Short prompts:
  ≥1.5x p50 latency improvement vs single 512 bucket

2K prompts:
  no regression vs baseline
  exact token match

8K+ prompts:
  no dense SxS mask allocation
  stable compile/load
  no HBM spill regression

GDN:
  fused path token IDs match old path
  final recurrent_state diff bounded
  conv_state diff bounded

262K:
  TP=4 [256] artifact loads
  cold prefill runs without DMA spill-ring allocation failure
```

---

# Priority order

Build in this order:

```text
1. Instrumentation and benchmark harness
2. Dynamic CTE bucket validation
3. Text-only CTE validation
4. Compact mask hardening
5. Fused GDN CTE validation
6. Safe cold-zero conv fast path
7. Tile and block-size sweep
8. 128K / 262K load and HBM sweep
```

The most likely cold-prefill wins are:

```text
dynamic CTE buckets
text-only CTE inputs
compact CTE masks
fused GDN CTE everywhere
safe cold-zero conv fast path
```

The riskiest change is `cold-zero-conv-fast-path`, because it is only correct for true position-0 cold prefill. Keep it behind the flag until the partial-prefix tests prove it cannot activate on restored-prefix requests.


i dont hv access to instance right now but u can code in the experimental branch