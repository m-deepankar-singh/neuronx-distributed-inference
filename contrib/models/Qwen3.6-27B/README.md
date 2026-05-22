# Contrib Model: Qwen3.6-27B

NeuronX Distributed Inference implementation of Qwen3.6-27B, a 27B parameter dense model from Alibaba Cloud with a hybrid DeltaNet + GQA attention architecture.

## Relationship to PR #140 and Qwen3.5-27B

Qwen3.6-27B is a **post-training update** of Qwen3.5-27B with improved agentic coding and thinking preservation. The models share **identical architecture** (`qwen3_5` model_type, `Qwen3_5ForConditionalGeneration`) -- only weights differ. This contrib builds on Jim Burtoft's Qwen3.6-27B contrib work in PR #140 and the shared Qwen3.5/Qwen3.6 hybrid architecture pattern.

This README reports Qwen3.6-27B validation only. Qwen3.5-27B is referenced for
architecture and code lineage; it was not re-benchmarked as part of this
Qwen3.6 contrib validation.

### Config differences from Qwen3.5-27B

| Field | Value | Impact |
|-------|-------|--------|
| `output_gate_type` | `"swish"` | **Ignored** -- not used by HF transformers or NxDI (gate uses sigmoid) |
| `language_model_only` | `false` | Informational, not used by model code |
| `bos_token_id` | `248044` | New but not architecture-relevant |
| `pad_token_id` | `null` | New at text_config level (already handled) |
| `partial_rotary_factor` | `0.25` | Already in rope_parameters, redundant copy |
| `transformers_version` | `4.57.1` | Updated from `4.57.0.dev0` |

No architecture changes are required relative to the Qwen3.5-27B hybrid
implementation. This contrib packages the NxDI Qwen3.6-27B model code,
DeltaNet NKI kernels, FP8/vLLM serving helpers, and validation coverage for the
Qwen3.6 weights.

## Model Family

| Model | HuggingFace ID | Params | Instance |
|-------|----------------|--------|----------|
| **Qwen3.6-27B** | [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) | 27B | trn2.3xlarge (TP=4) |

**License:** Apache 2.0

## Architecture Details

| Feature | Value |
|---------|-------|
| Layers | 64 (48 DeltaNet + 16 GQA) |
| Layer Pattern | [3 DeltaNet + 1 GQA] x 16 |
| Hidden Size | 5120 |
| GQA Attention | 24 heads, 4 KV heads, head_dim=256 |
| DeltaNet Attention | 48 value heads, 16 key heads, k_dim=v_dim=128 |
| Dense MLP | SwiGLU (gate_proj + up_proj: 5120 -> 17408, down_proj: 17408 -> 5120) |
| Position Encoding | Partial RoPE (25% of head_dim = 64 dims), mRoPE for VL |
| Vocabulary | 248,320 |
| Normalization | RMSNorm with +1 weight convention |
| Activation | SiLU gated MLP |

### Unique Architecture Features

- **Hybrid DeltaNet + GQA:** 48 of 64 layers use Gated DeltaNet (linear recurrent attention), 16 layers use standard GQA with KV cache. The pattern repeats every 4 layers: 3 DeltaNet + 1 GQA.
- **DeltaNet Linear Attention:** Uses the delta rule for recurrent state updates with gated decay. Per-step: `state *= exp(g); delta = (v - state^T @ k) * beta; state += outer(k, delta); output = state^T @ q`. Runs as a chunked algorithm for context encoding, per-token recurrence for token generation.
- **Custom NKI Kernels:** Three NKI kernels implement the DeltaNet forward pass on Neuron: a per-token recurrent kernel (TKG), a per-chunk kernel (legacy), and a fused single-kernel chunked forward (CTE). The fused CTE kernel uses the same direct lower-triangular intra-chunk solve strategy as the stable chunked path, with state persistence in SBUF across chunks. Earlier Neumann power-doubling experiments were not stable enough for Qwen3.6 gate scales.
- **GQA Output Gate:** Attention layers use a sigmoid output gate. `q_proj` is 2x sized and interleaved: `[head0_query | head0_gate | head1_query | ...]`. The gate is split during weight conversion and applied after attention.
- **Partial RoPE:** Only 25% of head_dim (64 of 256 dimensions) receives rotary embeddings. The remaining 192 dimensions are identity (no rotation).
- **+1 RMSNorm Convention:** HF weights use `output = norm(x) * (1 + weight)` where weight is initialized to zeros. Converted to standard `output = norm(x) * weight` during loading by adding 1.0 to all RMSNorm weights (except DeltaNet internal norms, which use standard convention).
- **Vision-Language Support:** Optional ViT encoder runs on CPU (HBM fully consumed by 27B text decoder). Vision embeddings are injected via a scatter mask at traced input positions.

## Test Results

### Unit Tests (CPU)

| Test Module | Tests | Status |
|-------------|-------|--------|
| test_config.py | 26 | 26/26 PASS |
| test_weight_conversion.py | 16 | 16/16 PASS |
| test_hybrid_cache_manager.py | 13 | 13/13 PASS |
| test_deltanet_decay.py | 2 | 2/2 PASS |
| **Total** | **57** | **57/57 PASS** |

Unit tests are architecture-level and do not depend on weights. Coverage includes config parsing, weight conversion, hybrid cache allocation/update behavior, and DeltaNet decay handling.

### Quality Validation (Qwen3.6-27B, trn2.3xlarge, TP=4)

7/7 text-only quality tests passed with `enable_thinking=False`:

| Test | Expected | Result |
|------|----------|--------|
| Speed of light | 299,792,458 m/s | PASS |
| 17 * 23 | 391 | PASS |
| 60mph * 2.5h | 150 miles | PASS |
| is_prime function | Correct Python | PASS |
| French translation | Bonjour, comment allez-vous ? | PASS |
| Capital of Japan | Tokyo | PASS |
| sqrt(144) | 12 | PASS |

## Performance Benchmarks

### Qwen3.6-27B on trn2.3xlarge (TP=4, LNC=2, BF16)

**TTFT (Time To First Token)**

| Input Length | P50 (ms) | P95 (ms) |
|-------------|----------|----------|
| 16 tokens | 305.3 | 305.6 |
| 64 tokens | 305.4 | 305.9 |
| 128 tokens | 306.6 | 306.8 |
| 256 tokens | 306.2 | 306.3 |

**TPOT / Throughput**

| Output Length | TPOT P50 (ms) | tok/s P50 | E2E P50 (ms) |
|--------------|---------------|-----------|---------------|
| 16 | 54.3 | 18.4 | 1,121 |
| 32 | 54.4 | 18.4 | 1,993 |
| 64 | 54.2 | 18.5 | 3,720 |
| 128 | 54.2 | 18.5 | 4,912 |

### Long-Context vLLM Baseline

A 128K FP8-MLP artifact was validated on trn2.3xlarge (TP=4, LNC=2)
with the vLLM Neuron plugin, Qwen chunked prefill, and native vLLM APC enabled.

| Metric | Result |
|--------|--------|
| Max model length | 131,072 tokens |
| Context encoding bucket | 512 |
| Prefill throughput | 404-428 tok/s from 512 through 64K prompt tokens |
| Decode throughput | 26.3-26.6 tok/s |
| 64K quality | needle retrieval prompts returned all expected codes |
| State reset | repeated short-after-long validation passed after 32K and 64K requests |
| Peak Neuron device memory | ~53.25 GB decimal during the 64K eval |

TTFT/TPOT details for the same 128K FP8/vLLM artifact:

| Metric | Result | Notes |
|--------|--------|-------|
| Decode TPOT | ~37.6-38.0 ms/token | Derived from 26.3-26.6 tok/s decode |
| Cold 512-token TTFT | ~1.2-1.3s | Derived from measured prefill throughput plus one decode step |
| Cold 32K-token TTFT | ~76.6-81.1s | Derived from measured prefill throughput plus one decode step |
| Cold 64K-token TTFT | ~153-162s | Derived from measured prefill throughput plus one decode step |
| Warm APC latency, ~10.8K prompt | 1.36-2.38s | Exact-repeat, partial-prefix, and cross-prefix validation runs |
| Cold APC baseline, ~10.8K prompt | 25.17-26.68s | Same prompts with prefix cache disabled or cold |

Native vLLM prefix caching/APC was also validated with exact greedy output
matches:

| APC Scenario | Cold | Warm | Speedup | Result |
|--------------|------|------|---------|--------|
| Server exact-repeat, ~10.8K prompt tokens | 26.68s | 1.67s | 16.0x | exact text match |
| Offline exact-repeat | 26.19s | 2.38s | 11.0x | exact token-ID match |
| Offline partial-prefix reuse | 25.52s | 1.70s | 15.0x | exact token-ID match |
| Server cross-prefix reuse | 25.17s | 1.36s | 18.5x | exact text match |

### Fused Direct-Solve Validation

The fused DeltaNet CTE kernel was revalidated with a direct triangular solve
instead of the earlier Neumann power-doubling correction. Detailed results and
raw result-file references are recorded in
[`docs/fused_directsolve_validation_20260522.md`](docs/fused_directsolve_validation_20260522.md).

| Metric | Result | Notes |
|--------|--------|-------|
| Coherence | PASS | Fact, code, and prefix-cache prompts produced real non-repetitive text with `enable_thinking=false` |
| Decode throughput | 21.63 tok/s | Offline vLLM/NxDI path, on-device greedy sampling, 128-token decode |
| Decode TPOT | 46.2 ms/token | Same run as decode throughput |
| Cold 512-token TTFT | 1.31s | 390 tok/s cold prefill |
| Cold 16K-token TTFT | 27.84s | 589 tok/s cold prefill |
| Warm 16K-token TTFT | 0.45s | 36.3K tok/s effective warm prefill |
| Peak Neuron HBM | 60.1 GiB | Sum across logical cores on trn2.3xlarge |

The validated direct-solve artifact used prefix buckets through 16K. A 32K
prompt exceeded that artifact's largest prefix bucket, so longer-context fused
validation requires recompiling the same code with larger prefix buckets.

Memory note: the `60.1 GiB` direct-solve number is `64.54 GB` decimal and is a
Neuron high-water peak from the Hybrid APC artifact, not a prompt-length-only
16K allocation. The artifact was compiled with `pa_num_blocks=512`,
`max_gdn_checkpoint_slots=64`, and token-generation buckets
`[8192, 32768, 131072]`. The 64-slot GDN checkpoint bank alone is expected to
reserve about `9.85 GB` decimal across TP=4 ranks
(`38.49 MB/checkpoint/rank * 64 * 4`), which explains most of the gap from the
PR 164 vLLM/APC README's `~53.25 GB` decimal 64K eval number. Treat the higher
direct-solve HBM as a Hybrid APC artifact/config observation requiring
like-for-like A/B validation, not as proof that the direct triangular solve
itself increases memory.

### Hybrid APC Follow-up Status

Follow-up work on the `experimental` branch extended the baseline vLLM/APC
path toward Qwen3.6 Hybrid APC, where attention KV prefix reuse is only correct
when the matching DeltaNet recurrent/conv checkpoint is also available.

What has been implemented and proven in that branch:

- Scheduler-side safety gating prevents vLLM from reading an attention prefix
  unless a matching GDN checkpoint is registered.
- Qwen request prep consumes scheduler-authorized, request-scoped restore keys
  instead of relying on prefix length alone.
- The CTE restore path handles suffix-only execution over a restored prefix:
  suffix tokens, slot mapping, `computed_context_lens`, `num_queries`, and GDN
  restore metadata are kept aligned.
- BF16 single-request backed-prefix validation passes with cold/warm exactness
  on the 2K checkpoint-boundary case. The proven shape restores a 256-token GDN
  checkpoint, executes a 16-token suffix, and matches cold output.
- The safety fallback also passes: if attention KV has a prefix hit but no GDN
  checkpoint exists, prefix reads are disabled and the request recomputes cold.

Current blocker:

- True generated-token batch-2 validation needs both `tkg_batch_size=2` and
  `ctx_batch_size=2`.
- A batch-2 artifact with `tkg_batch_size=2` but `ctx_batch_size=1` failed in
  vLLM-Neuron host-logits sampling because two prefills were packed into one
  CTE row, then logits were reordered for two live request ids.
- Single-bucket `ctx_batch_size=2` / `tkg_batch_size=2` BF16 artifacts for CTE
  bucket 256 and CTE bucket 512 compiled successfully.
- The combined multi-bucket artifact (`cte_buckets=256,512`,
  `prefix_buckets=256,512`) started compiling and the TKG priority HLO passed,
  but the smaller Trainium instance became SSH-unresponsive during all-HLO CTE
  compilation. This appears to be a Neuron/NxDI compile-capacity or compile
  orchestration issue, not a model-correctness failure.

Expected outcome after the batch-2 artifact or an equivalent prefill-only
proof is available:

- Batched Hybrid APC can preserve the same correctness rule as the
  single-request path:
  `usable_prefix_hit = attention_KV_prefix_hit AND matching_GDN_checkpoint_hit`.
- Warm repeated-prefix and partial-prefix requests should avoid replaying the
  shared cold prefill while restoring the required GDN state.
- This is the path expected to turn the current exact single-request APC proof
  into a measured cold-prefill performance win for batched serving.

The fused CTE kernel is no longer blocked on the earlier Neumann-series NaN
failure: the direct-solve fused path passed the coherence and performance
validation summarized above. The remaining Hybrid APC follow-up is serving
contract coverage, especially generated-token batch-2 validation with matching
`ctx_batch_size=2` and `tkg_batch_size=2` artifacts.

### Key Observations

- **BF16 TP=4 is HBM-limited:** The pure BF16 path is limited to short contexts on trn2.3xlarge. The validated 128K baseline uses MLP-only FP8 weights plus the hybrid cache manager.
- **DeltaNet enables efficient TKG:** Token generation uses O(1) per-token recurrence instead of O(n) KV cache attention for 48/64 layers.
- **vLLM APC is high leverage:** Repeated-prefix requests avoid replaying long chunked prefill and are the largest observed latency win for chat/RAG-style workloads.
- **Qwen3.6-only measurements:** The benchmark tables above are Qwen3.6 results. Qwen3.5 is referenced only because the two contrib models share the same architecture and implementation lineage.

## Usage

### Text-Only (trn2.3xlarge, TP=4)

```python
import json
import torch
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronQwen35ForCausalLM

model_path = "/path/to/Qwen3.6-27B"
compiled_path = "/scratch/qwen36_traced/"

neuron_config = NeuronConfig(
    tp_degree=4,
    batch_size=1,
    ctx_batch_size=1,
    tkg_batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
    logical_nc_config=2,
    enable_bucketing=False,
    flash_decoding_enabled=False,
    on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    save_sharded_checkpoint=True,
)

# Read config.json directly (model_type 'qwen3_5' may not be
# registered in all transformers versions)
import os
with open(os.path.join(model_path, "config.json")) as f:
    hf_config = json.load(f)
text_config = hf_config.get("text_config", hf_config)
config_dict = dict(text_config)
config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
config_dict.setdefault("tie_word_embeddings", False)

config = Qwen35InferenceConfig(
    neuron_config=neuron_config,
    **config_dict,
)

model = NeuronQwen35ForCausalLM(model_path, config)
model.compile(compiled_path)

# Reload from compiled artifacts
model = NeuronQwen35ForCausalLM(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
gen_config = GenerationConfig(
    do_sample=True, top_k=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

inputs = tokenizer("The capital of France is", return_tensors="pt")
gen_model = HuggingFaceGenerationAdapter(model)
outputs = gen_model.generate(
    inputs.input_ids,
    generation_config=gen_config,
    attention_mask=inputs.attention_mask,
    max_new_tokens=50,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Vision-Language (trn2.3xlarge, TP=4)

The VL pipeline uses the text decoder on Neuron and the vision encoder on CPU:

```python
from src.modeling_qwen35_vl import NeuronQwen35VLForCausalLM, Qwen35VLInferenceConfig

vl_model = NeuronQwen35VLForCausalLM(
    model_path="/path/to/Qwen3.6-27B",
    config=vl_config,
)
vl_model.compile(compiled_path)
vl_model.load(compiled_path)

# See test/integration/test_model.py for full VL usage example
```

### DeltaNet Kernel Selection

The DeltaNet forward path can be controlled via environment variables:

| Env Var | Forward Path | Use Case |
|---------|-------------|----------|
| `USE_NKI_FUSED=1` | Fused chunked NKI kernel | Best CTE performance in the validated build |
| `USE_NKI_CHUNKED=1` | Per-chunk NKI kernel | Legacy, superseded by fused |
| `USE_NKI=1` | Per-token NKI kernel | TKG (always used for token generation) |
| `DELTANET_SEQUENTIAL=1` | Sequential PyTorch | Debugging/reference |
| *(none)* | PyTorch chunked | Default fallback for CTE |

## Caveats

1. **BF16 HBM pressure at TP=4:** The pure BF16 model consumes nearly all HBM on trn2.3xlarge. Use the FP8/vLLM path for the validated 128K artifact, or a larger instance for additional batching/headroom.

2. **NKI 0.3.0+ required:** The NKI DeltaNet kernels were validated with the `nki` package version shown in the package table below. No library modifications were needed in the validated NxDI venv (`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`).

3. **No mini model test:** Unlike DeepSeek-V3, a mini model cannot be provided because DeltaNet layers require NKI kernels that only execute on Neuron devices. Integration tests require a trn2 instance with the full 27B weights.

4. **Vision encoder runs on CPU:** The ViT cannot be placed on Neuron because HBM is fully consumed by the text decoder. This adds ~918ms latency per image. Future optimization: quantize text decoder to free HBM, or use larger instance.

5. **Compilation time:** The short-context BF16 path compiles in roughly 13 minutes. The validated 128K FP8/vLLM artifact takes longer because it includes long-context cache shapes and presharded checkpoints.

6. **+1 RMSNorm convention:** Qwen3.5/3.6 uses `output = norm(x) * (1 + weight)` for most RMSNorm layers, but DeltaNet internal norms use standard `output = norm(x) * weight`. The weight conversion handles this automatically, but custom weight loading must be aware of both conventions.

7. **DeltaNet numerical stability:** DeltaNet kernels rely on normalized Q/K inputs and bounded decay handling. The chunked and fused CTE paths now use direct triangular solves for intra-chunk correction; changes to either path should be validated against the CPU reference, the fused NKI validator, and long-context stress prompts.

8. **Shared codebase with Qwen3.5-27B:** This contrib uses the same `Qwen35*` class names and `modeling_qwen35*.py` filenames as the [Qwen3.5-27B contrib](../Qwen3.5-27B/). This is intentional -- both models share the `qwen3_5` model_type. The code is identical; only the HuggingFace model ID and weights differ.

## Maximum Sequence Length

| seq_len | Path | Status | Notes |
|---------|------|--------|-------|
| 128 | BF16 NxDI | **PASS** | BF16 baseline/quality checks |
| 256 | BF16 NxDI | **PASS** | BF16 benchmark bucket |
| 512 | BF16 NxDI | **PASS** | 4 DeltaNet chunks |
| 65,536 | FP8/vLLM | **PASS** | chunked prefill, quality, and state-reset validation |
| 131,072 | FP8/vLLM | **PASS** | compiled and served with 512-token CTE bucket |

For production long-context serving on trn2.3xlarge, use the FP8/vLLM artifact
and 512-token context encoding bucket. Larger instances are recommended for
larger batches or additional serving headroom.

The fused direct-solve artifact summarized in this README was compiled with
prefix buckets through 16K. It is suitable for fused-kernel correctness and
short-to-mid-context performance validation, but not for proving 64K/128K warm
APC behavior without recompilation.

## Compatibility Matrix

| Instance | TP | LNC | Status | Notes |
|----------|-----|-----|--------|-------|
| trn2.3xlarge | 4 | 2 | **PASS** | BF16 short-context and FP8 128K vLLM/APC validated |
| trn2.48xlarge | 4 | 2 | Expected PASS | Untested for this contrib; use the same TP=4 artifact shape when compiling for trn2.3xlarge deployment |
| trn2u.48xlarge | 4 | 2 | Expected PASS | Untested for this contrib; same portability caveat as trn2.48xlarge |

### Validation Host Package Versions

These versions were checked on a running trn2.3xlarge validation host on
2026-05-22. Treat them as the observed validation environment, not as a generic
SDK release label.

| Component | Version |
|-----------|---------|
| neuronx-distributed-inference | 0.9.17334+ced6ae4e |
| neuronx-distributed | 0.18.27753+1cafd54f |
| neuronx-cc | 2.24.8799.0+6f62ff7c |
| nki | 0.3.0+23928721754.g18aa1271 |
| torch | 2.9.1 |
| torch-neuronx | 2.9.0.2.13.26312+8e870898 |
| torch-xla | 2.9.0 |
| transformers | 4.57.6 |
| aws-neuronx-runtime-lib | 2.31.24.0-0b044f4ce |
| aws-neuronx-tools | 2.29.22.0-b486b0ade |
| NXDI venv | `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` |

## Testing

### Unit Tests (CPU only, no device needed)

```bash
cd contrib/models/Qwen3.6-27B/
# On DLAMI: source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pytest test/unit/ -v
```

Tests: config parsing (26), weight conversion (16), hybrid cache manager (13), and DeltaNet decay handling (2) = **57 tests**.

### Integration Tests (needs trn2.3xlarge with 4 NeuronCores)

```bash
cd contrib/models/Qwen3.6-27B/

QWEN35_MODEL_PATH=/mnt/models/Qwen3.6-27B \
QWEN35_COMPILED_PATH=/mnt/models/qwen36_traced \
pytest test/integration/test_model.py --capture=tee-sys
```

Tests: model loads, generates, coherence, top-token valid, capital test, TTFT, throughput, multi-prompt = **8 tests**.

Note: The env var is `QWEN35_MODEL_PATH` (not `QWEN36`) because the code uses the `qwen3_5` model_type internally.

## Example Checkpoints

- [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) (BF16, ~52 GB)

**Last Updated:** 2026-05-22
