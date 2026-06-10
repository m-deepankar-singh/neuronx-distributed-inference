Decode-speed improvement plan

The plan should target two separate metrics:

single-stream decode tok/s:
  current ~26–27 tok/s

aggregate output tok/s:
  currently flat under concurrency because baseline v3 is compiled for max_num_seqs=1

Baseline v3 is already strong for APC and prefill reuse, but it is not throughput-scaled: the report shows 26.3–26.6 decode tok/s, strong prefix-cache reuse, and a known caveat that the artifact is compiled with max_num_seqs=1, so concurrent requests queue.

MTP v4 proves the model can reach 44–48 tok/s, but that path uses native MTP/speculation and was validated with prefix caching disabled, so it should stay parked for now.

So the non-spec target should be:

Near-term:
  27 -> 32–36 tok/s

Stretch without MTP/speculation:
  36–40 tok/s

Aggregate output tok/s:
  improve via max_num_seqs>1 or replicas, not client concurrency alone
Build one dedicated branch

Create a clean branch from baseline v3 / contrib/qwen36-27b-vllm-apc-pr:

git checkout contrib/qwen36-27b-vllm-apc-pr
git checkout -b qwen36-v3-decode-fastpath

Do not base this on MTP v4. The MTP branch adds accepted-token cache state, draft model, target/draft fused-spec paths, and step-state outputs. Those are useful later but are not a clean non-spec decode optimization.

Cherry-pick only these ideas:

from cold-prefill-perf:
  metrics and benchmarking harness

from experimental:
  nothing enabled by default for decode

from MTP v4:
  only use its 44–48 tok/s result as a ceiling reference
Phase 0 — Measure decode correctly

First, add a proper decode benchmark so every change is attributable.

The current vLLM offline runner prints elapsed time but does not split prefill and decode cleanly. It constructs the Neuron config with context_encoding_buckets, token_generation_buckets, dtype, chunked-prefill options, and prefix-cache options, but not a detailed decode metrics path.

Port the metrics style from qwen36-cold-prefill-perf:

GENERATION_METRICS
  prompt_tokens
  completion_tokens
  first_token_latency
  steady_decode_seconds
  decode_tok_s
  end_to_end_tok_s
  selected_tkg_bucket
  output_logits
  on_device_sampling
  async_mode
  max_num_seqs
  block_size
  prefix_cache_hit

Run every benchmark in two modes:

cold prompt:
  includes prefill

warm APC prompt:
  isolates decode path after prefix cache hit

Use this matrix:

prompt lengths:
  32, 512, 2K, 8K, 32K, 64K, 128K

completion:
  128 and 256 tokens

sampling:
  greedy only: temperature=0, top_k=1

Acceptance gate:

Exact token IDs must match baseline v3 for greedy generation.
No APC correctness regression.
Phase 1 — Fix the serving/logits path first

This is the highest-ROI first step.

Problem

The vLLM launcher currently sets the basic Neuron config but does not expose a greedy on-device sampling fast path. It sets torch_dtype, context_encoding_buckets, token_generation_buckets, seq_len, max_length, etc., but not on_device_sampling_config, async_mode, or token-specific output/logit controls.

In the model, lm_head uses:

gather_output=False if self.on_device_sampling else True

So if on_device_sampling is off, the model is more likely to gather full vocabulary logits. The base model already supports on-device sampling and can return int32 token IDs instead of full logits.

Change

Add a greedy fast path:

--enable-on-device-sampling
--output-logits false

Inject this into the Neuron config:

neuron_config.update({
    "on_device_sampling_config": {
        "do_sample": False,
        "top_k": 1,
        "top_p": 1.0,
        "temperature": 1.0,
    },
    "output_logits": False,
})

Serving rule:

temperature=0 / top_k=1 / no logprobs:
  use token fast path

sampling / logprobs / guided decoding / logits processors:
  fall back to logits path

Expected gain:

27 tok/s -> 30–34 tok/s

This depends on whether the current vLLM path is paying full-logit gather/sync overhead. If it already avoids that internally, the gain will be smaller.

Phase 2 — Enable non-spec async decode

NxDI has an async execution path that can run one token ahead when sequence IDs remain stable and bucket boundaries are not hit. It supports the non-spec prefix-caching path, while explicitly rejecting non-EAGLE fused speculation with prefix caching in async mode. The continuation path reuses prior ranked outputs and feeds them into the next token step.

Change

Add:

--async-mode

Inject:

neuron_config.update({
    "async_mode": True,
})

Use only for:

non-spec decode
stable seq_ids
greedy token fast path
no MTP
no EAGLE

Expected gain:

+5–15%

If device execution fully dominates, this will be small. If CPU sync/token handoff is visible, it will help.

Phase 3 — Add token-generation buckets

Right now the vLLM launcher sets:

"token_generation_buckets": [SEQ_LEN]
"enable_bucketing": False

That means the decode graph is effectively compiled around the maximum context length. For a 128K artifact, a 2K or 8K chat can still pay a large TKG shape.

Change

Enable TKG buckets.

For 128K:

token_generation_buckets:
  [4096, 8192, 16384, 32768, 65536, 131072]

Start simpler:

[8192, 32768, 131072]

For 262K:

[8192, 32768, 131072, 262144]

Add launcher flag:

--token-generation-buckets 8192,32768,131072

Inject:

neuron_config.update({
    "enable_bucketing": True,
    "token_generation_buckets": [8192, 32768, 131072],
})

Expected gain:

short/medium context decode:
  meaningful

near-max context decode:
  little or none

Acceptance:

No output drift.
No artifact load failure.
No APC regression.
Phase 4 — Remove avoidable GDN decode overhead

Qwen3.6 decode is not attention-only. Most layers are GDN/DeltaNet, and every token must update both recurrent state and conv state.

The current decode path performs causal conv state handling, then recurrent update. The baseline branch uses _recurrent_step for single-token decode. The MTP branch refactors this into _recurrent_decode_forward to support multiple decode steps and optional step-state outputs.

4A. Conv decode fast path

For seq_len == 1, replace the looped conv step with a direct fused expression.

Current pattern:

conv_out = torch.zeros_like(mixed)
for k in range(4):
    conv_out = conv_out + w[:, k].unsqueeze(0).unsqueeze(-1) * conv_input[:, :, k:k+1]

Fast path:

if is_decode and seq_len == 1:
    # conv_input: [B, conv_dim, 4]
    # w:          [conv_dim, 4]
    conv_out = (conv_input[:, :, :4] * w.unsqueeze(0)).sum(dim=-1, keepdim=True)
    mixed_post_conv = F.silu(conv_out)
else:
    existing_path()

Expected gain:

small, probably 1–3%

But it is safe and easy.

4B. Disable MTP step-state outputs in non-MTP mode

The MTP branch adds recurrent/conv step-state outputs for accepted-token cache selection. That is required for MTP, but it is overhead for normal decode. Keep this off:

enable_mtp_step_state_output = False
return_deltanet_step_states = False

The MTP branch appends these state tensors only when enable_mtp_step_state_output is set.

4C. Avoid unnecessary state dtype churn

Current GDN decode casts cached recurrent state to float for math, then casts the new state back to the recurrent buffer dtype.

For speed experiments, test:

A. bf16 recurrent cache, fp32 compute
B. fp32 recurrent cache, fp32 compute

Expected:

bf16 cache:
  lower HBM traffic, likely faster

fp32 cache:
  maybe better exactness, maybe slower

For the decode-speed branch, keep bf16 unless exactness tests force fp32.

Phase 5 — Build a GDN TKG NKI kernel

This is the highest-ROI model kernel project after serving-path fixes.

Current decode does many small operations per GDN layer:

l2norm(q)
l2norm(k)
query scale
exp(g)
state decay
kv memory reduction
delta update
state update
output reduction
state writeback

A fused GDN token-generation kernel should take:

query
key
value
g
beta
recurrent_state

and produce:

output
new_recurrent_state

Fuse:

l2norm q/k
scale
decay
state update
output projection input

Do this in stages:

1. one GDN layer, batch=1, seq=1
2. all GDN layers, batch=1, seq=1
3. batch=2/4
4. optional seq>1 path for future multi-token verification

Expected gain:

+10–25% if GDN recurrence dominates decode

This is the path most likely to move:

32–36 tok/s -> 36–40 tok/s

without speculative decoding.

Phase 6 — Attention decode experiments, but do not lead with them

The config exposes token-generation attention kernel flags, but there are compatibility constraints: attention TKG kernels do not yet support contexted/chunked prefill, and transposed K cache is incompatible with block KV cache.

So do this only as an ablation:

Artifact A:
  vLLM APC/block-KV path, current attention decode

Artifact B:
  no APC/block-KV, attention TKG kernel enabled

Purpose:

Measure the attention-decode ceiling.
Do not ship B if it gives up APC.

If B is much faster, the longer-term project is:

block-KV/APC-compatible head_dim=256 attention TKG kernel

But I would not start here. Qwen3.6 has only 16 full-attention layers out of 64; GDN and MLP are likely more important for decode.

Phase 7 — Improve aggregate output tok/s

The current system is correct under concurrent client load, but throughput does not increase because requests queue behind max_num_seqs=1. The concurrency report shows aggregate completion throughput stays flat while tail latency scales linearly.

You have two choices.

Option A — Compile max_num_seqs=2 and 4

Add:

--max-num-seqs 2
--max-batch-size 2
--token-generation-batches 1,2

Config:

neuron_config.update({
    "batch_size": max_num_seqs,
    "ctx_batch_size": 1,
    "tkg_batch_size": max_num_seqs,
    "max_batch_size": max_num_seqs,
    "kv_cache_batch_size": max_num_seqs,
    "token_generation_batches": [1, 2],
})

Then test:

max_num_seqs=2
max_num_seqs=4

Expected:

single-stream tok/s:
  may drop slightly

aggregate output tok/s:
  should improve if batched TKG works efficiently

Risk:

hybrid GDN state isolation under continuous batching
HBM growth
APC + block-table + GDN cache correctness
Option B — Multiple single-sequence replicas

This is safer and probably faster to production:

Run N replicas of max_num_seqs=1
Put them behind a load balancer
Keep APC per replica

Expected:

single-stream decode remains ~same
aggregate output tok/s scales with replicas
least risk to GDN cache correctness

If your production goal is immediate output tok/s, I would do replicas before deep hybrid continuous batching.

Branch-specific use
Baseline v3

Use as the base.

Keep:

vLLM APC
chunked prefill
mamba/GDN cache mode align
block_size=256 initially
MLP-only FP8 artifact if already used
MTP v4

Do not merge for now.

Use only as:

decode ceiling reference: 44–48 tok/s
future MTP/APC integration source

MTP validation explicitly used prefix caching disabled for isolation, so it is not the current production APC path.

Experimental branch

Do not use for decode speed.

Use later for:

hybrid APC correctness
GDN checkpoint restore/commit
production prefix-cache architecture
Cold-prefill branch

Port only:

metrics
benchmark scripts
launcher validation
bucket logging
guardrails

Do not expect it to improve decode. It is mainly prefill work.

Concrete implementation checklist
start_vllm_server.sh

Add:

--enable-on-device-sampling
--async-mode
--output-logits false
--token-generation-buckets 8192,32768,131072
--token-generation-batches 1,2
--max-batch-size 2

Add to config JSON:

if enable_on_device_sampling:
    neuron_config["on_device_sampling_config"] = {
        "do_sample": False,
        "top_k": 1,
        "top_p": 1.0,
        "temperature": 1.0,
    }
    neuron_config["output_logits"] = False

if async_mode:
    neuron_config["async_mode"] = True

if token_generation_buckets:
    neuron_config["enable_bucketing"] = True
    neuron_config["token_generation_buckets"] = token_generation_buckets

if token_generation_batches:
    neuron_config["token_generation_batches"] = token_generation_batches
run_offline_inference.py

Add:

prefill/decode split
decode tok/s
first-token latency
steady decode latency
selected bucket
on-device sampling flag
async flag
modeling_qwen35.py

Add:

seq_len == 1 GDN conv fast path
optional GDN TKG NKI kernel switch
assert MTP step states disabled in non-MTP
vLLM/NxDI adapter

Add request routing:

greedy/no-logprobs:
  use on-device token fast path

everything else:
  use full logits path

This is important because vLLM often expects logits for sampling/logprobs; the fast path must be conditional.

Benchmark order

Run this exact order.

Run 1 — baseline
v3 as-is
Run 2 — token fast path
v3 + on_device_sampling + output_logits=False
Run 3 — async
v3 + on_device_sampling + async_mode
Run 4 — TKG buckets
v3 + on_device_sampling + async_mode + token_generation_buckets
Run 5 — conv fast path
same + GDN seq_len=1 conv fast path
Run 6 — batch throughput
max_num_seqs=2
max_num_seqs=4
Run 7 — GDN TKG kernel
same + fused GDN decode kernel
Expected result ladder
Change	Single-stream decode	Aggregate output tok/s
Baseline v3	26–27 tok/s	Flat under concurrency
On-device token fast path	30–34 tok/s if logits overhead exists	Same
Async non-spec decode	+5–15%	Same
TKG buckets	Better short/medium contexts	Same
GDN conv fast path	+1–3%	Same
GDN TKG NKI kernel	+10–25% possible	Helps all streams
max_num_seqs=2/4	May reduce per-stream slightly	Improves aggregate
Multiple replicas	Same per-stream	Scales aggregate
MTP v4	44–48 tok/s observed	Defer; not APC-ready
Final recommendation

Build decode speed in this order:

1. Greedy on-device token fast path
2. Async non-spec decode
3. Token-generation buckets
4. GDN decode conv fast path
5. max_num_seqs=2 artifact or replicas for output tok/s
6. GDN TKG NKI kernel
7. Attention TKG/block-KV experiment

Do not spend this cycle on:

MTP + APC integration
EAGLE
speculative decoding
FP8 GDN state
hybrid APC rewrite
cold-prefill kernel work

The first milestone should be:

27 tok/s -> at least 32 tok/s
with exact token match and no APC regression

The second milestone should be:

aggregate output tok/s scales beyond one active sequence
either through max_num_seqs=2 or multiple replicas