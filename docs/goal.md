# Qwen3.6 Trainium Production Readiness Goal

Date: 2026-05-17
Branch: experimental

This document defines the practical end state for Qwen3.6-27B on Trainium
through NxDI + vLLM, plus the shortest correctness ladder from the current
Hybrid APC ABI/sampling failures to production readiness.

## Current Fix Assessment

The immediate fix is **short conceptually**, but **not necessarily easy** because the bug is in the Neuron/vLLM positional ABI boundary.

### My honest view

**Fix size:** likely small to medium
**Difficulty:** medium
**Risk:** high if patched casually

You probably do **not** need to redesign the model or kernels. But you do need to be very precise.

The fix is mainly:

```text
Do not use one shared expanded argument list for both CTE and TKG.

CTE must always receive the exact 24-tensor list it was compiled with.
TKG must receive its own exact list — 24 or 29, depending on the compiled TKG graph.
```

So the patch may only touch:

```text
Qwen35ModelWrapper.input_generator()
Qwen35ModelWrapper.pad_inputs()
NeuronQwen35ForCausalLM._get_model_outputs()
```

Possibly also a small helper for arg-list construction.

### Why it is not “easy”

Because even one wrong positional slot can produce:

```text
NaN logits
dummy token 0
NRT_EXEC_OOB
forward expected 25 args but received 30
```

The code may still compile, but fail only at runtime or first decode. That makes it annoying.

### Fastest safe fix

Add two explicit builders:

```python
build_cte_args(...)  # always 24 tensors
build_tkg_args(...)  # exact TKG tensor count
```

Then hard assert before invocation:

```python
assert len(cte_args) == 24
assert len(tkg_args) == 29  # if that is your compiled TKG contract
```

Also dump shape/dtype/name for each argument before compile and runtime.

### Estimate in practical terms

If you know the expected TKG ordering, it could be a **1–2 file fix**.

If the expected TKG ordering is still unclear, it becomes a **debugging task**, because you must first confirm the exact runtime order expected by vLLM Neuron and the compiled TKG NEFF.

### Verdict

**Not a big rewrite.**
**Not a kernel problem.**
But **not a blind 10-minute fix either**.

It is a careful ABI alignment fix.

Yes. Add a clear **final goal / production-readiness test plan** to the branch or PR so you do not only fix the current ABI bug, but also know when Qwen3.6 on Trainium is actually ready.

## Final goal

```text
Goal: Run Qwen3.6-27B on Trainium through NxDI + vLLM with correct CTE/TKG contracts, stable Hybrid APC, valid logits/sampling behavior, and measurable prefill/decode performance under realistic serving conditions.
```

## Production readiness checklist

### 1. ABI / contract validation

This is the first gate.

Test that:

```text
CTE compile args == CTE runtime args
TKG compile args == TKG runtime args
CTE and TKG have independent explicit arg builders
No shared expanded positional list leaks into CTE
No blind *empties list is used without named slot mapping
```

Add hard checks:

```python
assert len(cte_args) == 24
assert len(tkg_args) == EXPECTED_TKG_ARITY
```

Also log every arg:

```text
index | name | shape | dtype | CTE/TKG
```

Pass criteria:

```text
No forward expected X args but received Y args
No CTE/TKG positional drift
No unexpected empty tensor in required decode metadata slots
```

---

### 2. Basic single-request correctness

Run with the safest configuration first:

```text
prefix caching: off
chunked prefill: off
on-device sampling: off
output_logits: false
max_num_seqs: 1
temperature: 0
top_k: 1
```

Test:

```text
short prompt
medium prompt
long prompt
chat-template prompt
math prompt
code prompt
multi-turn style prompt
```

Pass criteria:

```text
No crash
No token 0 collapse
No repeated garbage output
No NaN hidden states
No NaN logits
Greedy output is stable across repeated runs
```

---

### 3. Logits / numerical validation

Before performance work, prove numerical safety.

Compare against:

```text
baseline-v3 branch
HF reference if possible
```

Check:

```text
hidden_states finite before lm_head
hidden_states finite after norm
logits finite before mask_padded_logits
logits finite after mask_padded_logits
top-1 token match
top-5 overlap
logit cosine similarity
first divergence position
```

Pass criteria:

```text
No NaN / Inf
Top-1 greedy token match for core test prompts
Acceptable logit cosine / top-k overlap
No divergence at token 1 for standard prompts
```

---

### 4. Host-side logits path

Test:

```text
output_logits=true
on-device sampling=false
CPU/host sampling
```

Pass criteria:

```text
vLLM receives finite logits
No dummy token 0 collapse
No output alias mismatch
No wrong tensor read as logits
```

This directly validates your current failure mode.

---

### 5. On-device sampling path

Test:

```text
output_logits=false
on-device sampling=true
temperature=0
top_k=1
```

Then later:

```text
temperature > 0
top_k > 1
top_p < 1
```

Pass criteria:

```text
No NRT_EXEC_OOB
No invalid token IDs
No token 0 collapse
Greedy on-device sampling matches host greedy output
```

---

### 6. CTE / prefill validation

Test prompt lengths:

```text
128
512
2K
8K
16K
32K
64K
128K
```

If final target is native Qwen3.6 context, also test:

```text
262K
```

Pass criteria:

```text
CTE completes
No dense mask memory blow-up
No NaN states after prefill
TTFT is recorded
Prefill tok/s is recorded
Memory usage is within expected range
```

---

### 7. TKG / decode validation

Test decode lengths:

```text
1 token
16 tokens
64 tokens
256 tokens
1024 tokens
```

Pass criteria:

```text
No OOB
No state corruption across tokens
Decode tok/s recorded
TPOT / inter-token latency recorded
Output remains coherent over long decode
```

---

### 8. Hybrid APC correctness

This is critical for Qwen3.6.

Test:

```text
exact same prompt twice
same system prompt + different user prompt
long shared prefix + short suffix
partial-prefix reuse
multi-turn reuse
cache miss after unrelated prompt
```

Validate that APC restores:

```text
attention KV cache
DeltaNet recurrent state
DeltaNet convolution state
position/context length
slot/block mapping
```

Pass criteria:

```text
Warm APC output matches cold full-prefill output
No cross-request contamination
No wrong continuation after prefix reuse
Cache hit improves TTFT
Cache miss behaves like cold prefill
```

---

### 9. Chunked prefill + Hybrid APC

Test:

```text
chunked prefill off + APC off
chunked prefill on + APC off
chunked prefill on + APC on
```

Pass criteria:

```text
All modes generate correct output
Chunk boundaries do not change output
Final DeltaNet recurrent state matches non-chunked path
Final conv state matches non-chunked path
Attention KV cache is correctly scattered by absolute position
```

---

### 10. Batch / concurrency validation

Test:

```text
max_num_seqs = 1
max_num_seqs = 2
max_num_seqs = 4
max_num_seqs = 8
```

Only increase after correctness passes.

Pass criteria:

```text
No request-to-request state leak
No seq_id collision
No PA block OOB
No wrong output assigned to wrong request
Throughput improves or queueing behavior is understood
Tail latency recorded
```

---

### 11. Cache/state reset validation

Run:

```text
request A
request B unrelated
request A again
clear cache
request A again
```

Pass criteria:

```text
No stale DeltaNet state reused accidentally
No stale conv state reused accidentally
No stale KV cache reused accidentally
State reset works between independent requests
```

---

### 12. Performance benchmarks

Record at minimum:

```text
TTFT
prefill tok/s
TPOT / ITL
decode tok/s
end-to-end tok/s
memory usage
NeuronCore utilization
compile/load time
prefix-cache speedup
```

Run benchmark matrix:

```text
cold prefill
warm prefix reuse
short prompt + long decode
long prompt + short decode
multi-request batch
```

Pass criteria:

```text
Performance is better than baseline-v3 in target scenarios
No correctness regression
No large unexplained latency spikes
```

---

### 13. Long-run stability

Run soak tests:

```text
100 requests
1,000 requests
mixed prompt lengths
mixed cached / uncached prompts
mixed decode lengths
```

Pass criteria:

```text
No memory growth
No Neuron runtime crash
No NaNs after many requests
No increasing latency trend
No cache corruption over time
```

---

### 14. Failure-mode tests

Test intentionally bad or edge inputs:

```text
empty prompt
very short prompt
max-length prompt
prompt near bucket boundary
batch with uneven prompt lengths
EOS early
ignore_eos=true
invalid / very long chat template
```

Pass criteria:

```text
Graceful failure or correct output
No runtime OOB
No silent wrong output
```

---

## Suggested branch roadmap

```text
v3 baseline
   ↓
experimental + vLLM APC PR
   ↓
fix CTE/TKG explicit ABI contracts
   ↓
host logits path validation
   ↓
on-device sampling validation
   ↓
Hybrid APC correctness
   ↓
cold-prefill-perf
   ↓
concurrency + long-run production tests
```

## Minimum “prod ready” acceptance gate

I would not call it production-ready until all these pass:

```text
1. CTE and TKG arg contract manifests match compile/runtime exactly
2. No NaN logits in host sampling
3. No NRT_EXEC_OOB in on-device sampling
4. Greedy output stable and token-correct vs baseline-v3
5. Hybrid APC warm output matches cold output
6. Chunked prefill output matches non-chunked output
7. No cross-request state contamination
8. 128K context works reliably, or 262K if that is the product target
9. max_num_seqs > 1 tested without seq/cache corruption
10. Soak test passes without memory/runtime instability
```

## Practical advice

Do **not** start with the full production matrix immediately.

First finish this small ladder:

```text
1. CTE 24-arg contract works
2. TKG explicit contract works
3. output_logits=false decode works
4. output_logits=true gives finite logits
5. on-device sampling works without OOB
6. APC warm/cold equality works
7. then move to cold-prefill-perf
```

That is the cleanest path from your current bug to a production-ready Trainium implementation.
