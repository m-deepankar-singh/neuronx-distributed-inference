## Bug-fix goal

```text
Fix the Qwen3.6 Hybrid APC Neuron/vLLM wrapper contract so that CTE and TKG use separate, explicit input/output/padding contracts, and ensure TKG receives valid token IDs plus correct context-length metadata without confusing active decode length with prefix/context length.
```

## What must be true after the fix

```text
1. CTE runtime args exactly match CTE compiled args.
2. TKG runtime args exactly match TKG compiled args.
3. CTE output token/logits handoff into TKG is valid.
4. TKG input_ids is always a real vocab token.
5. TKG active input length is always 1 for decode.
6. Prefix/context length, e.g. 207, is used only as context metadata.
7. Prefix-caching bucket selection does not treat context length as active TKG input length.
8. No NRT_EXEC_OOB.
9. No NaN logits.
10. Repeated isolated requests work without state or padding contamination.
```

## Focused fix plan

### 1. Freeze the contracts

Create named builders:

```python
build_cte_args(...)
build_tkg_args(...)
```

Do not use one shared argument list.

Add assertions:

```python
assert len(cte_args) == 24
assert len(tkg_args) == EXPECTED_TKG_ARITY
```

Also dump:

```text
index | name | shape | dtype | min | max
```

for CTE compile, CTE runtime, TKG compile, and TKG runtime.

---

### 2. Fix CTE first

Goal:

```text
CTE must compile and run with exactly the same 24-tensor input list.
```

Pass condition:

```text
No “forward expected 25 args but received 30” error.
Cold generation reaches CTE completion.
```

---

### 3. Fix CTE → TKG handoff

Goal:

```text
The first TKG input_ids must come from the correct CTE output.
```

Add a hard guard before TKG:

```python
assert input_ids.dtype in (torch.int32, torch.int64)
assert input_ids.min() >= 0
assert input_ids.max() < vocab_size
```

Pass condition:

```text
No garbage token like 2143289344.
TKG input_ids is a valid vocab token.
```

---

### 4. Fix TKG padding length semantics

Goal:

```text
TKG active input length and prefix/context length must be separate.
```

For TKG:

```text
input_ids.shape[-1] = 1          # active decode token length
position/context length = 207    # valid prefix/context metadata
bucket = 2048                    # valid TKG context bucket
```

Do not let this happen:

```text
get_target_2d_bucket_for_prefix_caching(input_len=207) on token_generation_model
```

because `207` is context length, not active TKG length.

Pass condition:

```text
Second isolated request does not fail before Neuron execution.
Prefix caching padding accepts decode input length = 1.
```

---

### 5. Fix TKG metadata slots

Goal:

```text
slot_mapping, block_table, computed_context_lens, num_queries, and active_mask must land in the exact compiled TKG positions.
```

Do not blindly pass:

```python
*empties
```

into decode slots.

Pass condition:

```text
block metadata remains in range.
TKG starts execution.
No NRT_EXEC_OOB.
```

---

### 6. Validate host logits mode

Run:

```text
on_device_sampling = false
output_logits = true
prefix_cache = false first
chunked_prefill = false first
max_num_seqs = 1
```

Pass condition:

```text
logits finite
no dummy token 0 collapse
first and second isolated requests both pass
```

---

### 7. Validate on-device sampling mode

Run:

```text
on_device_sampling = true
output_logits = false
temperature = 0
top_k = 1
```

Pass condition:

```text
CTE output[0] is valid sampled token
TKG input_ids equals sampled token
No OOB
Generated output matches host greedy mode
```

## One-line engineering goal

```text
Make CTE/TKG ABI explicit, make CTE→TKG token handoff valid, and prevent prefix-cache padding from using context length as TKG active input length.
```

## Suggested commit title

```text
Fix Qwen3.6 Hybrid APC CTE/TKG ABI and TKG prefix-cache padding contract
```

## Suggested PR description

```text
This patch fixes the Qwen3.6 Hybrid APC Neuron/vLLM contract mismatch by separating CTE and TKG argument builders, enforcing compile/runtime arity checks, validating the CTE-to-TKG token handoff, and correcting TKG prefix-cache padding so decode active length remains 1 while context length is passed only as metadata.

The goal is to eliminate:
- CTE 24-vs-29 argument mismatch
- invalid TKG input_ids from sampled-token handoff
- NRT_EXEC_OOB caused by invalid token/cache metadata
- prefix-caching bucket errors where context length is treated as TKG active input length
- host-logits NaNs or dummy token collapse caused by output-slot mismatch
```

Start with this minimum ladder:

```text
1. CTE 24-arg runtime works
2. First cold host-logits generation works
3. Second isolated host-logits request works
4. First TKG input_ids is always valid
5. On-device greedy sampling works
6. Prefix caching works
7. Hybrid APC warm/cold equality works
```
