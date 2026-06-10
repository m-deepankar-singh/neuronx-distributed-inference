Qwen3.6 decodefast runtime validation
=====================================

Artifact tested on Trn2 `16.51.6.234`:

`/mnt/trainium_artifacts/qwen_artifacts/LATEST_DECODEFAST_128K`

Runtime fix:

- Preserve suffix-only chunked CTE continuation tensors instead of treating the
  `256 suffix + 256 prefix` case as a `[512, 0]` no-prefix prefill.
- Build the prefix attention mask from `computed_context_lens` for suffix-only
  continuations and avoid left-padding their active slot mappings.

Validation:

- Remote Neuron unit tests:
  `PYTHONPATH=src python -m pytest test/unit/models/test_prefix_caching_bucket_selection.py test/unit/models/test_model_wrapper.py test/unit/modules/test_async_execution.py -q`
  passed: `124 passed`.
- Live OpenAI-compatible server, decodefast artifact:
  - 336-token fact prompt: coherent summary, 1.871 s.
  - 816-token fact prompt: coherent summary, 2.299 s. This previously returned only `\n`.
  - 5016-token fact prompt: coherent summary, 8.690 s. This previously returned only `\n`.
  - SillyTavern-shaped multi-system streaming request: HTTP 200, no socket hangup, 384 content chars, coherent Seraphina reply.
  - 32-token non-streaming decode: 32 completion tokens in 1.830 s, 17.48 tok/s.
- Neuron monitor during context sweep:
  - Peak Neuron device memory: 59.88 GiB.
  - Peak host memory attributed to runtime: 11.44 GiB.

Known limit:

- `LATEST_DECODEFAST_128K` has `prefix_buckets` capped at 16384. A near-64K
  semantic prompt eventually hit `Prefix len 16640 exceeds largest bucket 16384`
  and killed the vLLM backend. This artifact should be treated as decode-fast
  for prompts up to the pfx16K range unless the server is started with a lower
  `max_model_len` guard.
- `LATEST_FP8_128K_PFX128K` has prefix buckets up to 131072, but failed to load
  at `max_model_len=65536` on this Trn2 shape because vLLM required 16.0 GiB KV
  cache and only 8.33 GiB was available after loading the artifact.

Implication:

- The newline/socket-hangup issue for normal SillyTavern-size chunked prompts is
  fixed in runtime.
- Serving true 64K prompts needs a new decodefast-style artifact with wider
  prefix buckets and enough HBM/KV headroom, or a larger serving shape.
