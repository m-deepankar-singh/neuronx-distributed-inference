# Qwen3.6 Fused Sample+Logits Vocab-Parallel Validation

Validation host: `ubuntu@16.50.153.24`

Artifact:
`qwen36_27b_128k_fp8_mlp_edgebf16_hybrid_apc_nki_fusedstable_directsolve_samplelogits_vocabparallel_b256_cte256_512_pfx16k_slots64_tkg8192_32768_131072_async_20260523T123711Z`

Code head before the local clone fix: `aad1987`

## Results

- EC2-to-EC2 artifact copy verified: `38110339067` bytes, `6` files.
- EC2-to-EC2 model copy verified: `55586111245` bytes, `59` files.
- Artifact config verified: `vocab_parallel=true`, `output_logits=true`, on-device greedy sampling present, TKG buckets `[8192, 32768, 131072]`, `async_mode=true`, `tkg_batch_size=1`, `pa_num_blocks=512`, Qwen Hybrid APC NKI chunked prefill enabled.
- Offline decode sanity: `16` generated tokens, `14.69 tok/s`, real thinking text emitted.
- Raw sample-vs-returned-logits-local-shard comparison: `15/18` positions match. The three mismatches are expected for this debug view because returned logits are rank-local shard shape `[1, 1, 62080]`, while sampled tokens are global IDs (`248068`, `90700`) selected by vocab-parallel sampling.
- Strict APC exactness with unbacked prefix reads disabled and backed prefix reads enabled passed:
  - `full_prefix_exact=true`
  - `partial_prefix_exact=true`
  - `real_generated_tokens_passed=true`
- HF greedy reference comparison using an isolated Transformers-main install passed the target threshold:
  - `156/160` token positions matched HF greedy (`97.5%`)
  - `9/10` prompts matched exactly for all `16` generated tokens
  - The only mismatch was prompt index `8`, after `12` exact tokens: HF selected token `1826`, Neuron selected token `34099`
  - Neuron offline decode throughput during this comparison was about `11.08-14.75 tok/s` across the 10 prompts

## Notes

The host Neuron/vLLM virtualenv was left unchanged. HF greedy references were generated with an isolated Transformers-main install at `/home/ubuntu/hf_ref_pkgs_transformers_main_clean2_20260523`, because the packaged Transformers build in the Neuron environment did not recognize `model_type=qwen3_5`.
