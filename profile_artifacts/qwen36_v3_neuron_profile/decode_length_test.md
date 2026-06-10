# Qwen3.6-27B vLLM Decode Length Test

Environment:

- Instance: `18.220.20.118`, trn2.48xlarge
- Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_v3_profile`
- vLLM settings: prefix caching on, chunked prefill on, max model length 131072, CTE bucket 512
- Prompt length: about 3449 tokens
- Sampling: greedy, `ignore_eos=True` for forced-length decode

Results:

| Max Tokens | Run | Output Tokens | Elapsed | Output Speed |
|---:|---|---:|---:|---:|
| 55 effective | Cold | 55 | 11.959 s | 4.60 tok/s |
| 55 effective | Warm/APC | 55 | 3.207 s | 17.19 tok/s |
| 256 forced | Cold | 256 | 18.360 s | 13.95 tok/s |
| 256 forced | Warm/APC | 256 | 9.598 s | 26.69 tok/s |
| 512 forced | Cold | 512 | 26.562 s | 19.28 tok/s |
| 512 forced | Warm/APC | 512 | 17.738 s | 28.88 tok/s |

Conclusion:

The earlier 32-token warm profile reported only about 13 tok/s because fixed vLLM/request overhead dominated the short generation. Once generation is long enough, warm vLLM decode reaches 26.7-28.9 tok/s, matching the standalone decode baseline.
