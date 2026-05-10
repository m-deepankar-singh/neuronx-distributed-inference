# Qwen3.6 vLLM APC Baseline v3 Remote Snapshot

Snapshot taken from Trainium instance `16.51.5.28` before expected
termination.

Contents:

- `qwen36_v3_preserve_20260510T103642Z.tar.gz`: raw remote snapshot archive.
- `qwen36_v3_preserve_20260510T103642Z/repo_state.txt`: remote branch, dirty
  state, recent commits, and tags.
- `qwen36_v3_preserve_20260510T103642Z/processes.txt`: running vLLM/proxy
  command lines.
- `qwen36_v3_preserve_20260510T103642Z/models.json`: `/v1/models` response
  from the guarded proxy.
- `qwen36_v3_preserve_20260510T103642Z/qwen36_vllm_apc_hardening_eval_20260510T101345Z.json`:
  APC hardening result.
- `qwen36_v3_preserve_20260510T103642Z/qwen36_27b_vllm_long_chat_eval_20260510T101703Z.json`:
  long-context quality result under APC.
- `qwen36_v3_preserve_20260510T103642Z/qwen36_apc_backend.log`,
  `qwen36_v3_preserve_20260510T103642Z/qwen36_vllm_apc_backend.log`, and
  `qwen36_v3_preserve_20260510T103642Z/qwen36_apc_proxy.log`: backend/proxy logs
  captured from `/tmp`.

The source baseline is also preserved in git tag
`qwen36-27b-vllm-apc-baseline-v3`.
