# Qwen3.6 256k FP8 Sparse Runtime Error Log

## 2026-05-25: Full Sparse Artifact HBM Load Failure on TRN2

- What failed: first offline runtime smoke test on TRN2 for the full sparse 256k FP8 artifact failed during model/NEFF load, before generation.
- Host: `ubuntu@16.51.165.37`, `trn2.3xlarge`, 4 NeuronCores, logical NC config 2.
- Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_sparsepfx_nki_fusedstable_directsolve_hostlogits_b256_cte512_768_1536_3072_pfx32k_3072to256k_slots64_tkg8192_32768_131072_262144_async_20260525T204424Z`.
- Command path: `validation_scripts/qwen36_hybrid_apc_context_sweep.py --lengths 1024 --max-model-len 262144 --pa-num-blocks 1024 --cte-buckets 512,768,1536,3072 --token-generation-buckets 8192,32768,131072,262144`.
- Exact error:
  - `Failed to allocate 1.000GB (alignment: 4.000MB, usage: shared scratchpad) on ND 0:NC 6`
  - `Aligned allocations may fail even if HBM has space for the actual size requested, due to running out of alignment boundaries`
  - Memory dump files: `/tmp/nrt_mem_log_device_0_6a14c4e1.csv`, `/tmp/neuron_mem_log_device_0_hbm_3_6a14c4e1.csv`, `/tmp/neuron_mem_table_device_0_nc_2.log`.
- How we got there: runtime was asked to load every compiled context pair in the full sparse artifact at once: dense short-prefix pairs for CTE `512/768/1536/3072` through prefix `32768`, plus long-prefix fallback pairs `3072:65536`, `3072:131072`, `3072:262144`, plus token-generation buckets.
- Root cause / hypothesis: the compiled artifact itself is valid, but loading all NEFF bucket variants at once exceeds or fragments the per-LNC shared scratchpad/HBM allocation budget on this `trn2.3xlarge`.
- Fix / mitigation being tested: run production/tests with a context-tier subset per process, so each process loads only the buckets needed for that traffic tier:
  - short-context tier: selected dense short-prefix pairs only;
  - long-context tier: `3072` active-token long-prefix fallback pairs only.
- Verification status: pending subset-load smoke, then cold/warm context sweep, TTFT/TPOT, and memory capture.

## 2026-05-25: Runtime Override Did Not Reduce Loaded NEFF Buckets

- What failed: a second smoke test tried to load only CTE `512` with explicit runtime pairs `512:0` and `512:512`, plus only token-generation bucket `8192`, but engine initialization failed before generation with the same shared-scratchpad allocation class.
- Host: `ubuntu@16.51.165.37`, `trn2.3xlarge`, 4 NeuronCores, logical NC config 2.
- Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_sparsepfx_nki_fusedstable_directsolve_hostlogits_b256_cte512_768_1536_3072_pfx32k_3072to256k_slots64_tkg8192_32768_131072_262144_async_20260525T204424Z`.
- Command path: `validation_scripts/qwen36_hybrid_apc_context_sweep.py --lengths 1024 --cte-buckets 512 --context-encoding-bucket-pairs 512:0 512:512 --token-generation-buckets 8192`.
- Exact error:
  - `Failed to allocate 1.000GB (alignment: 4.000MB, usage: shared scratchpad) on ND 0:NC 4`
  - `Aligned allocations may fail even if HBM has space for the actual size requested, due to running out of alignment boundaries`
  - Memory dump files included `/tmp/nrt_mem_log_device_0_6a14c5f1.csv`, `/tmp/neuron_mem_log_device_0_hbm_2_6a14c5f1.csv`, and `/tmp/neuron_mem_table_device_0_nc_0.log`.
- How we got there: the runtime override changed the visible `NeuronConfig` values, but the saved `model.pt` still contains references to the compiled NxD workdir bucket NEFFs. The startup log listed many `_tp0_bk*` NEFFs from `_nxd_model_workdir_...204424Z/context_encoding_model/`, so the process still staged the full compiled bucket set.
- Root cause / hypothesis: with this NxD artifact format, the bucket set loaded by Neuron Runtime is determined by the TorchScript/NxD model saved in `model.pt`, not only by the runtime override JSON. Runtime bucket overrides control selection/routing after load; they do not prune the NEFFs embedded/referenced by the compiled model.
- Fix / mitigation being tested:
  - first try the documented Neuron Runtime scratchpad page-size mitigation (`NEURON_SCRATCHPAD_PAGE_SIZE=2048`, then `1536`/`1024` if needed);
  - if that does not load, recompile separate production artifacts for short and long context tiers so each `model.pt` is traced with only the buckets that tier must serve.
- Verification status: pending page-size smoke load.

## 2026-05-25: `NEURON_SCRATCHPAD_PAGE_SIZE=2048` Still Failed

- What failed: a smoke load with `NEURON_SCRATCHPAD_PAGE_SIZE=2048` failed during `self.traced_model.nxd_model.initialize(...)`, before any prompt generation.
- Host: `ubuntu@16.51.165.37`, `trn2.3xlarge`, 4 NeuronCores, logical NC config 2.
- Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_sparsepfx_nki_fusedstable_directsolve_hostlogits_b256_cte512_768_1536_3072_pfx32k_3072to256k_slots64_tkg8192_32768_131072_262144_async_20260525T204424Z`.
- Command path: `NEURON_SCRATCHPAD_PAGE_SIZE=2048 validation_scripts/qwen36_hybrid_apc_context_sweep.py --lengths 1024 ...`.
- Exact error:
  - `Failed to allocate 2.000GB (alignment: 4.000MB, usage: shared scratchpad) on ND 0:NC 6`
  - `RuntimeError: Could not load the model status=4 message=Allocation Failure`
  - Memory dump files included `/tmp/nrt_mem_log_device_0_6a14c76e.csv`, `/tmp/neuron_mem_log_device_0_hbm_3_6a14c76e.csv`, and `/tmp/neuron_mem_table_device_0_nc_2.log`.
- How we got there: AWS documents that shared scratchpad is allocated in pages and can be controlled with `NEURON_SCRATCHPAD_PAGE_SIZE`. The artifact was compiled with `scratchpad_page_size=1024`; runtime page size `2048` was tested as the simplest no-recompile mitigation.
- Root cause / hypothesis: larger page size changes the requested contiguous page from 1 GB to 2 GB, but the full traced bucket set still leaves too little aligned HBM headroom on one LNC. The artifact has too many CTE/TKG bucket NEFFs embedded in the saved model for this instance shape.
- Fix / mitigation: abandon runtime-only page-size tuning for the full sparse artifact on `trn2.3xlarge`; use separately traced/compiled production artifacts with smaller bucket sets per serving tier.
- Verification status: page-size mitigation failed; next step is to identify/load a smaller artifact or recompile tiered artifacts.

## 2026-05-25: Long-Tier Compile Restart Needed Because Remote Compile Script Was Stale

- What failed: the first production long-context tier compile launch on TRN2 exited immediately.
- Host: `ubuntu@16.51.165.37`.
- Command path: `tmp_compile_qwen256k_fp8_full_prod_long_tier_hostlogits.sh`, artifact base `qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_long_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx0_32k_64k_128k_256k_slots64_tkg32768_131072_262144_async_20260525T221513Z`.
- Exact error:
  - `qwen36_27b_compile_fp8.py: error: unrecognized arguments: --context-encoding-bucket-pairs 3072:0 3072:32768 3072:65536 3072:131072 3072:262144`
- How we got there: the runtime and validation files had been synced to TRN2, but `contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py` on the remote was still older than the local branch and lacked the sparse pair CLI.
- Fix / mitigation: sync the current local compile script to TRN2, then restart the same long-tier compile.
- Verification status: pending restart after sync.

## 2026-05-25: Production Validation Harness Could Hang on Early Server Failure

- What failed: while reviewing the post-compile validation runner before the long-tier artifact finished compiling, the memory sampler cleanup path was found to be incomplete.
- Host / path: local repo, then synced to `ubuntu@16.51.165.37:/home/ubuntu/inferentia-gdn-fused-noclamp-4340808`.
- Command path: `tmp_run_qwen256k_fp8_prod_validation.sh` starts `validation_scripts/neuron_memory_sampler.py` before offline sweeps and vLLM server tests.
- Exact error class: no observed runtime failure yet; this was a pre-run robustness defect. If vLLM exited before any process matched the sampler regex, `--stop-when-no-match` would never observe a prior match and the runner could block forever at `wait "${sampler_pid}"`.
- How we got there: the sampler was designed to exit after a matching process appeared and later disappeared. That is correct for normal benchmark completion, but not for failed startup paths where the benchmark process never reaches the expected command name.
- Fix / mitigation:
  - `validation_scripts/neuron_memory_sampler.py` now ignores its own PID and handles `SIGTERM`/`SIGINT` by breaking the sampling loop and writing the summary JSON.
  - `tmp_run_qwen256k_fp8_prod_validation.sh` now explicitly stops the sampler after each benchmark/server phase instead of relying only on process-regex disappearance.
  - The chat sampler regex was widened to include vLLM server process names during startup.
- Verification status: local `python3 -m py_compile validation_scripts/neuron_memory_sampler.py` passed, local `bash -n tmp_run_qwen256k_fp8_prod_validation.sh` passed, both files were synced to TRN2, and remote py_compile/shell syntax checks passed.

## 2026-05-25: TRN2 SSH Timeout During Long-Tier Compile Monitoring

- What failed: new SSH sessions to the TRN2 validation host temporarily stopped completing the SSH banner exchange while the long-tier compile was active.
- Host: `ubuntu@16.51.165.37`.
- Command path: repeated lightweight SSH probes with `/Users/deepankarsingh1312/Downloads/trainium.pem` and `UserKnownHostsFile=/private/tmp/codex_known_hosts_16_51_165_37`.
- Exact error:
  - `Connection timed out during banner exchange`
  - `Connection to 16.51.165.37 port 22 timed out`
- How we got there: the compile had progressed past priority-HLO `Compiler status PASS` and was compiling all HLOs. The main process was last observed alive as PID `45762`, with the workdir containing 2 NEFFs and 8 compiler logs, before subsequent SSH tail/status commands began timing out.
- Root cause / hypothesis: not yet proven. The most likely causes are host load, sshd backlog, or temporary network/instance responsiveness during heavy compile. This is not yet evidence that the compile failed.
- Fix / mitigation: stop opening long live-tail sessions and use lighter periodic SSH probes. If SSH returns, inspect the compile PID/log/artifact first. If SSH remains unavailable across repeated checks, use AWS instance status/console routes or wait for the heartbeat monitor to regain access before deciding whether to restart anything.
- Verification status: pending; SSH was still unavailable on the first retry after a short wait.

## 2026-05-25: Local AWS CLI Could Not Inspect TRN2 Instance

- What failed: attempted to inspect the EC2 state of `16.51.165.37` from the local AWS CLI after SSH banner exchange timeouts.
- Command path: `aws ec2 describe-instances --filters Name=ip-address,Values=16.51.165.37 ...`.
- Exact error:
  - `An error occurred (AuthFailure) when calling the DescribeInstances operation: AWS was not able to validate the provided access credentials`
- How we got there: SSH was not completing, so EC2 API status was the next non-SSH way to distinguish host overload from instance/network failure.
- Root cause / hypothesis: the local default AWS credentials are expired, invalid, or not authorized for this account/region.
- Fix / mitigation: cannot use local EC2 APIs until credentials are refreshed. Continue with periodic SSH probes and existing heartbeat automation; if manual intervention is needed, refresh AWS credentials or use the AWS console to inspect/stop/start the instance.
- Verification status: `aws configure list-profiles` only showed `default`; no alternate local profile was available to try.

## 2026-05-25: Jump-Host Cross-Check Was Not Available

- What failed: tried to use earlier EC2 hosts as alternate network vantage points for checking `16.51.165.37:22`.
- Command path: SSH to `ubuntu@16.26.202.235`, `ubuntu@16.50.54.122`, and `ubuntu@16.50.51.125`, then run `nc -vz -w 8 16.51.165.37 22`.
- Exact errors:
  - `ssh: connect to host 16.26.202.235 port 22: Operation timed out`
  - `ssh: connect to host 16.50.54.122 port 22: Operation timed out`
  - `ssh: connect to host 16.50.51.125 port 22: Operation timed out`
- How we got there: direct local TCP to `16.51.165.37:22` succeeded, but SSH banner exchange still timed out. A jump-host check would have helped distinguish local routing from TRN2 sshd/host responsiveness.
- Root cause / hypothesis: those previous compile hosts are likely stopped, firewalled, or otherwise not accepting SSH now.
- Fix / mitigation: no jump-host route is available from the known public IPs. Continue direct lightweight probes to TRN2 and use the heartbeat automation once SSH recovers.
- Verification status: pending TRN2 SSH recovery.

## 2026-05-25: TRN2 SSH Banner Timeout Persisted Across Goal Continuations

- What failed: a fresh lightweight access check again found `16.51.165.37:22` reachable at TCP level, but SSH still failed before authentication.
- Host: `ubuntu@16.51.165.37`.
- Command path:
  - `nc -vz -w 8 16.51.165.37 22`
  - `ssh -4 -i /Users/deepankarsingh1312/Downloads/trainium.pem ... ubuntu@16.51.165.37 ...`
- Exact output:
  - TCP probe: `Connection to 16.51.165.37 port 22 [tcp/ssh] succeeded!`
  - SSH probe: `Connection timed out during banner exchange`
  - SSH probe: `Connection to 16.51.165.37 port 22 timed out`
- How we got there: this was a third consecutive goal continuation where the same TRN2 access failure prevented checking the long-tier compile result or launching validation.
- Root cause / hypothesis: the instance network path is open, but sshd or the host is not responding to new sessions. Based on the last successful status, the host was under active Neuron compile load, but without SSH or valid EC2 API credentials this cannot be proven from current state.
- Fix / mitigation: the goal is blocked until the instance can complete SSH login or EC2/API/console access is restored. Once access returns, first inspect the compile PID/log/artifact, then run the production validation runner already synced on the host.
- Verification status: blocked on external access recovery.

## 2026-05-25: TRN2 Port 22 Stopped Accepting TCP During Heartbeat Probe

- What failed: a later heartbeat probe no longer reached SSH far enough to time out during banner exchange; the TCP connection itself timed out.
- Host: `ubuntu@16.51.165.37`.
- Command path:
  - `ssh -4 -i /Users/deepankarsingh1312/Downloads/trainium.pem ... ubuntu@16.51.165.37 ...`
  - `nc -vz -w 8 16.51.165.37 22`
- Exact output:
  - SSH probe: `ssh: connect to host 16.51.165.37 port 22: Operation timed out`
  - TCP probe: `nc: connectx to 16.51.165.37 port 22 (tcp) failed: Operation timed out`
- How we got there: previous heartbeats showed TCP port 22 reachable but SSH banner exchange failing. This heartbeat showed port 22 itself was no longer reachable from the local environment.
- Root cause / hypothesis: the instance may have become network-unreachable, stopped, rebooted, failed status checks, or become too overloaded to accept new TCP connections. Without valid EC2 API credentials or console access, the exact state cannot be verified locally.
- Fix / mitigation: recover or inspect the instance through AWS console/API, then retry SSH. Do not assume the compile failed until the remote disk/log/artifact state can be inspected.
- Verification status: blocked on external instance/network access recovery.

## 2026-05-26: Production Long-Tier Compile Failed Large Prefix Buckets with F137

- What failed: after SSH recovered, the production long-context tier compile was still running but had recorded failed Neuron compiler invocations for the largest long-prefix context buckets.
- Host: `ubuntu@16.51.165.37`.
- Artifact target: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_long_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx0_32k_64k_128k_256k_slots64_tkg32768_131072_262144_async_20260525T221644Z`.
- Workdir: `/mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_256k_fp8_full_prod_long_cte3072_pfx0_32k_64k_128k_256k_tkg32768_131072_262144_20260525T221644Z`.
- Failed buckets:
  - `context_encoding_model/_tp0_bk4`, corresponding to `[CTE 3072, prefix 262144]`.
  - `context_encoding_model/_tp0_bk3`, corresponding to `[CTE 3072, prefix 131072]`.
- Exact errors:
  - `Backend exited with code -9`
  - `An Internal Compiler Error has occurred`
  - `[F137] neuronx-cc was forcibly killed - This most commonly occurs due to insufficient system memory. Using a smaller data type, dimensions, batch size, or a larger instance type may help.`
- How we got there: to keep one long-tier artifact, the compile traced five explicit long-prefix pairs: `3072:0`, `3072:32768`, `3072:65536`, `3072:131072`, and `3072:262144`. The priority token-generation HLO passed, smaller context work had produced three NEFFs, then the two largest prefix buckets were killed during backend compilation.
- Root cause / hypothesis: this is no longer the earlier `NCC_ITIN902 TensorInitialization` affine bug. It is a compile-host memory pressure failure for the largest sparse prefix shapes, likely amplified by compiling multiple HLOs in the same compile run.
- Fix / mitigation to try next:
  - Do not validate this artifact; it only has `neuron_config.json` in the target and missing NEFFs for the largest required long-context buckets.
  - Recompile the largest prefix bucket(s) in smaller isolated artifacts or on a larger-memory compile host.
  - First robust/simple retry candidate: split the long tier further so each compile contains fewer large shapes, for example one artifact for `3072:32768,3072:65536` and a separate artifact for `3072:131072,3072:262144`, or compile `3072:262144` alone if memory remains tight.
  - Keep runtime serving tiered; do not reintroduce the full combined sparse bucket matrix into one artifact on `trn2.3xlarge`.
- Verification status: pending; current compile process had not fully exited at the time of inspection, but the target artifact was incomplete and validation must not start from it.
