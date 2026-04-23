#!/usr/bin/env bash
# Run a python script on the remote dev instance, inside the activated nxd_inference venv.
# Usage: scripts/run_remote.sh tests/test_ref_gdn_parity.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

IP="${REMOTE_IP:-$(cat .remote-ip 2>/dev/null || true)}"
if [[ -z "$IP" ]]; then
  echo "error: no IP; set REMOTE_IP or write to .remote-ip" >&2
  exit 1
fi

KEY="${SSH_KEY:-$HOME/Downloads/qwen.pem}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <script.py> [args...]" >&2
  exit 2
fi

SCRIPT="$1"
shift

ssh -i "$KEY" -o StrictHostKeyChecking=no "ubuntu@$IP" \
  "source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && cd ~/inferentia-gdn && python $SCRIPT $*"
