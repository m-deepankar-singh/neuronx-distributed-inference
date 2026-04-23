#!/usr/bin/env bash
# Sync local project dir to the trn1/inf2 dev instance.
# Usage: scripts/sync.sh [ip]
#   ip defaults to contents of .remote-ip in the project root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

IP="${1:-$(cat .remote-ip 2>/dev/null || true)}"
if [[ -z "$IP" ]]; then
  echo "error: no IP provided and .remote-ip is empty" >&2
  echo "usage: $0 <ip>    (or write the IP to .remote-ip)" >&2
  exit 1
fi

KEY="${SSH_KEY:-$HOME/Downloads/qwen.pem}"

rsync -av --delete \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.neff' \
  --exclude='*.hlo' \
  --exclude='compiled/' \
  --exclude='.remote-ip' \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  ./ "ubuntu@$IP:~/inferentia-gdn/"

echo "--- synced to ubuntu@$IP:~/inferentia-gdn/"
