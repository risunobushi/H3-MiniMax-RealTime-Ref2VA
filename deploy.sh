#!/usr/bin/env bash
# Upload the runtime to a fresh Runpod pod and install it. Run from your machine, in the repository root:
#   bash deploy.sh root@<pod-ip> <ssh-port>
# Afterwards start the server on the pod with /workspace/Minimax-H3-Turbo/start_live.sh (see README.md).
set -euo pipefail
HOST=${1:?usage: bash deploy.sh root@<pod-ip> <ssh-port>}
PORT=${2:?usage: bash deploy.sh root@<pod-ip> <ssh-port>}
HERE=$(cd "$(dirname "$0")" && pwd)
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT $HOST"
SCP="scp -o ConnectTimeout=20 -q -P $PORT"

# /workspace/Minimax-H3-Turbo must NOT exist yet: setup_live.sh skips its git clone when the directory is there
$SSH "mkdir -p /workspace/stage /workspace/garments /workspace/outputs"
$SCP "$HERE"/server/* "$HERE"/setup/setup_live.sh "$HERE"/tools/ws_*.py "$HOST:/workspace/stage/"
$SCP "$HERE"/garments/* "$HOST:/workspace/garments/"
echo "== installing on the pod (log: /workspace/outputs/setup.log)"
$SSH "sed -i 's/\r\$//' /workspace/stage/*.sh && bash /workspace/stage/setup_live.sh 2>&1 | tee /workspace/outputs/setup.log | grep -a '^==\|SETUP DONE\|rror' \
  && cp /workspace/stage/* /workspace/Minimax-H3-Turbo/ && echo DEPLOYED"
echo "next, on the pod:  nohup bash /workspace/Minimax-H3-Turbo/start_live.sh > /workspace/outputs/live.log 2>&1 < /dev/null &"
