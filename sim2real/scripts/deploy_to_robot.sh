#!/usr/bin/env bash
# Rsync the so-frame sim2real project to a robot host over SSH, then optionally
# provision it (uv sync). Adapted from livekit-actuate/scripts/deploy_to_robot.sh
# for this repo's single-project layout (one uv project, robot + policy operators).
#
# The remote path is ALWAYS ~/so-frame-sim2real -- you only provide the host.
#
# Usage:
#   ./scripts/deploy_to_robot.sh <host> [--sync]
#
#   <host>   ssh target, e.g. pi@robot.local  or  binhpham@robotbox
#   --sync   run `uv sync` on the remote after the copy (first run is slow)
#
# Ships robot/ + policy/ (incl. the camera-mapping JSONs) + common.py +
# portal.yaml + pyproject.toml + uv.lock + .env. Excludes venv/caches, model
# weights (*.pt), and recordings -- see scripts/deploy.rsyncignore. `.env` IS
# synced so the robot inherits LiveKit config; per-machine overrides go in
# `.env.local`, which is NOT synced.
#
# The trained checkpoint (*.pt) is NOT shipped (it lives on the training box).
# On the policy host, scp the .pt over and set SQUINT_CHECKPOINT, or point it at
# a shared path.
set -euo pipefail

REMOTE_PATH="~/so-frame-sim2real"   # fixed target on the robot host

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 <host> [--sync]" >&2
    echo "  e.g. $0 pi@robot.local --sync" >&2
    exit 1
fi

HOST="$1"
DO_SYNC=0
[[ "${2:-}" == "--sync" ]] && DO_SYNC=1

command -v rsync >/dev/null 2>&1 || { echo "rsync not found on this machine" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the sim2real/ dir (script is in scripts/)
IGNORE_FILE="$HERE/scripts/deploy.rsyncignore"

echo "[deploy] syncing $HERE/ -> $HOST:$REMOTE_PATH/"
# Unquoted $REMOTE_PATH in the remote command so the remote shell expands ~.
ssh "$HOST" "mkdir -p $REMOTE_PATH"
rsync -azP --delete --exclude-from="$IGNORE_FILE" "$HERE/" "$HOST:$REMOTE_PATH/"

if [[ "$DO_SYNC" == "1" ]]; then
    echo "[deploy] running 'uv sync' on $HOST (first run is slow) ..."
    if ! ssh "$HOST" "cd $REMOTE_PATH && \${UV:-\$HOME/.local/bin/uv} sync"; then
        echo "[deploy] WARN: remote 'uv sync' failed -- run it on the host yourself" >&2
    fi
fi

cat <<EOF

[deploy] done. On the robot host:

  cd ${REMOTE_PATH}
$( [[ "$DO_SYNC" != "1" ]] && echo "  uv sync                                       # once (or re-run this with --sync)" )
  cp .env.example .env    # then fill in LIVEKIT_URL / API key+secret if not already synced
  uv run robot                                  # robot: SO-101 arm + rail + cameras

  # Policy operator (on a GPU host; scp the .pt over and set SQUINT_CHECKPOINT):
  #   SQUINT_CHECKPOINT=/path/to/ckpt_best.pt uv run policy-squint
  # Verify the wiring first, no motion:  uv run python policy/debug_policy.py --bridge

EOF
