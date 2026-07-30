#!/usr/bin/env bash
# Rsync the so-frame deploy project to a robot host over SSH, then optionally provision
# it (uv sync). See README for invocation.
#
# Remote paths are fixed: ~/so-frame-deploy and ~/policy. rl/policy (the shared
# soframe-policy package) ships as a SIBLING of the deploy dir, because pyproject.toml
# takes it as a path dependency at `../policy` and that string must resolve identically
# here and on the robot: rl/policy next to rl/deploy locally, ~/policy next to
# ~/so-frame-deploy remotely. Otherwise `uv sync` fails with
# "Distribution not found at: file:///~/policy".
#
# Excludes venvs/caches, model weights (*.pt) and recordings, see
# scripts/deploy.rsyncignore. `.env` IS synced so the robot inherits LiveKit config;
# per-machine overrides go in `.env.local`, which is not synced. The checkpoint stays on
# the training box: scp it to the policy host and set POLICY_CHECKPOINT.
set -euo pipefail

REMOTE_PATH="~/so-frame-deploy"   # fixed target on the robot host
REMOTE_POLICY_PATH="~/policy"     # must stay `../policy` relative to REMOTE_PATH

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 <host> [--sync]" >&2
    echo "  e.g. $0 pi@robot.local --sync" >&2
    exit 1
fi

HOST="$1"
DO_SYNC=0
[[ "${2:-}" == "--sync" ]] && DO_SYNC=1

command -v rsync >/dev/null 2>&1 || { echo "rsync not found on this machine" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the rl/deploy/ dir (script is in scripts/)
IGNORE_FILE="$HERE/scripts/deploy.rsyncignore"
POLICY_SRC="$(cd "$HERE/../policy" && pwd)"               # rl/policy/, the shared package

echo "[deploy] syncing $POLICY_SRC/ -> $HOST:$REMOTE_POLICY_PATH/"
# Unquoted remote paths in the ssh commands so the remote shell expands ~.
ssh "$HOST" "mkdir -p $REMOTE_POLICY_PATH"
rsync -azP --delete --exclude-from="$IGNORE_FILE" "$POLICY_SRC/" "$HOST:$REMOTE_POLICY_PATH/"

echo "[deploy] syncing $HERE/ -> $HOST:$REMOTE_PATH/"
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

  # Policy operator (on a GPU host; scp the .pt over and set POLICY_CHECKPOINT):
  #   POLICY_CHECKPOINT=/path/to/ckpt_best.pt uv run policy
  # Verify the wiring first, no motion:  cd ../calibrate && uv run calibrate --bridge

EOF
