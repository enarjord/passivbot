#!/usr/bin/env sh
# Wrapper to sync directories between local and remote passivbot trees.
# Usage:
#   sh vpssync.sh push <local-subdir> <vps-alias-or-host>
#   sh vpssync.sh pull <vps-alias-or-host> <remote-subdir>
#
# Notes:
#   - Quote or escape wildcard patterns so your local shell does not expand them
#     before this script runs, for example:
#       sh vpssync.sh pull vps3 "logs/20260318*"

set -eu

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SYNC_TOOL="$REPO_ROOT/sync_tar.py"

if [ "$#" -lt 3 ]; then
    cat >&2 <<EOF
Usage:
  sh vpssync.sh push <local-subdir> <vps-alias>
  sh vpssync.sh pull <vps-alias> <remote-subdir>

Quote or escape wildcard patterns when pulling, for example:
  sh vpssync.sh pull vps3 "logs/20260318*"
EOF
    exit 1
fi

MODE="$1"

case "$MODE" in
    push)
        if [ "$#" -ne 3 ]; then
            echo "push usage: sh vpssync.sh push <local-subdir> <vps-alias>" >&2
            exit 1
        fi
        LOCAL_SUBDIR="${2%/}"
        REMOTE_ALIAS="$3"
        LOCAL_PATH="$REPO_ROOT/$LOCAL_SUBDIR"
        if [ ! -d "$LOCAL_PATH" ]; then
            echo "Error: local directory '$LOCAL_PATH' does not exist." >&2
            exit 1
        fi
        LOCAL_PARENT="$(dirname "$LOCAL_SUBDIR")"
        if [ "$LOCAL_PARENT" = "." ]; then
            REMOTE_DEST="/root/passivbot"
        else
            REMOTE_DEST="/root/passivbot/$LOCAL_PARENT"
        fi
        python3 "$SYNC_TOOL" push "$LOCAL_PATH" "$REMOTE_ALIAS" "$REMOTE_DEST" --remote-extract
        ;;
    pull)
        if [ "$#" -ne 3 ]; then
            echo "pull usage: sh vpssync.sh pull <vps-alias> <remote-subdir>" >&2
            exit 1
        fi
        REMOTE_ALIAS="$2"
        REMOTE_SUBDIR="$3"
        python3 "$SYNC_TOOL" pull "/root/passivbot/$REMOTE_SUBDIR" "$REMOTE_SUBDIR" --remote "$REMOTE_ALIAS" --extract
        ;;
    *)
        echo "Unknown mode '$MODE'. Use 'push' or 'pull'." >&2
        exit 1
        ;;
esac
