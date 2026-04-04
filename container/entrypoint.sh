#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_ROOT=${PB_APP_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)}
RUNTIME_ROOT=${PB_RUNTIME_ROOT:-/run/passivbot}
API_KEYS_TARGET="$APP_ROOT/api-keys.json"
GENERATED_CONFIG_PATH="$RUNTIME_ROOT/config.runtime.json"
GENERATED_API_KEYS_PATH="$RUNTIME_ROOT/api-keys.json"

mkdir -p "$RUNTIME_ROOT"

export PASSIVBOT_IGNORE_ENV_MISMATCH=1

if [ -z "${PB_USER:-}" ]; then
    echo "PB_USER is required for container live runs." >&2
    exit 2
fi

if [ -n "${PB_API_KEYS_PATH:-}" ]; then
    if [ ! -f "$PB_API_KEYS_PATH" ]; then
        echo "PB_API_KEYS_PATH does not exist: $PB_API_KEYS_PATH" >&2
        exit 2
    fi
    ln -sf "$PB_API_KEYS_PATH" "$API_KEYS_TARGET"
elif [ -f "$API_KEYS_TARGET" ]; then
    :
else
    if [ -z "${PB_EXCHANGE:-}" ] || [ -z "${PB_API_KEY:-}" ] || [ -z "${PB_API_SECRET:-}" ]; then
        echo "Set PB_API_KEYS_PATH or provide PB_EXCHANGE, PB_API_KEY, and PB_API_SECRET." >&2
        exit 2
    fi
    python3 "$SCRIPT_DIR/render_api_keys.py" "$GENERATED_API_KEYS_PATH"
    ln -sf "$GENERATED_API_KEYS_PATH" "$API_KEYS_TARGET"
fi

CONFIG_PATH=${PB_CONFIG_PATH:-}
if [ -n "$CONFIG_PATH" ] && [ ! -f "$CONFIG_PATH" ]; then
    echo "PB_CONFIG_PATH does not exist: $CONFIG_PATH" >&2
    exit 2
fi

if [ -n "${PB_CONFIG_INLINE:-}" ] || [ -n "${PB_APPROVED_COINS:-}" ] || [ -n "${PB_LOG_LEVEL:-}" ] || [ -n "${PB_MONITOR_ENABLED:-}" ] || [ -n "${PB_MONITOR_ROOT:-}" ]; then
    python3 "$SCRIPT_DIR/render_config.py" "$GENERATED_CONFIG_PATH"
    CONFIG_PATH="$GENERATED_CONFIG_PATH"
fi

if [ -n "$CONFIG_PATH" ]; then
    exec passivbot live "$CONFIG_PATH" -u "$PB_USER" "$@"
fi

exec passivbot live -u "$PB_USER" "$@"
