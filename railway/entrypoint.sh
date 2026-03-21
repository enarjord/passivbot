#!/usr/bin/env sh
set -eu

cat > /app/api-keys.json <<EOF
{
  "bitget_01": {
    "exchange": "${EXCHANGE}",
    "key": "${API_KEY}",
    "secret": "${API_SECRET}"$([ -n "${API_PASSPHRASE:-}" ] && printf ',\n    "passphrase": "%s"' "$API_PASSPHRASE")
  }
}
EOF

exec python src/main.py /app/runtime/configs/live.json
