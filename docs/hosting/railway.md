# Railway

Railway should be treated as a thin deployment target for the canonical live container, not as a separate Passivbot runtime.

## Recommended Approach

Deploy `Dockerfile_live` and use the same env contract documented in [`docs/container_deployment.md`](../container_deployment.md).

Recommended env:

- `PB_USER`
- `PB_EXCHANGE`
- `PB_API_KEY`
- `PB_API_SECRET`

Optional:

- `PB_API_PASSPHRASE`
- `PB_CONFIG_INLINE`
- `PB_APPROVED_COINS`
- `PB_LOG_DIR`
- `PB_LOG_LEVEL`
- `PB_MONITOR_ENABLED`
- `PB_MONITOR_ROOT`

## Persistent Storage

If you want monitor output or file logs to persist across restarts, attach a Railway volume and point writable paths such as `PB_MONITOR_ROOT` and `PB_LOG_DIR` into that mounted location. `PB_LOG_DIR` maps onto Passivbot's canonical `logging.dir`, so Railway uses the same file-logging path as normal live runs.

## What Not To Do

- Do not add a large `railway/configs/live.json` snapshot.
- Do not depend on relative sidecar config files such as a bare `approved_coins.json`.
- Do not fork Railway startup away from the normal `passivbot live` CLI path.

## Recommended Usage Modes

### Simple Hosted Mode

- env-generated credentials
- schema defaults or inline config
- optional `PB_APPROVED_COINS`

Useful for demos and simple deployments.

### Operator Mode

- mounted or managed config
- managed secrets
- persistent volume

This is the safer mode for serious live use.
