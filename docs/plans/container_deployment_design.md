# Container Deployment Design

## Goal

Create one canonical container deployment path for Passivbot live trading, then let Docker, Docker Compose, Railway, and similar hosts consume that same path.

This design intentionally avoids platform-specific config forks such as a large `railway/configs/live.json`.

## Non-Goals

- Do not create a separate deployment architecture for Railway.
- Do not add a second config system for containers.
- Do not make direct-script startup the canonical container path.
- Do not bake large environment-specific live configs into images.

## Current Problems

The repo already has:

- `Dockerfile`
- `Dockerfile_live`
- `docker-compose.yml`

But the current setup has several weaknesses:

1. The live container still defaults to `python src/main.py`, while current `master` now prefers the unified `passivbot` CLI.
2. The Compose examples are comment-driven and do not define a stable env/config contract.
3. There is no canonical startup wrapper for turning env vars into runtime inputs such as `api-keys.json`.
4. A Railway-specific implementation can easily drift from the real config pipeline, which is what happened in PR #573.

## Design Summary

Use a single container runtime contract:

- one canonical live image
- one canonical startup wrapper
- one canonical env/config/secrets contract
- Docker and Compose as the reference deployment paths
- Railway as a thin consumer of that same image/entrypoint

Optionally add a separate full image for backtest/optimize/tooling, but keep live deployment support independent from that.

## Proposed Files

### Canonical Runtime

- `Dockerfile_live`
  Primary live-trading container image.
- `container/entrypoint.sh`
  Canonical startup wrapper for all containerized live deployments.
- `container/render_api_keys.py`
  Small helper to generate `api-keys.json` from env vars when a mounted file is not provided.
- `container/render_config.py`
  Optional helper to generate a tiny config override file from env vars.

### Optional Full Runtime

- `Dockerfile.full`
  Full-deps image for backtest/optimize/tools.

### Example Assets

- `container/examples/docker-compose.live.yml`
  Reference local deployment example.
- `docs/container_deployment.md`
  Final user-facing documentation after implementation lands.
- `docs/hosting/railway.md`
  Thin Railway setup guide using the same image/env contract.

## Canonical Env Contract

The live container should support the following environment variables.

### Required For Live

- `PB_USER`
  User key to run, equivalent to `passivbot live -u ...`.

### Config Source

- `PB_CONFIG_PATH`
  Mounted config path inside the container.
- `PB_CONFIG_INLINE`
  Inline JSON or HJSON config payload.

Resolution order:

1. `PB_CONFIG_INLINE`
2. `PB_CONFIG_PATH`
3. no config path, which means schema defaults

### Credentials

- `PB_API_KEYS_PATH`
  Mounted `api-keys.json` path.

If `PB_API_KEYS_PATH` is unset, allow env-based credential materialization with:

- `PB_EXCHANGE`
- `PB_API_KEY`
- `PB_API_SECRET`
- `PB_API_PASSPHRASE` optional

### Execution

- extra container arguments
  Pass-through args appended after `passivbot live ...`, for example via `docker run ... --log-level info` or Compose `command: [...]`.
- `PB_SKIP_RUST_COMPILE`
  Default `1` in the runtime image after the wheel is already built.

### Optional Convenience Inputs

- `PB_APPROVED_COINS`
  Optional comma-separated list or `--symbols`-style path for simple hosted deployments.
- `PB_LOG_LEVEL`
- `PB_MONITOR_ENABLED`
- `PB_LOG_DIR`
- `PB_LOG_FILE`

When a mounted config path is in use, these should be translated into normal live CLI overrides where possible. Inline-config mode may render them into a temporary config file.

## Canonical Writable Paths

Use a small number of stable in-container paths:

- `/app`
  Application code and installed files.
- `/data`
  Persistent writable mount for operators.
- `/run/passivbot`
  Ephemeral generated runtime files such as rendered config and generated `api-keys.json`.

Recommended operator mounts:

- `/data/configs`
- `/data/logs`
- `/data/monitor`
- `/data/caches` if containerized caching is desired

## Startup Flow

`container/entrypoint.sh` should do the following.

1. Validate env contract.
2. Determine credentials source.
3. If needed, render `/run/passivbot/api-keys.json`.
4. Determine config source.
5. If `PB_CONFIG_INLINE` is used, render `/run/passivbot/config.override.json`.
6. If a mounted config path is used, translate deployment overrides into normal live CLI flags so the original config path stays intact.
7. Choose the final config path.
8. Exec the canonical CLI:

```bash
passivbot live [config_path] -u "$PB_USER" ...
```

Important:

- Use `passivbot live`, not `python src/main.py`.
- Keep the wrapper thin.
- Avoid mutating business logic in the wrapper.

## Config Strategy

### Rules

1. Prefer mounted real config files for serious use.
2. Use schema defaults when no config is provided.
3. Use generated tiny overrides only for deployment-specific settings.
4. Never bake a large live config snapshot into a platform-specific directory.

### Good Override Candidates

- `live.user`
- `live.approved_coins`
- `logging.level`
- `monitor.enabled`

### Bad Override Candidates

- large `bot` sections
- large `optimize` sections
- stale copied templates
- exchange-specific strategy snapshots

## Approved Coins Strategy

Do not depend on bare relative paths like `approved_coins.json` unless config-relative path resolution is explicitly implemented and tested.

Safer supported patterns:

1. Embed approved coins directly in config.
2. Mount a file and reference an absolute path such as `/data/configs/approved_coins.json`.
3. Use `PB_APPROVED_COINS=BTC,ETH,SOL` for simple hosted deployments.

For hosted platforms, env-to-config rendering is the most reliable path.

## Dockerfile_live Design

### Keep

- multi-stage build
- Rust wheel built in builder stage
- slim runtime image

### Change

1. Install and expose the canonical CLI runtime path.
2. Copy `container/entrypoint.sh` and helper scripts into the image.
3. Set:

```dockerfile
ENV SKIP_RUST_COMPILE=true
```

4. Default command should be:

```dockerfile
ENTRYPOINT ["/app/container/entrypoint.sh"]
```

### Notes

- The image should not depend on repository bind mounts to function.
- The runtime should work in a clean hosted environment with only env vars and optional mounted config data.

## Dockerfile.full Design

Purpose:

- backtest
- optimize
- tools
- heavier research workflows

This should use the same entrypoint pattern where possible, but it is secondary to the live image.

If added, keep it separate from `Dockerfile_live` so live operators do not pay the size and dependency cost of full installs.

## Compose Design

Replace the current comment-only `docker-compose.yml` approach with a documented reference example.

Recommended example:

```yaml
services:
  passivbot-live:
    build:
      context: .
      dockerfile: Dockerfile_live
    environment:
      PB_USER: bitget_01
      PB_CONFIG_PATH: /data/configs/live.json
      PB_API_KEYS_PATH: /data/api-keys.json
    volumes:
      - ./configs:/data/configs:ro
      - ./api-keys.json:/data/api-keys.json:ro
      - ./logs:/data/logs
      - ./monitor:/data/monitor
```

This gives users one clear contract:

- mount config
- mount keys
- set user
- run

## Railway Design

Railway support should be documentation plus environment wiring, not its own runtime branch.

Recommended Railway support model:

1. Deploy `Dockerfile_live`.
2. Set env vars in Railway UI.
3. Use the same `container/entrypoint.sh`.
4. Optionally attach a persistent volume for logs, monitor output, and caches.

### Supported Railway Modes

#### Simple Hosted Mode

- no mounted config file
- env credentials
- optional `PB_APPROVED_COINS`
- schema defaults plus a tiny generated override

This is acceptable for demos and simple deployments.

#### Real Operator Mode

- mounted config file or managed secret file
- env credentials or mounted `api-keys.json`
- persistent volume

This is the recommended serious mode.

## Why This Is Better Than A Railway-Specific Config Snapshot

This design:

- stays aligned with current `master`
- avoids stale copied templates
- avoids path-resolution surprises
- works for Docker users first
- automatically extends to Railway, Coolify, Render, and similar platforms
- keeps deployment concerns separate from strategy/config evolution

## Concrete Implementation Plan

### Phase 1: Canonical Live Container

1. Add `container/entrypoint.sh`.
2. Add `container/render_api_keys.py`.
3. Update `Dockerfile_live` to use the wrapper.
4. Make startup use `passivbot live`.
5. Verify container startup with:
   - mounted `api-keys.json`
   - env-generated credentials
   - mounted config
   - no config path

### Phase 2: Reference Compose Path

1. Add `container/examples/docker-compose.live.yml`.
2. Optionally simplify or replace current `docker-compose.yml`.
3. Document the env/volume contract.

### Phase 3: Thin Hosted-Platform Support

1. Add `docs/hosting/railway.md`.
2. Ensure Railway setup only references:
   - `Dockerfile_live`
   - env vars
   - optional persistent volume
3. Do not add `railway/configs/live.json`.

### Phase 4: Optional Full Image

1. Add `Dockerfile.full`.
2. Document it separately for research workflows.

## Suggested `container/entrypoint.sh`

This is the intended behavior, not final shell syntax:

```sh
#!/usr/bin/env sh
set -eu

mkdir -p /run/passivbot

API_KEYS_PATH="${PB_API_KEYS_PATH:-}"
CONFIG_PATH="${PB_CONFIG_PATH:-}"

if [ -z "$API_KEYS_PATH" ]; then
  python /app/container/render_api_keys.py
  API_KEYS_PATH="/run/passivbot/api-keys.json"
fi

if [ -n "${PB_CONFIG_INLINE:-}" ]; then
  python /app/container/render_config.py
  CONFIG_PATH="/run/passivbot/config.override.json"
fi

if [ -n "${PB_LOG_LEVEL:-}" ]; then
  set -- --log-level "$PB_LOG_LEVEL" "$@"
fi

if [ -n "${PB_APPROVED_COINS:-}" ]; then
  set -- --symbols "$PB_APPROVED_COINS" "$@"
fi

if [ -n "${PB_MONITOR_ENABLED:-}" ]; then
  set -- --monitor.enabled "$PB_MONITOR_ENABLED" "$@"
fi

if [ -n "${PB_MONITOR_ROOT:-}" ]; then
  set -- --monitor.root_dir "$PB_MONITOR_ROOT" "$@"
fi

export PB_API_KEYS_PATH_EFFECTIVE="$API_KEYS_PATH"

if [ -n "$CONFIG_PATH" ]; then
  exec passivbot live "$CONFIG_PATH" -u "$PB_USER"
else
  exec passivbot live -u "$PB_USER"
fi
```

The final implementation may merge mounted config with generated overrides rather than replacing it. If so, `render_config.py` should explicitly support:

- base config path
- override patch values
- deterministic output path

In practice, mounted config paths should prefer CLI overrides so relative sidecar resolution keeps using the original config path semantics.

## Suggested `render_api_keys.py`

Responsibilities:

- read `PB_USER`
- read `PB_EXCHANGE`
- read `PB_API_KEY`
- read `PB_API_SECRET`
- read optional `PB_API_PASSPHRASE`
- write deterministic JSON to `/run/passivbot/api-keys.json`
- fail loudly on missing required fields

It should not contain any exchange-specific business logic beyond field naming.

## Suggested `render_config.py`

Responsibilities:

- optionally start from mounted `PB_CONFIG_PATH`
- otherwise start from schema defaults by omitting the config path entirely, or by rendering only a tiny runtime config if needed
- apply a narrow set of deployment-oriented overrides
- write deterministic JSON to `/run/passivbot/config.override.json`

It should not duplicate the schema template in source.

## Open Design Decisions

1. Whether generated overrides should replace the main config path or produce a merged temp config from base file plus overrides.
2. Whether logs, monitor output, and caches should default under `/data` directly in the container setup.
3. Whether `docker-compose.yml` should be replaced or just supplemented with a cleaner example file.
4. Whether a later `Dockerfile.full` should reuse the same wrapper contract for non-live workloads.

## Recommended Decision Defaults

1. Merge overrides on top of an optional mounted base config.
2. Default mutable outputs to `/data` when container mode is detected or documented.
3. Add a clean example compose file first; do not break existing local workflows immediately.
4. Reuse the same wrapper contract if and when a full research image is added.

## Review Checklist For Implementation

Before merging the actual implementation:

1. Verify `passivbot live` is the runtime entrypoint inside the container.
2. Verify env-generated credentials work without a mounted `api-keys.json`.
3. Verify mounted config paths work.
4. Verify no-config mode works from schema defaults.
5. Verify approved-coins handling is explicit and tested.
6. Verify no large platform-specific config snapshot is added.
7. Verify Railway docs use the same image and wrapper as Docker users.
