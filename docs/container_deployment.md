# Container Deployment

Passivbot now has a canonical live-container contract built around `Dockerfile_live` and `container/entrypoint.sh`.

The container startup path uses:

- the installed `passivbot` CLI
- a thin entrypoint wrapper
- optional env-generated `api-keys.json`
- optional env-generated config overrides
- Passivbot's canonical `logging.*` config for file persistence

This is the intended base for Docker, Docker Compose, and hosted platforms such as Railway.

## Live Image

Build the live image from the repo root:

```bash
docker build -f Dockerfile_live -t passivbot-live:latest .
```

The image already includes the compiled Rust extension and launches the wrapper entrypoint by default.

## Required Env

- `PB_USER`

You must also provide credentials in one of two ways.

## Credentials

### Option 1: Mount `api-keys.json`

Set:

- `PB_API_KEYS_PATH=/data/api-keys.json`

and mount your file there.

### Option 2: Generate credentials from env

Set:

- `PB_EXCHANGE`
- `PB_API_KEY`
- `PB_API_SECRET`
- `PB_API_PASSPHRASE` optional

The entrypoint renders a temporary `api-keys.json` and links it into the working tree so the existing live runtime can use it without code-path divergence.

## Config

### Option 1: Mounted config file

Set:

- `PB_CONFIG_PATH=/data/configs/live.json`

### Option 2: Inline config

Set:

- `PB_CONFIG_INLINE`

with a JSON or HJSON payload.

### Option 3: Schema defaults

If neither config env is set, the container starts from the in-code schema defaults.

## Optional Deployment Overrides

- `PB_APPROVED_COINS`
  Comma-separated list such as `BTC,ETH,SOL`, or a path accepted by the normal `--symbols` live CLI flag.
- `PB_LOG_LEVEL`
  For example `1`, `2`, `debug`, or `trace`.
- `PB_MONITOR_ENABLED`
  `true` / `false`
- `PB_MONITOR_ROOT`
  Override the monitor root directory, for example `/data/monitor`
- `PB_LOG_DIR`
  Override `logging.dir` for container deployments that mount a persistent logs volume

When `PB_CONFIG_PATH` is set, these overrides are translated into normal `passivbot live` CLI flags so the original config path stays intact and config-relative sidecar semantics are preserved.

When `PB_CONFIG_INLINE` is set, the entrypoint renders a temporary config file and starts `passivbot live` with that rendered config.

`PB_CONFIG_INLINE` and `PB_CONFIG_PATH` are mutually exclusive.

## File Logging

Live runs use Passivbot's canonical logging config.

- `logging.persist_to_file` defaults to `true`
- `logging.dir` defaults to `logs`
- log files are written as `${logging.dir}/${live.user}.log`

Container logs still go to stdout/stderr, and file persistence uses the same live logging handler rather than a wrapper-level tee.

For containerized deployments, set `PB_LOG_DIR` to point `logging.dir` at a mounted volume such as `/data/logs`.

## Docker Run Example

Using a mounted config and mounted API keys:

```bash
docker run --rm \
  -e PB_USER=bitget_01 \
  -e PB_CONFIG_PATH=/data/configs/live.json \
  -e PB_API_KEYS_PATH=/data/api-keys.json \
  -e PB_LOG_DIR=/data/logs \
  -e PB_MONITOR_ROOT=/data/monitor \
  -v "$(pwd)/configs:/data/configs:ro" \
  -v "$(pwd)/api-keys.json:/data/api-keys.json:ro" \
  -v "$(pwd)/logs:/data/logs" \
  -v "$(pwd)/monitor:/data/monitor" \
  passivbot-live:latest
```

Using env-generated credentials and schema defaults:

```bash
docker run --rm \
  -e PB_USER=bitget_01 \
  -e PB_EXCHANGE=bitget \
  -e PB_API_KEY=... \
  -e PB_API_SECRET=... \
  -e PB_APPROVED_COINS=BTC,ETH,SOL \
  passivbot-live:latest
```

## Docker Compose Example

See `container/examples/docker-compose.live.yml`.

## Notes

- The entrypoint uses `passivbot live`, not `python src/main.py`.
- The runtime contract is intentionally thin and platform-agnostic.
- File logging stays on the normal `logging.*` config path; the container wrapper does not implement its own parallel log persistence system.
- Hosted platforms should reuse this image and env contract instead of introducing platform-specific config snapshots.
