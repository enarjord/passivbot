# Releases and development versions

The latest tagged release is **[v8.1.0](https://github.com/enarjord/passivbot/releases/tag/v8.1.0)**,
published on 2026-08-10. The [release notes](release_notes_v8.1.0.md) describe that tag.

`master` also contains subsequent changes. A normal `git clone` or a pull on `master` includes
those changes. The development package reports `8.2.0.dev0`; this is not a new tagged release. Read
[Unreleased in the changelog](https://github.com/enarjord/passivbot/blob/master/CHANGELOG.md#unreleased) for the net changes since the tag.
`Unreleased` does not mean unmerged or unavailable on `master`.

| Version surface | Meaning |
|---|---|
| Release tag, such as `v8.1.0` | A fixed published revision with release notes |
| Package version (`passivbot --version`) | The package's declared version; record the Git commit too when using `master` |
| `config_version` | Config schema compatibility, independent of the package/release number |

This revision uses config schema **v9.1.0**. V8 and v9.0.0 configurations require explicit
[v9 migration](v8_to_v9_migration.md) before ordinary loading. These schema numbers do not
imply corresponding published releases. Use
examples and documentation from the same revision as the installed runtime. A config saved by a
newer runtime may not load in an older tagged release; retain the original when upgrading.
See [Config Workflow](config_workflow.md) for migration and review steps.

The v9 risk model is an incompatible behavior change: choose new loss-memory spans,
retune and re-backtest migrated configurations, and discard old optimizer fitness/checkpoints.
The declared development package number does not imply compatibility with v8 HSL or
auto-unstuck memory, and a schema change does not itself publish a package release.

## Choose an installation revision

A fresh clone defaults to `master`. To install the tagged release instead, select it before
installing dependencies:

```bash
git clone https://github.com/enarjord/passivbot.git
cd passivbot
git switch --detach v8.1.0
```

Then follow [Installation](installation.md), using the guide and examples at that revision.
A detached release checkout does not advance with `git pull`. To follow development again,
commit or back up local changes, switch to `master`, and follow the update instructions.

Record both the package version and source revision when reporting a problem:

```bash
passivbot --version
git rev-parse HEAD
git status --short
```

Review status output before sharing it; include only relevant public code changes.

## Release history

- [v8.1.0 — 2026-08-10](release_notes_v8.1.0.md)
- [v8.0.0 — 2026-07-14](release_notes_v8.0.0.md), including the v7 upgrade boundary
- [Complete changelog](https://github.com/enarjord/passivbot/blob/master/CHANGELOG.md)
- [Earlier post-v8.1.0 implementation ledger](development_history_since_v8.1.0.md)

Contributors should add net user-facing changes under `Unreleased`, combining superseded entries
for the same feature. Leave tagged sections faithful to their release. Version bumps, release
validation, tagging, and publication follow the [release runbook](ai/runbooks/release.md).
