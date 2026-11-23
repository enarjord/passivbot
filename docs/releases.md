# Releases and development versions

The latest tagged release is **[v8.1.0](https://github.com/enarjord/passivbot/releases/tag/v8.1.0)**,
published on 2026-08-10. The [release notes](release_notes_v8.1.0.md) describe that tag.

Development branches contain subsequent changes and can differ in config compatibility. Read
`CHANGELOG.md` under **Unreleased** at the revision you intend to install. The v9 development
package reports `9.0.0.dev0` and uses config schema `v9.0.0`. Record the exact Git commit
alongside `passivbot --version` when comparing development checkouts.

| Version surface | Meaning |
|---|---|
| Release tag, such as `v8.1.0` | A fixed published revision with release notes |
| Package version (`passivbot --version`) | The package's declared version; record the Git commit too when using `master` |
| `config_version` | Config schema compatibility, independent of the package/release number |

This revision uses config schema **v9.0.0**. Use the
[v8 migration guide](v8_to_v9_migration.md) to identify supported inputs and prepare a
compatible config before switching to v9. Keep the original config, checkout, and environment
together for rollback. Use examples and documentation from the installed revision.

The v9 risk model is an incompatible behavior change: choose new loss-memory spans,
retune and re-backtest migrated configurations, and discard old optimizer fitness/checkpoints.
The declared development package number does not imply compatibility with v8 HSL or
auto-unstuck memory, and a schema change does not itself publish a package release.

## v9 release review

The [v9 release preview](release_notes_v9.0.0.md) summarizes the major changes and gives a
migration, validation, and rollback checklist.

### v8 rollback reference

The v8 `master` revision integrated into this candidate is
[commit `15d6404f3`](https://github.com/enarjord/passivbot/tree/15d6404f33bd5623d46a46e04ea694188ea7acea).
For an individual installation, retain the exact commit or tag that ran it together with its
matching configuration, native extension, environment, and launch command.

Immediately before the v9 cutover, fetch both branches and compare the current `origin/master`
SHA with the linked rollback reference. If `master` advanced, integrate its changes into v9,
update the reference, and qualify the resulting candidate. Verify that
`git merge-base --is-ancestor origin/master origin/v9` succeeds. Check Python 3.12 and 3.14,
Rust, the installed-wheel research smoke, the offline fake-live HSL restart, and the Docker
image smoke on the candidate. The Docker workflow supports manual dispatch when its PR path
filter skips the candidate. Follow the [release runbook](ai/runbooks/release.md) for sign-off.

## Choose an installation revision

A fresh clone selects the default branch. To evaluate the v9 development branch explicitly, use
`git clone --branch v9 https://github.com/enarjord/passivbot.git`. To install the tagged release,
select the tag before installing dependencies.

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
- [Earlier post-v8.1.0 implementation ledger](https://github.com/enarjord/passivbot/blob/0acfeaae6d70dc5527dc1e6ff56cb6e2ace06655/docs/development_history_since_v8.1.0.md)

Contributors should add net user-facing changes under `Unreleased`, combining superseded entries
for the same feature. Leave tagged sections faithful to their release. Version bumps, release
validation, tagging, and publication follow the [release runbook](ai/runbooks/release.md).
