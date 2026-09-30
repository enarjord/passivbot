# Live Logging Overhaul Progress

This file is the historical progress ledger for
`docs/plans/live_logging_overhaul_plan.md`. It is not the active backlog or the
resume point. Read `docs/plans/live_logging_overhaul_current_status.md` first.
Keep this history for evidence, but do not continue the former one-entry-per-PR
operational diary.

## Update Policy

- Add an entry only when a logging-overhaul milestone materially changes the
  architecture, closes a completion criterion, or records evidence needed to
  understand a later design decision.
- Do not append routine PR, deployment, restart, reviewer, or smoke chronology
  here. Runtime actions still require the bounded durable record and
  public/private evidence split defined by step 13 of
  `live_logging_overhaul_pr_loop_workflow.md`.
- Adjacent trading, performance, restart, process, report, and operator-tool
  work belongs in its own plan or the live-operations backlog even when it uses
  the event pipeline.
- Keep any new entry factual and compact: milestone, scope, validation, and the
  remaining finite gap.
- Do not use this file for design churn; unresolved design details belong in the
  plan or a focused handoff doc.

The entries below predate this focus rule and are retained unchanged as
historical evidence. Their branch names, “active follow-up” language, deployment
state, and next-slice statements are not current instructions. Unresolved
deployment blockers remain safety evidence: revalidate them against the target
host's exact deployed SHA, local config, and intended target delta before any
pull or restart.

## Latest Canonical Merge, Deployment Deferred (PR #1356)

- PR #1356 merged exact approved head `b7047af8c8` as canonical
  `248895745e85077cdec51f97415d8c8b91b937d5` after exact-head Hermes
  approval and green Python/Rust CI. It redacts candle remote-fetch callbacks,
  HLCV progress logs, archive diagnostics, structured events, and fake-live
  traces while retaining bounded classification, timing, and correlation.
- VPS5 remains at PR #1348 canonical `acccfdac51d2`. Both active configs still
  contain legacy `initial_entry_exec_max_market_dist_pct=0.005`; pulling the
  accumulated master delta would activate the intervening account-wide
  replacement-churn gate. Deployment is therefore deferred pending an explicit
  config/rollout decision rather than treated as a failed logging rollout.
- A read-only process preflight found five stable exact processes with zero
  hard failures and one process remaining in I/O wait throughout the short
  three-sample window, without a hard process-contract failure. No pull,
  restart, signal, authenticated exchange request, or manufactured event
  occurred. The active follow-up redacts local candle cache/index/lock
  diagnostics; direct live consumer diagnostics remain a later slice.

## Latest Canonical Deployment (PR #1348)

- PR #1348 merged exact approved head `3adba8fd3d` as canonical
  `acccfdac51d27c5fde114821c939cd77b933685f` after exact-head Hermes
  approval, green Python/Rust CI, and a direct-consumer correction retaining
  bounded shutdown `error_type` evidence in live smoke reports.
- VPS5 fast-forwarded tracked-clean and restarted only configured panes
  `%358`-`%362`; the bounded restart window proved complete five-bot shutdown
  and startup identity with zero hard failures, hard text-log matches,
  monitor warnings/errors, or event-pipeline integrity failures. Protected
  `misc:0.0` remained `%8`/PID `434835`.
- A 2026-07-23 local-only preflight found the checkout still exact and
  tracked-clean at the same merge commit, five stable expected PIDs, and zero
  missing, duplicate, extra, config, or scan failures. The short sample
  retained transient I/O-wait pressure without a hard process-contract
  failure. No authenticated exchange request, manufactured event, or process
  action occurred.

## Previous Canonical Deployment (PR #1347)

- PR #1347 merged exact approved head `5ef012ed5f` as canonical
  `67416ee4f7fef2dc3ffac001d50e82c0322ee72b` after exact-head Hermes
  approval, green Python/Rust CI, and finding-free built-in Codex and
  independent Sol reviews.
- VPS5 guarded-prepared tracked-clean from `4192e709d46` without a Rust build;
  the Rust source fingerprint/stamp and compiled artifact remained unchanged.
  The exact-target orchestrator gracefully restarted only panes `%358`-`%362`;
  old PIDs `1079121/1079130/1079124/1079133/1079127` exited and replacement
  PIDs `1080755/1080764/1080758/1080767/1080761` relaunched and verified
  without force or broad-pattern signals.
- The integrated 120-second smoke was hard-green with complete shutdown and
  startup identity, zero hard failures, zero hard text-log or attention
  matches, and zero monitor warnings or errors. A bounded settled smoke found
  all five exact processes stable with no PID churn, persistent uninterruptible
  state, failed fill refresh, remote-call failure, event-pipeline integrity
  issue, or hard diagnostic evidence. The checkout remained exact and
  tracked-clean, and protected `misc:0.0` stayed `%8`/PID `434835`. No direct
  authenticated exchange call or event was manufactured. The next slice
  redacts shutdown-stage failure diagnostics; candle refresh/cache-maintenance
  diagnostics remain the next candidate after that slice.

## Previous Canonical Deployment (PR #1346)

- PR #1346 merged exact approved head `377cb1dc60` as canonical
  `4192e709d46d3ef9516025e121ddad15c0d0cd6e` after exact-head Hermes
  approval, green Python/Rust CI, and finding-free built-in Codex and
  independent Sol reviews.
- VPS5 guarded-prepared tracked-clean from `49cb68e56b2` without a Rust build;
  the Rust source fingerprint/stamp and compiled artifact remained unchanged.
  The exact-target orchestrator gracefully restarted only panes `%358`-`%362`;
  old PIDs `1077958/1077967/1077961/1077970/1077964` exited and replacement
  PIDs `1079121/1079130/1079124/1079133/1079127` relaunched and verified
  without force or broad-pattern signals.
- The integrated 120-second smoke was hard-green with complete shutdown and
  startup identity, zero hard failures, zero hard text-log or attention
  matches, and zero monitor warnings or errors. A bounded settled check found
  all five exact processes stable, the checkout exact and tracked-clean, and
  protected `misc:0.0` preserved as `%8`/PID `434835`. No direct authenticated
  exchange call or event was manufactured. The next redaction slice removes
  arbitrary exception values from fill-history refresh diagnostics.

## Previous Canonical Deployment (PR #1345)

- PR #1345 merged exact approved head `e9322f9b50` as canonical
  `49cb68e56b20d71fbe42d33f31906d3a9c793e90` after exact-head Hermes
  approval, green Python/Rust CI, and finding-free built-in Codex and
  independent Sol reviews.
- VPS5 guarded-prepared tracked-clean from `986e5d52f8` without a Rust build;
  the Rust source fingerprint/stamp and compiled artifact remained unchanged.
  The exact-target orchestrator gracefully restarted only panes `%358`-`%362`
  as PIDs `1077958/1077967/1077961/1077970/1077964`; all five prior processes
  exited and all five targets relaunched and verified without force or
  broad-pattern signals.
- The integrated 120-second smoke was hard-green with complete shutdown and
  startup identity, zero hard failures, zero hard log or attention matches,
  and zero monitor warnings or errors. Final three-sample target verification
  retained five stable exact processes, a tracked-clean checkout, and
  protected `misc:0.0` `%8`/PID `434835`. No direct authenticated exchange
  call or event was manufactured. The next redaction slice removes arbitrary
  caught exception values from best-effort live event emitter diagnostics.

## Previous Canonical Deployment (PR #1344)

- PR #1344 merged exact approved head
  `6a4fe26f90e8cb25f0dc09ecc39a067658addfa8` as canonical
  `986e5d52f88692d1b6531bc38c307352c08e9cb9` after exact-head Hermes approval,
  green Python/Rust CI, and a finding-free built-in Codex review. Independent
  Sol review approved the semantic predecessor `f442e4a66e`; the final delta
  only corrected the active handoff.
- VPS5 guarded-prepared tracked-clean from `7e26a9062a8` without a Rust build;
  the Rust source fingerprint/stamp and compiled artifact remained unchanged.
  The exact-target orchestrator gracefully restarted only panes `%358`-`%362`
  as PIDs `1076279/1076288/1076282/1076291/1076285`; all five exited,
  relaunched, and verified without force or broad-pattern signals.
- The integrated smoke correctly stayed red on one natural KuCoin
  authoritative-balance `RequestTimeout` and had zero hard log matches or
  monitor warnings/errors. A strictly post-incident settled window had a green
  internal smoke contract with zero hard problem events, hard log matches, or
  monitor warnings/errors. Final three-sample target verification retained five
  stable exact processes in state `R`, a tracked-clean checkout, preserved
  untracked artifacts, and protected `misc:0.0` `%8`/PID `434835`. No direct
  authenticated exchange call or event was manufactured. The next redaction
  slice removes arbitrary exception text from legacy monitor error events and
  WebSocket reconnect diagnostics.

## Previous Canonical Deployment (PR #1343)

- PR #1343 merged exact approved head
  `2ee1382781e28e8e1d3341a43e22ef527b86f283` as canonical
  `7e26a9062a88ecf211729d7d718fb4530630c4ba` after exact-head Hermes approval,
  green Python/Rust CI, and a finding-free built-in Codex review.
- VPS5 guarded-prepared tracked-clean from `644d058f3975` without a Rust build;
  the Rust source fingerprint/stamp stayed
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`.
- The exact-target orchestrator gracefully restarted only panes `%358`-`%362`
  as PIDs `1075081/1075090/1075084/1075093/1075087`; shutdown, startup, and
  stable target coverage were complete without force or broad-pattern signals.
- The integrated 120-second smoke was hard-green with zero hard failures, log
  attention matches, monitor warnings, or monitor errors. Repository,
  shutdown, startup, target, and smoke-contract gates all passed; final
  three-sample target verification retained five stable exact processes and a
  tracked-clean checkout, while protected `misc:0.0` stayed `%8`/PID `434835`.
  No direct authenticated exchange call or event was manufactured. The next
  logging slice removes arbitrary exception and fallback-reason text from EMA
  diagnostics upstream of every sink.

## Previous Canonical Deployment (PR #1342)

- PR #1342 merged exact approved head
  `8bae713d56f682744a09033f86a46a546b92ca3e` as canonical
  `644d058f3975a2772f96bf10281f3873dca7112c` after exact-head Hermes approval,
  green Python/Rust CI, and a finding-free built-in Codex review.
- VPS5 guarded-prepared tracked-clean from `524a6d2795` without a Rust build;
  the Rust source fingerprint/stamp stayed
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`.
- The exact-target orchestrator gracefully restarted only panes `%358`-`%362`
  as PIDs `1073251/1073259/1073253/1073261/1073255`; shutdown, startup, and
  stable target coverage were complete without force or broad-pattern signals.
- The integrated smoke retained one natural KuCoin `RequestTimeout` and
  correctly stayed non-green. A strictly post-timeout settled window was
  hard-green with zero hard problem events, log hard matches, monitor warnings,
  or monitor errors. All five exact processes were stable, the tracked checkout
  remained clean, and protected `misc:0.0` stayed `%8`/PID `434835`. No direct
  authenticated exchange call or event was manufactured. The next logging
  migration consolidates noncritical market-snapshot diagnostic warnings under
  their existing structured event.

## Previous Canonical Deployment (PR #1341)

- PR #1341 merged exact approved head
  `d56ffc3617d9ebed4e0ebb3b98d4b6a80fb2b89c` as canonical
  `524a6d2795015afee7cc1dd916880e4e48a6e13b` after exact-head Hermes approval,
  green Python/Rust CI, and a finding-free built-in Codex semantic review.
- VPS5 guarded-prepared tracked-clean from `bc31fafdbdf` without a Rust build;
  the Rust source fingerprint/stamp stayed
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`.
- The exact-target orchestrator gracefully stopped all five old bot PIDs and
  relaunched only panes `%358`-`%362` as PIDs
  `1072420/1072429/1072423/1072432/1072426`. Shutdown, startup, and stable target
  coverage were complete, with no force or broad-pattern signal.
- The 120-second settled smoke was hard-green with zero hard failures, log
  attention matches, monitor warnings, or monitor errors. Final passive states
  were `R/R/R/R/S`; the checkout remained clean, pane parents remained stable,
  and protected `misc:0.0` stayed `%8`/PID `434835`. No direct authenticated
  exchange call or event was manufactured. The next logging migration removes
  duplicate legacy warnings from existing pre-create snapshot skip events.

## Previous Canonical Deployment (PR #1340)

- PR #1340 merged exact reviewed head
  `d9e88d6da9282bc53a355dc3ce18a9cba6de45eb` as canonical
  `bc31fafdbdf045f9c003e49fa6877b721c834600` after exact-head Hermes approval,
  green Python/Rust CI, and a finding-free built-in Codex review.
- VPS5 guarded-prepared tracked-clean from `3311085dce` without a Rust build,
  restart, or signal; the Rust source fingerprint/stamp remained
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`.
- The bounded incident bundle retained exact matching log `scan_cost` in its
  command result, manifest, and archived full smoke report: eight `full_scan`
  files, 3,787 selected records, 573,578 known physical/decoded bytes, and
  847.006 ms. It correctly remained non-green on one natural KuCoin
  authoritative-balance `RequestTimeout`.
- All five exact bot PIDs and pane parents remained intact; settled passive
  sampling retained normal `R`/`S` states and protected `misc:0.0` `%8`/PID
  `434835`. No direct exchange call, manufactured event, build, restart, or
  process signal occurred. The next logging migration replaces the remaining
  bulk open-order snapshot stdlib count lines with a bounded structured event.

## Previous Canonical Deployment (PR #1339)

- PR #1339 merged exact reviewed head
  `c66a306a4b20485d31e3806913c1f2d90279ecb7` as canonical
  `3311085dce8b7540f5a26e809229180d5f784c25` after exact-head Hermes approval
  and green Python/Rust CI. Built-in Codex was additionally requested and had
  posted no finding when the proportional temporary gate merged the PR.
- VPS5 guarded-prepared tracked-clean from `f9ceb96784` without a Rust build,
  restart, or signal; the Rust source fingerprint/stamp remained
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`.
- The bounded direct smoke reported eight `full_scan` log files, 3,442 selected
  records, 512,497 known physical/decoded bytes, and 742.868 ms. Its non-green
  verdict correctly retained two natural KuCoin authoritative-balance
  `RequestTimeout` events rather than masking them.
- The incident archive retained exact log scan cost in `smoke_report.json`:
  eight `full_scan` files, 3,446 selected records, 513,283 known
  physical/decoded bytes, and 642.116 ms. The manifest and command-result
  summaries omitted the same field, motivating the active
  `codex/live-incident-log-scan-cost` projection-only slice.
- All five unchanged bot PIDs settled to `R/R/R/S/R`; pane parents
  `%358`-`%362` and protected `misc:0.0` `%8`/PID `434835` remained unchanged.
  No direct exchange call, manufactured event, build, restart, or process
  signal occurred.

## Previous Canonical Deployment (PR #1338)

- PR #1338 merged exact reviewed head
  `8924c4ff30f96b8ea3dbf58fc6a97d1faa54514d` as canonical
  `f9ceb9678448201af0c0cc5f40f889b661d3021c` after exact-head Hermes approval
  and green Python/Rust CI. Built-in Codex was additionally requested and had
  posted no finding when the proportional temporary gate merged the PR.
- VPS5 guarded-prepared tracked-clean from `848eb60c50` without a Rust build,
  bot restart, or signal; the Rust source fingerprint/stamp remained
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`.
- The bounded five-minute incident bundle was `ok=true`, found all five exact
  processes, retained a clean `f9ceb967` repository, and projected identical
  time-window scan cost through the full report, manifest, and command result:
  six `seek_tail` files, 8,848 records, 15,501,809 known physical/decoded bytes,
  and 3,515.111 ms.
- The immediate post-bundle identity sample observed four unchanged bot PIDs
  in transient `D` state. Thirty seconds later those same PIDs were
  `R/R/S/R/R`; pane parents `%358`-`%362` and protected `misc:0.0`
  `%8`/PID `434835` remained unchanged. No direct exchange call, manufactured
  event, build, restart, or process signal occurred. The same bundle scanned
  eight selected text logs but exposed no byte/read-method cost, motivating the
  active `codex/live-log-scan-cost` slice.

## Previous Canonical Deployment (PR #1329 After PR #1337)

- PR #1337 merged exact reviewed head
  `51a82c23230daf4aea5ec68784ef7168293b408e` as canonical
  `7d463dca56e1b29840954876a25a3f4d0e5df961` after exact-head Hermes and
  built-in Codex approval plus green Python/Rust CI. VPS5 fast-forwarded
  tracked-clean from `e0927ed86f` without a Rust build or process signal.
- Its settled bounded five-minute smoke was `ok=true` with zero hard failures,
  all five exact processes, and known scan-cost evidence: six `seek_tail`
  files, 10,388 records, 18,777,444 physical/decoded bytes, and 5,751.099 ms.
  The immediate sample retained a natural KuCoin timeout and correctly stayed
  red until that event aged out.
- External PR #1329 then merged exact mechanical-integration head
  `6bd2eb0a0a49d21107a2c45ff9784e737d5c4d1c` as canonical
  `848eb60c502b7af21dd7aa52f50f86aece552bd9`. The maintainer explicitly
  approved contributor CI; Hermes and Python/Rust CI were green, and the
  target-relative result remained the reviewed one-line aware-UTC timestamp
  fix. VPS5 fast-forwarded tracked-clean without a Rust build or restart; the
  startup-only maintenance-path change did not warrant disrupting running
  bots.
- Bot PIDs `1066081/1066091/1066084/1066093/1066087`, pane parents
  `%358`-`%362`, and protected `misc:0.0` `%8`/PID `434835` remained unchanged.
  A later natural KuCoin orders timeout kept the immediate post-pull sample
  red. The final settled five-minute smoke was `ok=true` with zero hard
  failures, zero failed remote calls, all five exact processes, a tracked-clean
  repository at `848eb60c`, and six `seek_tail` files reading 9,471 records and
  17,038,544 known physical/decoded bytes in 4,021.646 ms. No direct exchange
  call, manufactured event, restart, or process signal occurred. The remaining
  incident-bundle time-window scan lacks exact byte/read-method cost, motivating
  the active `codex/live-incident-scan-cost` slice.

## Previous Canonical Deployment (PR #1333)

- PR #1333 merged exact reviewed head
  `ce5a24d72ea811c6b04a376bbd9fcd228ab4c9af` as canonical
  `e0927ed86f0b10ed8187d8e6a523017baefb98b3` after exact-head Hermes and
  built-in Codex reviews plus green Python/Rust CI. VPS5 fast-forwarded
  tracked-clean from `02fb43f639` without a Rust build; the Rust source
  fingerprint/stamp remained unchanged.
- The bounded five-minute event query reported five files, 8,133 records,
  13,926,420 known physical/decoded bytes, and 4,793.586 ms. The matching
  performance report reported six files, 8,661 records, 14,740,828 known
  physical/decoded bytes, and 4,781.454 ms with zero errors/warnings. Both
  reports were `ok=true` and used only bounded `seek_tail` reads.
- No bot restart or signal was required. Bot PIDs
  `1066081/1066091/1066084/1066093/1066087`, pane parents `%358`-`%362`, and
  protected `misc:0.0` `%8`/PID `434835` remained unchanged; all five bots
  were `Rl+` and the checkout remained tracked-clean. A bounded smoke-report
  baseline took 4.29 seconds but exposed no byte/read-method cost, motivating
  the active diagnostic-only `codex/live-smoke-scan-cost` slice. No direct
  exchange call or event was manufactured.

## Previous Canonical Deployment (PR #1335)

- PR #1335 merged exact reviewed head
  `c4d5b8e55f6fd453fd707999ae74ec2dd127e55d` as canonical
  `02fb43f6398fc9edba64849cf2ed0bf0f7a6af09`. VPS5 fast-forwarded
  tracked-clean from `97aa36da4c` without a Rust build; the Rust source
  fingerprint/stamp remained unchanged.
- The guarded restart replaced bot PIDs
  `1063302/1063311/1064329/1063314/1063308` with
  `1066081/1066091/1066084/1066093/1066087` under unchanged pane parents. A
  post-action three-sample target report confirmed all five targets stable,
  with no missing, duplicate, or extra process; protected `misc:0.0` remained
  `%8`/PID `434835`.
- The immediate smoke retained a natural KuCoin authoritative-refresh
  `RequestTimeout` and remained red. The fresh settled two-minute smoke was
  hard-green with `47/47` account-critical and `184/185` remote calls,
  successful latest cycles, a clean event pipeline, and five stable processes.
  The only retained remote failure was a non-hard OHLCV `RequestTimeout`. No
  direct exchange call or event was manufactured. Active
  `codex/live-artifact-scan-cost` remains read-only tooling and requires no bot
  restart.

## Previous Canonical Deployment (PR #1334)

- Emergency PR #1334 merged exact reviewed head
  `1fb218a61999ed1cf06ba1974407dbfe350ed0ea` as canonical
  `97aa36da4c9a9885c840ea48590837e28e5b8069`. VPS5 fast-forwarded
  tracked-clean from `77b97ab8c7` without a Rust build; the Rust source
  fingerprint/stamp and artifact SHA-256 remained unchanged.
- The recovery relaunched only stopped Gate.io pane `%360`; its bot became PID
  `1064329` under unchanged pane parent `856364`. The other four bot PIDs
  remained `1063302/1063311/1063314/1063308`, and protected `misc:0.0`
  remained `%8`/PID `434835`.
- Stable three-sample exact-target validation retained five configured
  processes with no missing, duplicate, or extra target. The final fresh
  two-minute smoke was `ok=true` with zero hard failures and five stable
  processes. An earlier wider smoke retained an unrelated natural KuCoin
  `RequestTimeout` and was not mislabeled green; the final window observed one
  non-hard recovered KuCoin `InvalidNonce`. No direct exchange call or event
  was manufactured. Active `codex/live-artifact-scan-cost` remains read-only
  tooling and requires no bot restart.

## Previous Canonical Deployment (PR #1331)

## Previous Canonical Deployment (PR #1328)

- PR #1328 merged as canonical
  `c0386ff5673d93b732786cf12c4cd48f6a381767` after exact-head Hermes and
  built-in Codex reviews plus green Python/Rust CI. VPS5 fast-forwarded
  tracked-clean from `0a57187ff9f0def7eb4976721f5b04d17f03fb74` without a
  Rust build.
- The guarded runner gracefully restarted only exact panes `%358`-`%362`.
  Old bot PIDs `1056607/1056616/1056610/1056619/1056613` became
  `1057982/1057991/1057985/1057994/1057988`; protected `misc:0.0` stayed
  `%8`/PID `434835`. The two-minute smoke was hard-green with complete five-bot
  lifecycle evidence and zero hard, monitor, or text-log failures.
- Natural `snapshot.built` evidence exposed that the secret-key sanitizer
  replaced `signature_row_count` with `[redacted]`. The performance report
  therefore classified all seven observed summaries as malformed and produced
  zero freshness metrics. No direct exchange call or event was manufactured.
  Merged PR #1331 renamed only the public diagnostic/report count fields so the
  values remain numeric without weakening secret redaction.

## Previous Canonical Deployment (PR #1326)

- PR #1326 merged as canonical
  `0a57187ff9f0def7eb4976721f5b04d17f03fb74` after exact-head Hermes and
  built-in Codex reviews plus green Python/Rust CI. VPS5 fast-forwarded
  tracked-clean from `ac9ff15029` without a Rust build, bot restart, or signal.
- A bounded offline 240-minute/eight-symbol HSL benchmark matched compact and
  dense-reference final state and all 240 replay samples, emitted both exclusive
  stage taxonomies, and reported zero network/cache/latch/monitor side effects.
- Exact bot PIDs `1056607/1056616/1056610/1056619/1056613` and protected
  `misc:0.0` `%8`/PID `434835` remained unchanged. No direct exchange call or
  event was manufactured. The active freshness slice projects completed-candle
  readiness from frozen planning signatures into bounded diagnostics.

## Previous Canonical Deployment (PR #1325)

## Previous Canonical Deployment (PR #1322)

- PR #1322 merged as canonical
  `881978dec502296c4ab990c34bf06fdf74f024b2`. VPS5 fast-forwarded
  tracked-clean from `8b433cc22` without a Rust build, bot restart, or process
  signal.
- The exact-file Hyperliquid attribution smoke scanned 26 artifacts and
  `15763500` bytes, read 51 fill records, returned 37 trailing fills and zero
  warnings, and reconciled the previous `1731` ms manifest/log skew into one
  four-source runtime cohort. Smaller explicit bounds failed closed before the
  successful run; temporary scanner PIDs from interrupted commands were
  identity-checked and terminated exactly without touching bots or panes.
- Final local-only target sampling was hard-green with five expected/stable
  processes and no missing, duplicate, extra, or temporary scanner process.
  One transient `D` observation recovered naturally to `R=5`; protected
  `misc:0.0` remained `%8`/PID `434835`. No direct exchange call or event was
  manufactured.

## Previous Canonical Deployment (PR #1312)

- PR #1312 merged as canonical `8b433cc22b087b0efab51ba2bcf003f1e2b31806`.
  Guarded tracked-clean fast-forwarded `30870252` to that exact merge without a
  Rust build; source fingerprint/stamp remained `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`.
- The guarded runner gracefully restarted only panes `%358`-`%362` without
  force. Bot PIDs `1044483/1044492/1044486/1044495/1044489` became
  `1048663/1048672/1048666/1048675/1048669`; pane parents and protected
  `misc:0.0` `%8`/PID `434835` were unchanged.
- The complete five-bot lifecycle window retained one natural hard event. A
  fresh settled window had zero smoke hard failures, log errors, or monitor
  errors, and the ordinary two-minute report was hard-green with `46/46`
  account-critical calls successful and no latest degraded cycle. Final exact
  target sampling was 3/3 stable with no extras or issues. No direct exchange
  request or event was manufactured.

## Previous Canonical Deployment (PR #1315)

- PR #1315 merged as canonical `308702523760ae7a0b309419ae1616b0a4938721`.
  Guarded tracked-clean fast-forwarded `fc9dad83` to that exact merge after the
  target Rust fingerprint/stamp
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449` matched;
  no Rust build, bot restart, or signal was needed.
- The bounded Hyperliquid attribution smoke selected 13 files/`3695111` bytes,
  scanned 50 fills (37 trailing), and had zero warnings. All 37 legacy
  first-ingestion and 37 producer-attribution records remained correctly
  unattributed; the full manifest `1784414326269` ms and matching prefix log
  `1784414328000` ms exposed the `1731` ms duplicate-runtime condition.
- Bot PIDs `1044483/1044492/1044486/1044495/1044489` and protected `misc:0.0`
  `%8`/PID `434835` were unchanged. Final exact sampling retained normal `R/S`
  states and a tracked-clean checkout; no direct exchange call or event was
  manufactured.

## Previous Canonical Deployment (PR #1321)

- PR #1321 merged as canonical
  `fc9dad83cd3ecf51cae15e8dda66afb7cfb895a1`. It completes bounded latest
  shutdown lifecycle diagnostics without changing producers, trading, or
  process behavior.
- VPS5 fast-forwarded tracked-clean at that head. Bot PIDs
  `1044483/1044492/1044486/1044495/1044489` were unchanged, and `misc:0.0`
  remained pane `%8`/PID `434835`.
- The initial smoke retained four natural KuCoin degraded cycles/timeouts. The
  settled two-minute smoke was `ok=true`, `hard_failures=0`, with `44/44`
  account-critical and `243/243` remote calls successful and five configured
  processes stable. No Rust build, bot restart, signal, direct exchange call,
  or event was manufactured.

## Previous Canonical Deployment (PR #1320; Through PR #1319)

- PR #1320 merged as canonical
  `bc2c90267be418344ec883fcf17fae856bc568cd` after exact-head Hermes approval
  and green Python/Rust CI. It exposes event-pipeline drops, sink errors, and
  unexpectedly dead workers through a separate bounded diagnostic integrity
  verdict while preserving general smoke and trading/process semantics.
- VPS5 fast-forwarded tracked-clean from PR #1314 without a Rust build, bot
  restart, or process signal. Exact five-bot target sampling remained 3/3
  stable with unchanged PIDs and no extras or issues; protected `misc:0.0`
  remained `%8`/PID `434835`.
- All five latest pipeline snapshots were integrity-green with zero drops, sink
  errors, unexpectedly dead workers, backlog, unfinished work, or degraded
  counters. The wider report retained natural KuCoin `RequestTimeout` events;
  the latest affected cycle subsequently completed successfully without
  intervention. No direct exchange request, process action, or event was
  manufactured. The active follow-up makes bounded shutdown evidence complete
  per distinct bot rather than trusting aggregate lifecycle counts.

## Previous Canonical Deployment (PR #1314; Through PR #1319)

- PR #1319 merged as canonical `84c8e040334820ccc049787c82048358e18179c6`.
  Its offline fake-live shutdown-clock repair changes no live runtime, producer,
  exchange, strategy, or process behavior, so no VPS action was warranted.
- PR #1314 then merged as canonical
  `0dbfbca74b029353a0e11888e71077fa711835ff`. It records immutable runtime,
  Rust, config, and fill-attribution provenance without changing strategy,
  order, risk, or exchange behavior.
- VPS5 fast-forwarded tracked-clean from `eb82e256c2`. A deliberately wrong
  Rust fingerprint failed closed before build or signal. The exact
  target-derived `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`
  fingerprint rebuilt and verified the loaded extension after the non-login
  shell PATH was explicitly bound to `/root/.cargo/bin`.
- The guarded runner gracefully restarted only exact panes `%358`-`%362`
  without force. Old PIDs `1042130/1042139/1042133/1042142/1042136` became
  `1044483/1044492/1044486/1044495/1044489`. The bounded
  `1784414260464..1784414921350` window selected 10/1,011 segments totaling
  `73414648` bytes and retained complete lifecycle evidence, but correctly
  remained red on one natural KuCoin authoritative-refresh `RequestTimeout`.
- KuCoin recovered without intervention. The settled
  `1784414681768..1784415199226` report was hard-green with zero hard problem
  events, log matches, monitor errors, or process failures. Final target
  sampling was 3/3 stable with no extras or issues; protected `misc:0.0`
  remained `%8`/PID `434835`. No direct exchange request or event was
  manufactured. The active follow-up exposes event-pipeline loss as a separate
  diagnostics-only integrity verdict while preserving top-level smoke
  semantics.

## Previous Canonical Deployment (PR #1317)

- PR #1317 merged as canonical `eb82e256c2dfeac29af158f389f93a7ddba8eae2`.
  It adds bounded Hyperliquid unified-account composition diagnostics without
  changing scalar balance, exchange calls, planning, orders, risk, or console
  admission.
- VPS5 fast-forwarded tracked-clean from `e7fe7f79` without a Rust rebuild and
  gracefully restarted only exact panes `%358`-`%362` without force. Old PIDs
  `1040903/1040911/1040905/1040914/1040908` became
  `1042130/1042139/1042133/1042142/1042136`.
- The exact `1784407239491..1784407888217` window selected 10/1,011 event
  segments totaling `72553076` bytes, retained all five shutdown/startup
  cohorts, and was hard-green across monitor, text-log, repository, and target
  gates. Delayed 3/3 sampling recovered one transient GateIO `D` observation to
  final `R=4,S=1`, with five stable PIDs and no extras. Protected `misc:0.0`
  remained `%8`/PID `434835`. No direct exchange call or event was manufactured.
- The exact merge is the base for the offline fake-live post-panic
  observability regression slice.

## Previous Canonical Deployment (PR #1316)

- PR #1316 merged as canonical `e7fe7f796fb76a829003933dc7e5d937c6df8c64`.
  It adds bounded Binance CCXT unified composition diagnostics without changing
  scalar balance, exchange calls, planning, orders, risk, or console admission.
- VPS5 fast-forwarded tracked-clean from `a0db60f9` without a Rust rebuild and
  gracefully restarted only exact panes `%358`-`%362` without force. Old PIDs
  `1038760/1038769/1038763/1038772/1038766` became
  `1040903/1040911/1040905/1040914/1040908`.
- The exact `1784404124757..1784404777859` window selected 7/1,012 event files
  totaling `19420705` bytes, retained five stopping/stopped/startup cohorts, and
  had zero hard, monitor, or text-log issues. Delayed 3/3 target sampling was
  `R=4,S=1` with no extras; `misc:0.0` remained `%8`/PID `434835`. No direct
  exchange call or event was manufactured.
- The exact merge was the base for the Hyperliquid unified-only composition slice.

## Previous Canonical Deployment (PR #1313)

- PR #1313 merged as canonical `a0db60f9ca97dbc5b9b37aa3230fce97eb0917ce`.
  It adds bounded OKX asset composition to `balance.changed` without changing
  scalar balance, exchange calls, planning, orders, risk, or console admission.
- VPS5 fast-forwarded cleanly from `32156cbc`. A wrong caller-side Rust
  fingerprint failed closed after the repository move and before any bot
  signal; the same-head target-derived proof then passed without a Rust build
  and matched the loaded extension stamp and artifact.
- The guarded executor gracefully restarted only the five exact configured
  panes without force. The exact `1784401342000..1784402056447` lifecycle
  window selected 6/1,011 managed segments totaling `39263703` bytes, recovered
  all five stopping/stopped/startup cohorts, and was hard-green across target,
  repository, monitor, and text-log gates. A broader pre-action window retained
  one earlier natural KuCoin `RequestTimeout`, correctly separating it from the
  restart lifecycle rather than suppressing it.
- Final targets were 3/3 stable with new exact bot PIDs
  `1038760/1038769/1038763/1038772/1038766`; pane parents and protected
  `misc:0.0` `%8`/PID `434835` were unchanged. Natural post-restart OKX balance
  events carried the bounded two-asset composition. No direct exchange request
  or event was manufactured.
- The exact merge is the base for the Binance CCXT unified composition slice.

## Previous Canonical Deployment (PR #1311)

- PR #1311 merged as canonical `32156cbc251d666902f20b8b000a9a1dfe05a0a2`.
  It carries bounded hard-only problem-event evidence into concise smoke,
  brief, and incident-bundle projections without changing runtime event
  production or trading behavior.
- VPS5 prepared the exact clean merge without a Rust rebuild or bot restart.
  The immediate bounded smoke retained one natural KuCoin positions timeout;
  the settled five-minute smoke was hard-green with an empty hard sample. A
  bounded incident bundle for that natural timeout carried the same hard-only
  evidence in the command result and archived manifest. The five configured
  bot panes and `misc:0.0` remained unchanged.
- The exact merge is the base for the following balance-composition slice.

## Previous Canonical Deployment (PR #1310)

- PR #1310 merged as canonical `5d06887b78c2790efd15e1bd67bae6b3f5d96636`.
  It added full-report `hard_problem_events` with authoritative `count`, a
  bounded chronological `sample`, and explicit `retained`/`truncated` counts
  while preserving the existing mixed sample, verdicts, recovery
  classification, and runtime behavior.
- VPS5 prepared that exact tracked-clean merge without a Rust rebuild or bot
  restart. A bounded read-only smoke was hard-green and exposed
  `hard_problem_events={count:0,retained:0,truncated:0,sample:[]}`; all five
  pane parents and protected `misc:0.0` remained unchanged.

## Previous Canonical Deployment (PR #1309; Current Through PR #1299)

- PR #1309 merged as canonical
  `50c37db6049206634b62f45798a8b240a035e3b5` after exact-current-head Hermes
  approval and green Python/Rust CI. It removes raw sink exception text while
  retaining stable sink, reason, exception-type, counter, and timing evidence.
- VPS5 prepared the exact merge without a Rust rebuild and gracefully stopped,
  exited, relaunched, and verified all five configured panes without force. The
  exact `1784339097380..1784339763543` window recovered complete shutdown and
  startup cohorts and left every bot running, but correctly remained red on two
  hard structured events, including a real KuCoin positions-fetch
  `RequestTimeout`. Only one hard classification remained in the bounded mixed
  problem-event sample, exposing the active hard-only retention follow-up.
- Canonical master later advanced through behavior-changing PR #1299 to
  `f1ae7970393e8299d1b0a98c8ff68d42adddd2d0`. VPS5 prepared that exact head,
  restarted only the same five verified panes, and returned hard-green over
  `1784392681320..1784393372572`: 10/1,011 managed segments and `66399577`
  bytes, complete five-bot shutdown/startup evidence, and zero hard, monitor,
  text-log, repository, or target failures. The checkout stayed tracked-clean,
  all pane parents remained stable, and protected `misc:0.0` stayed `%8`, PID
  `434835`. No direct exchange request or event was manufactured.

## Previous Canonical Deployment (PR #1307)

- PR #1307 merged as canonical
  `8aefdbc82339b756ff642e726ae0924d5ca8774d` after exact-current-head Hermes
  approval and green Python/Rust CI. It composes the exact local restart
  executor with one bounded restart-through-observation smoke collection.
- VPS5 prepared the exact merge cleanly, then stopped, exited, relaunched, and
  verified all five configured targets without force. The exact window
  `1784332307933..1784332994590` selected 6/1,012 retained segments and recovered
  five stopping/stopped/startup cohorts. Smoke correctly remained red on one
  real KuCoin positions-fetch `RequestTimeout`; a second later timeout meant
  recovery was not yet proven, so every bot was left running.
- Repository and target gates remained green, pane parents and `misc:0.0` stayed
  unchanged, and no direct exchange request or event was manufactured. The raw
  exception string retained by `cycle.degraded` exposed the active bounded
  redaction follow-up; remote-host control and force escalation remain separate.

## Previous Canonical Deployment (PR #1306)

- PR #1306 merged as canonical
  `0d1b06b82f3bab011e29a350b4a5276c2ebd5356` after exact-current-head Hermes
  approval and green Python/Rust CI. It binds canonical `origin/master`
  fast-forward and Rust runtime preparation to exact caller-confirmed current
  and target commits plus a source fingerprint before restart execution.
- VPS5 first fast-forwarded to make the tool available without a bot restart or
  signal. A same-head execution returned green with no repository move or build;
  a valid wrong target failed with `fetched_target_head_mismatch` before build.
  Tracked state, all five configured pane parents, and `misc:0.0` stayed
  unchanged.
- The active follow-up composes the existing exact-pane graceful executor and
  bounded retained-window collector over one restart-through-observation window.
  Remote-host control and force escalation remain separate.

## Previous Canonical Deployment (PR #1305)

- PR #1305 merged as canonical
  `300fdd703fee9e1ce0e9c54df43bb7b1dcb858d8` after exact-current-head Hermes
  approval and green Python/Rust CI. It bounds exact historical restart-smoke
  collection by managed rotation intervals with predecessor, per-bot,
  global-file, and projected uncompressed-byte proof before event scanning.
- VPS5 fast-forwarded without a restart or bot signal. The retained PR #1296
  window selected 10/1,008 segments and `131834602` projected bytes, recovered
  all five shutdown/startup cohorts, and returned `ok=true` with zero hard
  failures. A malformed expected head rejected before producers with exit 2.
- All five pane parents and `misc:0.0` retained their exact IDs/PIDs and the
  checkout stayed tracked-clean. PR #1306 subsequently bound canonical
  fetch/fast-forward and Rust runtime preparation before restart execution;
  force escalation remains separate.

## Previous Canonical Deployment (PR #1303)

- PR #1303 merged as canonical
  `9e8d1343e0f1f43fc3207d611a8b06d88af8b6c0` after exact-current-head Hermes
  approval and green Python/Rust CI. It composes stable target sampling, exact
  smoke-window collection, and sanitized evidence evaluation in memory without
  intermediate reports or process control.
- VPS5 fast-forwarded without a bot restart or signal. The first real collector
  run discovered 1,012 retained event segments totaling about 801 MB and stayed
  CPU-active beyond ten minutes. Only the exact collector PID was interrupted;
  all five bot panes, `misc:0.0`, and tracked checkout state remained unchanged.
- A two-recent-segment cap returned in 20.9 seconds but correctly failed closed
  because it omitted the restart lifecycle. A read-only interval-selection
  prototype then selected 10 overlapping segments and returned the complete
  five-bot green lifecycle verdict in 37.3 seconds. The active follow-up makes
  that selection fail closed under explicit per-bot, global-file, and byte caps.

## Previous Canonical Deployment (PR #1302)

- PR #1302 merged as canonical
  `0b5503b2a9ee4817618b7aca25dab417af4292dd` after exact-current-head Hermes
  approval and green Python/Rust CI. It preserves and compares exact bounded
  epoch-ms smoke windows and fails closed on dropped hard-looking log evidence.
- VPS5 fast-forwarded without a restart or signal. The retained PR #1296 window
  evaluated green with exact bounds `1784316350000..1784317500000`; an in-memory
  one-millisecond mismatch and dropped-hard count each evaluated red with
  `log_scan_invalid`. All five pane parents and bot PIDs plus `misc:0.0` remained
  unchanged, and tracked state stayed clean.
- PR #1303 directly composed the existing target, smoke, and evidence builders
  in memory. Pull/build, SSH, process control, exchange access, and force
  escalation remain separate.

## Previous Canonical Deployment (PR #1301)

- PR #1301 merged as canonical
  `46a28795dec40acbee0dbaa3602be955bbecf23e` after exact-current-head Hermes
  approval and green Python/Rust CI. It adds a pure, bounded evaluator for
  already-generated full restart target and smoke JSON reports.
- VPS5 fast-forwarded without a restart or signal. The real PR #1296
  shutdown-through-startup window evaluated green with all five lifecycle and
  startup cohorts, zero hard failures, stable exact targets, and a clean exact
  repository head. A wider three-hour window correctly evaluated red on 30
  existing hard events. Bot and pane PIDs plus `misc:0.0` remained unchanged.
- The deploy exposed a fidelity defect: generic count projection clamped both
  modern epoch-ms bounds to `1000000000` and the log-bound comparison reused
  those lossy values. The active follow-up preserves exact bounded timestamps
  and fails closed on dropped hard-looking log evidence.

## Previous Canonical Deployment (PR #1300)

- PR #1300 merged as canonical
  `e1a4837914c1e4768cd7963bba47212499d32937` after exact-current-head Hermes
  approval and green Python/Rust CI. It requires an operator-confirmed Rust
  build-input fingerprint and rehashes the inputs after loaded-extension
  verification at every runtime boundary, detecting ignored-input drift inside
  the check.
- VPS5 fast-forwarded cleanly from `491f3192` without restarting or signalling
  bots. A deliberately wrong expected fingerprint failed before target sampling
  with `action_started=false`. The final 3/3 exact-target report retained the
  same five PIDs, zero extras or issues, and states `R=3,S=2`; tracked state and
  unrelated `misc:0.0` remained unchanged.
- The following slice made post-restart target and smoke evidence
  machine-evaluable from already generated full JSON reports. Automatic report
  collection, pull/build orchestration, and force escalation remain separate.

## Previous Canonical Deployment (PRs #1296 And #1295)

- Docs-only PR #1295 and behavior-changing PR #1296 merged as canonical
  `491f319251076b82799a5212efe2d797c56b1b31` after exact-current-head Hermes
  approval and green Python/Rust CI. PR #1296 hardens trailing-fill association
  across position snapshots and requires post-snapshot fill confirmation when
  authoritative position-update timestamps are unavailable.
- VPS5 fast-forwarded cleanly from `4a7a6753`. The exact local executor stopped
  only old bot PIDs `1015403/1015406/1015410/1015412/1015414` in panes
  `%358/%359/%360/%361/%362` and relaunched replacement PIDs
  `1019670/1019679/1019673/1019681/1019676` under the same pane parents and
  private supervisor fingerprint. Repository and Rust source/stamp checks
  remained exact and tracked-clean at every action boundary.
- Executor verification and the independent settled report were hard-green
  with 3/3 stable samples, five relaunch-ready targets, no extras or duplicates,
  and zero issues. Unrelated `misc:0.0` remained `%8`, PID `434835`. No direct
  exchange probe or event was manufactured.

## Previous Canonical Deployment (PR #1298)

- PR #1298 merged as `4a7a6753bff00f9b8749d9707f9bdccc4b3a5ffc`
  after exact-head Hermes approval and green Python/Rust CI. It requires the
  local restart executor to prove an exact Git head, zero tracked changes, and
  a source-matched loaded Rust extension before target sampling and again at
  both process-action boundaries.
- VPS5 fast-forwarded cleanly from `7b833471` without a restart or signal. A
  deliberately wrong expected head returned exit 1 with
  `action_started=false`, no target preflight, zero tracked changes, and a Rust
  source/stamp match at `7869bc3d...`. The final 3/3 exact-target report was
  hard-green with all five unchanged bot PIDs relaunch-ready, states `R=3,S=2`,
  zero issues, and the unchanged private supervisor fingerprint. Unrelated
  `misc:0.0` remained `%8`, PID `434835`.
- The tracked Rust Git tree matched the clean local tree, but its local
  fingerprint was `d14a5363...` because ignored `Cargo.lock` is intentionally
  included and differs by host. The active follow-up requires the operator to
  confirm the expected host-local Rust fingerprint rather than accepting any
  internally self-consistent source/stamp pair. No exchange request, process
  action, or event was manufactured.

## Previous Canonical Deployment (PR #1297)

- PR #1297 merged as `7b833471c1e770ceb8650a4c3b395713b6a76dcb`
  after exact-head Hermes approval and green Python/Rust CI. It added the first
  local exact-target graceful restart executor over the previously proven pane,
  process, relaunch, and private supervisor-fingerprint contracts.
- VPS5 fast-forwarded cleanly from `0f366b6f` without a restart or process
  signal. The executor CLI/help loaded, and the post-deploy 3/3 target report
  was hard-green with all five exact targets relaunch-ready, zero issues, and
  the unchanged private command fingerprint. Bot PIDs
  `1015403/1015406/1015410/1015412/1015414`, all pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged.
- The active follow-up binds restart authorization to the intended clean Git
  head and an exactly source-matched Rust extension before any signal and again
  before relaunch. No exchange request, process action, or event was
  manufactured during the PR #1297 deployment.

## Previous Canonical Deployment (PRs #1294 And #1293)

- PR #1294 merged as `3de024c76d5c07bda2b4e64400c1a204d6be38a8`
  and moved the restart supervisor fingerprint to complete private commands
  before redaction. VPS5 produced 3/3 stable hard-green target samples for all
  five configured bots with fingerprint
  `01d200d4a38c5c85a2123b5210224a18cdef08d0a8be3efc48edb4a159fc5db4`
  and no command exposure.
- PR #1293 merged as `0f366b6f6e6b05ab0bd748b012c2c86ed85f978c`
  and repaired live trailing-extrema position/fill/candle confirmation. VPS5
  fast-forwarded cleanly and one exact-pane Ctrl-C round stopped old bot PIDs
  `1013205/1013207/1013209/1013210/1013211`. Replacement PIDs
  `1015403/1015406/1015410/1015412/1015414` retained exact panes
  `%358/%359/%360/%361/%362`, pane parents
  `856294/856332/856364/856398/856434`, and the same fingerprint. Unrelated
  `misc:0.0` remained `%8`, PID `434835`.
- The immediate target report and smoke were hard-green with all five exact
  processes and no hard failures. A later bounded window correctly retained two
  real KuCoin account-state timeouts and one hard degraded cycle while all
  sampled `D` states recovered without PID churn. The fresh recovery smoke was
  hard-green with `326/326` remote calls, `61/61` account-critical calls, nine
  successful fill refreshes, and zero hard failures. A quiet exact-PID sample
  reached `R=4,S=1`; the final target report retained 3/3 stable samples, all
  five relaunch paths, zero issues, and the same fingerprint. No direct
  exchange probe or event was manufactured.
- The local exact-target executor followed in PR #1297. It excludes SSH, git
  pull/build, direct exchange access, automatic force escalation, and integrated
  post-restart smoke orchestration.

## Previous Canonical Deployment (PR #1277)

- PR #1277 merged to `master` as
  `deee460a18f1f532a4c3a4c6e89a4befd5469d2a`. It projects existing
  `rust_orchestrator.returned` and `action.planned` events into bounded
  correlated latest-per-bot planning-output health without copying raw orders,
  quantities, prices, or hashes or changing verdicts/runtime behavior.
- VPS5 fast-forwarded cleanly with no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged; tracked state stayed clean.
- Immediate and settled two-minute smokes were `ok=true` with zero
  hard/log/monitor/process failures, `189/189` and `191/191` remote calls
  successful, and five exact/config-valid live processes. Final exact states
  were `R=4,S=1`.
- The focused five-minute report naturally projected 60 planning-output events
  as 30 correlated pairs across all five bots with zero latest count
  mismatches. The same window contained 87 packet-update events, motivating the
  next report-only packet-health slice. No event or trading activity was
  manufactured.

## Previous Canonical Deployment (PR #1276)

- PR #1276 merged to `master` as
  `6378d40a55116f94fe65f1b24f4981101d838e74`. It projects existing
  `entry.initial_eligibility` events into bounded latest-per-bot
  staged-readiness health without copying raw per-symbol records or changing
  verdicts or runtime behavior.
- VPS5 fast-forwarded cleanly with no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged; tracked state stayed clean.
- The settled bounded two-minute smoke was `ok=true` with zero
  hard/log/monitor/process failures, `233/233` remote and `63/63`
  account-critical calls successful, eight successful fill refreshes, and five
  exact/config-valid live processes. Transient exact-process `D` samples
  cleared immediately to final states `R=3,S=2`.
- The focused report naturally projected 18 initial-entry observations across
  all five bots and 152 latest records: 12 blocked candidates and 140
  no-candidate outcomes, with producer truncation on three bots. A bounded
  five-minute inventory also found 36 correlated Rust-return/action-planned
  pairs, motivating the next report-only aggregate slice. No event or trading
  activity was manufactured.

## Previous Canonical Deployment (PR #1275)

- PR #1275 merged to `master` as
  `e5603244565807167494ffc53898f98f736f876a`. It projects existing
  `planning.symbol_state` events into bounded latest-per-bot staged-readiness
  health without changing verdicts or runtime behavior.
- VPS5 fast-forwarded cleanly with no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged; tracked state stayed clean.
- The bounded two-minute smoke was `ok=true` with zero hard/log/monitor/process
  failures, `291/291` remote and `58/58` account-critical calls successful,
  nine successful fill refreshes, and five exact/config-valid live processes.
  Transient `D` samples cleared to a final exact state of `R=4,S=1`.
- The focused report naturally projected 13 symbol-state events across four
  bots with bounded aggregate and redacted symbol evidence. The same settled
  inventory contained 12 natural `entry.initial_eligibility` events, motivating
  the next report-only aggregate slice. No event or trading activity was
  manufactured.

## Previous Canonical Deployment (PR #1274)

- PR #1274 merged to `master` as
  `bcbfa12808a00e39c1e4eb78e43e13746be553fa`. It scopes monitor event-type
  inventory and sampled cycle IDs to the requested smoke-report time window
  while preserving full-file validation and scanned-record counters.
- VPS5 fast-forwarded cleanly with no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged; tracked state stayed clean.
- A bounded 30-minute report observed one natural KuCoin eligibility event and
  counted that event type exactly once. A fresh two-minute focused report had
  no eligibility event and omitted the event type from inventory, directly
  validating the fixed boundary. The fresh smoke was `ok=true` with zero
  hard/log/monitor/process failures, `248/248` remote and `57/57`
  account-critical calls successful, eight successful fill refreshes, and all
  five exact/config-valid processes in state `R` with no uninterruptible sleep.
- The same fresh event inventory naturally contained 18
  `planning.symbol_state` events. The next report-only slice adds bounded
  latest-per-bot symbol-state evidence to staged-readiness health; no event or
  trading activity was manufactured.

## Previous Canonical Deployment (PR #1273)

- PR #1273 merged to `master` as
  `20238b50792da0a69b6fa2b13272c75ea4a0eade`. It projects existing bounded
  `forager.eligibility_changed` evidence into full, summary, brief, and
  section-selective smoke reports without changing verdicts or runtime
  behavior.
- VPS5 fast-forwarded cleanly with no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged; tracked state stayed clean.
- A current-plus-rotated 180-minute query found six natural eligibility events
  across all five bots, including two on KuCoin. The focused report retained
  bounded redacted symbol samples. The final two-minute smoke was `ok=true`
  with zero hard/log/monitor/process failures, `210/210` remote and `55/55`
  account-critical calls successful, nine successful fill refreshes, and all
  five matching/config-valid processes. One report-time OKX `D` sample cleared
  after ten seconds to final states `R=4,S=1`.
- The deploy also proved that monitor event-type inventory was scan-wide even
  with a requested time window. The next read-only report slice scopes that
  inventory and sampled cycle IDs to `since_ms`/`until_ms`; no event or trading
  activity was manufactured.

## Previous Canonical Deployment (PR #1272)

- PR #1272 merged to `master` as
  `af69725ed04b9a9a0402634455fd6bb05d71d7f5`. It adds existing bounded
  `planning.defer_summary` evidence to staged-readiness smoke reports and keeps
  registered optional selectors valid when their event family is absent,
  without changing verdicts or runtime behavior.
- VPS5 fast-forwarded cleanly with no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged; tracked state stayed clean.
- The final two-minute smoke was `ok=true` with zero hard/log/monitor/process
  failures, `279/279` remote and `44/44` account-critical calls successful,
  seven successful fill refreshes, and all five matching/config-valid
  processes in states `R=3,S=2` with no uninterruptible sleep. Twelve non-hard
  EMA readiness events remained durable.
- The formerly failing absent `forager_features` selector returned a valid
  base-only report, and the staged-readiness selector returned a valid
  zero-event section. No planning-defer event occurred naturally. The
  monitor-wide inventory listed `forager.eligibility_changed`, but later
  window-scoped query evidence showed that event was not in the settled
  two-minute window. PR #1273 subsequently exposed and validated six natural
  rotated eligibility events without manufacturing activity.

## Previous Canonical Deployment (PR #1271)

- PR #1271 merged to `master` as
  `e3640fa7338db2b64942a0773464f6499982dd8a`. It projects existing bounded
  `forager.feature_unavailable` events into full, summary, brief, and selected
  smoke reports without changing verdicts or runtime behavior.
- VPS5 fast-forwarded cleanly with no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged; tracked state stayed clean.
- The settled two-minute smoke was `ok=true` with zero hard/log/monitor/
  process/pipeline failures, `186/186` remote and `55/55` account-critical
  calls successful, nine successful fill refreshes, and all five matching/
  config-valid processes in states `R=4,S=1` with no uninterruptible sleep.
  Fourteen non-hard EMA readiness events remained durable.
- No forager-feature-unavailable event occurred naturally. The bounded section
  probe instead exposed that a registered optional selector was rejected when
  its zero-event section was omitted. The next report-only slice repairs that
  path and consumes existing `planning.defer_summary` evidence; nothing was
  manufactured.

## Previous Canonical Deployment (PR #1270)

- PR #1270 merged to `master` as
  `9869263d74829f163b0bc8010fa18d1bf41de055`. It makes the read-only live
  performance report retain latest-lifecycle configured startup-budget
  assessments and aggregate their status without changing startup or trading.
- VPS5 fast-forwarded cleanly with no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged.
- A bounded performance report was `ok=true` with zero errors or warnings and
  retained the legacy no-budget shape for naturally unconfigured events. The
  final two-minute smoke was `ok=true` with zero hard/log/monitor/process/
  pipeline failures, `147/147` remote and `37/37` account-critical calls
  successful, nine successful fill refreshes, and all five matching/config-
  valid processes in states `R=4,S=1` with no uninterruptible sleep.
- Fifteen non-hard attention events remained durable: eleven EMA readiness,
  two staged-cycle degradation, and two websocket reconnect events. The next
  report-only slice exposes existing structured forager feature-unavailability
  evidence in live smoke reports without changing verdicts or behavior.

## Previous Canonical Deployment (PR #1269)

## Previous Canonical Deployment (PR #1268)

- PR #1268 merged to `master` as
  `f8ec74792d69e229fd63bf4cdf7ab7a092a79cd4`. It adds an aggregate-only
  projection to the bounded read-only log secret inventory while preserving
  the full per-file report as the default.
- VPS5 fast-forwarded cleanly with no restart. The summary scanned 40 of 1,182
  discovered files and 8,153,519 bytes, reported ten positive and 25 truncated
  files, skipped six symlinks, and had zero unreadable files or discovery
  errors. It retained aggregate counts of 144 secret-query and 143 private
  websocket-query matches without per-file paths, ages, or hashes.
- No source values or lines were emitted, no artifacts were remediated, and no
  process signal was sent. All five configured bot panes and unrelated
  `misc:0.0` remained present.
- PR #1269 subsequently added optional durable diagnostic budgets to existing
  startup timing events and gave those configured targets report precedence
  without enforcing them.

## Previous Canonical Deployment (PR #1267)

- PR #1267 merged to `master` as
  `951e42d07303d5c78ca31d4df9ee5f21e5b931cb`. It extends the value-free
  inventory's existing query classification to scheme-less request paths and
  fragments without changing report retention or runtime behavior.
- VPS5 fast-forwarded cleanly with no restart. The bounded 40-file,
  250,000-byte scan found 144 secret-query matches versus 143 private
  websocket-query matches, proving one additional non-websocket query fragment
  was classified. It discovered 1,182 files, skipped six symlinks, and had zero
  discovery or read errors.
- No source values or lines were emitted, no artifacts were remediated, and no
  process signal was sent. All five configured bot panes and unrelated
  `misc:0.0` remained present.
- Even compact JSON was roughly 6,000 tokens because it retained every
  per-file row; PR #1268 added an aggregate-only summary while preserving the
  full report as the default.

## Previous Canonical Deployment (PR #1266)

- PR #1266 merged to `master` as
  `4015a3b3f145d58db3e1a873ec4cc5ce465368f7`. It adds existing startup
  elapsed/phase budget status coverage to the brief smoke projection without
  changing events, baselines, thresholds, startup, or trading behavior.
- VPS5 fast-forwarded cleanly with no restart. A bounded four-hour
  current-segment-only report scanned six monitor files and 19,163 records
  with zero monitor parse errors, and emitted the new coverage fields. Startup
  timing rows had rotated out, so their counts were naturally zero.
- The section-only report retained two hard and 319 attention problem events
  and was not treated as a green deploy-health verdict. Exact bot PIDs and
  unrelated `misc:0.0` PID `434835` remained unchanged. A transient GateIO
  `D` sample cleared to `R` after 20 seconds.
- Static follow-up confirmed the inventory's query classifier still required
  an HTTP/websocket scheme; the next read-only slice recognizes scheme-less
  request paths and query fragments without retaining values.

## Previous Canonical Deployment (PR #1265)

- PR #1265 merged to `master` as
  `fc4a134fb7b856ea426afb750559ff7ae94c7a07`. It adds a bounded, value-free,
  read-only historical log inventory; it does not mutate artifacts, contact an
  exchange, or change bot behavior.
- VPS5 fast-forwarded cleanly with no restart. A bounded dry run scanned 40 of
  1,182 discovered files at most 250,000 decompressed bytes each, skipped six
  symlinks, and had zero discovery or read errors. Ten retained May logs
  contained private websocket/query credential classes; no matched values or
  source lines were emitted, and no remediation was attempted.
- Exact bot PIDs `979190/979193/979196/979199/979202`, pane parents, and
  unrelated `misc:0.0` PID `434835` remained unchanged. The tracked checkout
  remained clean with expected untracked artifacts preserved.
- The next report-only slice makes brief startup timing output disclose
  existing budget-assessment status coverage, so zero over-budget phases
  cannot be mistaken for complete assessment when baselines or latest timings
  are unavailable.

## Previous Canonical Deployment (PR #1264)

## Previous Canonical Deployment (PR #1263)

- PR #1263 merged to `master` as
  `e6d2461a54c9e446a3aee4ff367653aedafa8494`. It consolidates startup
  lifecycle console ownership while retaining structured/monitor durability;
  task ordering, waits, exchange calls, and trading behavior are unchanged.
- VPS5 sent one SIGINT at `2026-07-16T07:54:15Z` to exact old PIDs
  `975507/975510/975511/975513/975515`; all exited naturally by `07:54:37Z`.
  New PIDs are `977722/977725/977728/977731/977734`; pane parents and unrelated
  `misc:0.0` PID `434835` were unchanged.
- The settled two-minute smoke was `ok=true` with zero hard, log, monitor,
  process, or event-pipeline failures, `251/252` remote and all `37/37`
  account-critical calls successful, seven successful fill refreshes, five
  matching/config-valid processes in states `R=4,S=1`, and a clean tracked
  checkout. One non-hard KuCoin candle timeout was retained.
- Every bot naturally emitted exactly one `[bot] started`, one retained
  `phase=startup-ready`, zero removed lifecycle INFO lines, and a durable
  `bot.ready` event. Hyperliquid also completed full warmup without the demoted
  success/jitter detail. The same exact startup segments retained four empty
  maintainer-stop summaries and five hourly scheduler-jitter INFO lines; the
  next slice demotes only that routine detail.

## Previous Canonical Deployment (PR #1262)

## Previous Canonical Deployment (PR #1261)

## Previous Canonical Deployment (PR #1260)

## Previous Canonical Deployment (PR #1259)

## Previous Canonical Deployment (PR #1258)

## Previous Canonical Deployment (PR #1257)

## Previous Canonical Deployment (PR #1256)

## Previous Canonical Deployment (PR #1255)

## Previous Canonical Deployment (PR #1254)

## Previous Canonical Deployment (PR #1253)

## Previous Canonical Deployment (PR #1252)

## Previous Canonical Deployment (PR #1251)

## Previous Canonical Deployment (PR #1250)

## Previous Canonical Deployment (PR #1249)

## Previous Canonical Deployment (PR #1248)

- PR #1248 merged to `master` as `75bf6cd67f61b902fb5ca4399939c2b2e5945088`
  after exact-head Hermes approval and green Python/Rust CI while Grok was
  temporarily halted. It bounds only the `live.approved_coins` config-change
  projection; config application, schema, non-target config logs, and trading
  behavior are unchanged.
- VPS5 fast-forwarded cleanly and gracefully restarted the five exact bot
  panes. Old PIDs `955200/955304/955306/955366/955422` exited naturally within
  ten seconds after one SIGINT round; pane PIDs and unrelated `misc:0.0` PID
  `434835` were preserved. New bot PIDs are
  `957519/957674/957669/957739/957675` for
  Binance/KuCoin/GateIO/OKX/Hyperliquid respectively.
- Natural KuCoin startup output reduced the exact approved-coin override from
  678 to 154 visible characters while retaining the old collection count/
  sample and both 19-coin side counts/samples. Real pre- and post-restart
  KuCoin timeouts recovered without intervention. The final fresh two-minute
  smoke was `ok=true` with zero hard/log/monitor/process failures, `56/56`
  account-critical calls successful, nine successful fill refreshes, all five
  expected config-valid processes in exact `R/S` states, and a clean tracked
  repository. One non-account-critical candle timeout remained visible as
  non-hard evidence.
- The same restart exposed six simultaneous `fetch_lock_hold_timeout` warnings
  at 346-349 characters each. They repeated deterministic path and owner-scope
  fields already present in the warning. The next slice compacts only that
  projection while preserving one warning per affected symbol and all lock
  behavior.

## Previous Canonical Deployment (PR #1247)

- PR #1247 merged to `master` as `e3c6332c6b1f4ff692f94867f9dbfaf182cc0684`
  after exact-head Hermes approval and green Python/Rust CI while Grok was
  temporarily halted. The change compacts only the human projection of visible
  `trailing.status` records; structured payloads, materiality admission,
  cadence, risk calculations, and trading behavior are unchanged.
- VPS5 fast-forwarded cleanly and gracefully restarted the five exact bot
  panes. Old PIDs `953285/953287/953289/953291/953292` exited naturally after
  one SIGINT round; pane PIDs and unrelated `misc:0.0` PID `434835` were
  preserved. New bot PIDs are `955304/955366/955200/955422/955306` for
  Binance/KuCoin/GateIO/OKX/Hyperliquid respectively.
- The immediate five-minute smoke was hard-green with `338/338` remote and
  `93/93` account-critical calls successful, twenty successful fill refreshes,
  five expected processes matched, no hard/log/monitor/pipeline failures, and
  a clean tracked repository. The settled two-minute smoke remained
  `ok=true`, with zero hard failures and `55/55` account-critical calls
  successful; one non-account-critical KuCoin candle timeout remained visible
  as non-hard evidence. Two report-time `D` samples cleared to exact `R/S`
  states on the quiet follow-up.
- Natural Hyperliquid output validated the compact formatter at 211 visible
  characters versus the prior 311, retaining cycle, close/armed state, mode,
  threshold and retracement gates/values, current price, symbol, and side.
  The same restart exposed the next boundedness gap: a CLI-approved-coin
  override printed a 678-character full collection diff. The next slice limits
  that startup projection without changing the applied config.

## Historical Status Snapshot (PR #1191)

Snapshot captured: 2026-07-11.

This block is retained as historical context and is not the operational current
status. Use `docs/plans/live_logging_overhaul_current_status.md` for the active
PR, deployed head, review gate, VPS state, and next action.

Current `origin/v8` head:

Current logging-overhaul head:

Current work:

- Active slice: add centralized readiness scope and trading-impact metadata to
  five existing `bot.startup_timing` phases, then expose per-bot and
  aggregate readiness SLA timing in performance and smoke reports without
  changing startup or trading behavior. Keep the best-effort `active-candle`
  phase timing-only because it cannot prove readiness after a tolerated warmup
  failure.

Current review gate:

- Composer has been stopped/retired from this loop. While Claude Opus 4.8 is
  rate-limited, the required review gate is Hermes + Grok 4.5 + CI; Claude may
  rejoin when available. Findings from any additional reviewer still require
  verification and resolution. A degraded gate after reviewer absence must be
  explicitly authorized and called out in the progress evidence.

Retuned goal boundary:

VPS5 deployment status:

## Phase Checklist

## Historical Work Snapshot (Superseded)

This block records the in-progress handoff from 2026-06-30 and is not the
current automation target. Use `live_logging_overhaul_current_status.md` for the
active branch, PR, review gate, and rollout instructions.

## Merged Slices

### PR #976: Trailing Status Risk Activity Report

- Branch: `codex/v8-live-performance-trailing-risk-activity`.
- Scope: read-only live performance report projection and tests.
- Result: `live-performance-report` includes existing `trailing.status` events
  in the bounded `risk_activity` section, so trailing/waiting position state can
  be found beside HSL and unstuck risk-state events without opening the full
  event stream. The projection uses event-envelope labels and bounded symbol
  samples only; threshold/retracement prices and detailed event payload values
  remain out of the shareable report.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Local validation covered the full live-performance-report test file,
  py_compile, `git diff --check`, and an added-line silent-handling scan.
- VPS5 evidence: deployed to `v8` at `1823d3e1` with all five configured bots
  restarted and left running. Repeated smoke checks stayed hard-green with no
  failed remote calls or account-critical remote calls. The final performance
  report showed `risk_activity` populated with deployed `trailing.status`
  events.

### PR #924: Startup Phase Readiness Summary

- Branch: `codex/v8-startup-readiness-summary`.
- Scope: read-only live performance report tooling and tests.
- Result: `passivbot tool live-performance-report` now adds bounded aggregate
  startup phase timing counters to `startup_readiness`, so operators can see
  cross-bot startup phase elapsed and since-previous summaries without opening
  every per-bot phase row. Startup phase labels are whitelisted before they
  appear in either `startup_readiness` or `operation_durations`.
- Review evidence: Cursor, Hermes, Claude, and CI approved the final head
  after the Hermes finding about raw startup phase labels in
  `operation_durations` was fixed. Local validation covered focused
  live-performance-report tests plus py_compile and `git diff --check`. No
  event producers, exchange calls, cache mutation, readiness gates, console
  routing, monitor writes, order/risk logic, or trading behavior changed.
- VPS5 evidence: deployed to `v8` at `652b019e` without bot restart because
  the change is read-only tooling. Smoke stayed hard-green with all five
  configured bots running, clean tracked repository state, no failed remote
  calls, and no failed account-critical remote calls. Current monitor segments
  had no startup timing rows, so the new `startup_readiness` aggregate fields
  were present but empty.

### PR #923: Market Snapshot Staleness Summary

- Branch: `codex/v8-market-snapshot-staleness-summary`.
- Scope: read-only live performance report tooling and tests.
- Result: `passivbot tool live-performance-report` now adds aggregate market
  snapshot staleness counters to `input_staleness`, so operators can see
  snapshot observations, missing symbol totals, configured-age excess, bounded
  source labels, and age/count summaries without opening each `snapshot.built`
  event.
- Review evidence: Cursor, Hermes, Claude, and CI approved the final head.
  Local validation covered focused live-performance-report tests plus
  py_compile and `git diff --check`. No event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, order/risk logic,
  or trading behavior changed.
- VPS5 evidence: deployed to `v8` at `31ef4d40` without bot restart because
  the change is read-only tooling. Smoke stayed hard-green with all five
  configured bots running, clean tracked repository state, no failed remote
  calls, no failed account-critical remote calls, and
  `live-performance-report --summary` showed
  `input_staleness.market_snapshot` from live monitor data.

### PR #921: Incident Bundle Problem Report Discovery Summary

- Branch: `codex/v8-incident-problem-discovery-summary`.
- Scope: read-only incident bundle tooling and tests.
- Result: `passivbot tool live-incident-bundle` now projects
  `problem_event_report.file_discovery` and
  `problem_event_report.event_window` into the compact incident-bundle result
  so scoped problem-event queries can be verified without opening the archive.
- Review evidence: Cursor, Hermes, Claude, and CI approved the final head.
  Local validation covered focused incident-bundle and event-query tests plus
  py_compile and `git diff --check`. No event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, order/risk logic,
  or trading behavior changed.
- VPS5 evidence: deployed to `v8` at `5808f679` without bot restart because
  the change is read-only tooling. Smoke stayed hard-green with all five
  configured bots running, clean tracked repository state, and focused OKX
  incident-bundle output showing `problem_event_report.files_scanned=1` plus
  path-pruned discovery/window metadata.

### PR #920: Incident Bundle Time-Window Discovery Summary

- Branch: `codex/v8-incident-window-discovery-summary`.
- Scope: read-only incident bundle tooling and tests.
- Result: `passivbot tool live-incident-bundle` now projects
  `time_window.files_scanned` and
  `time_window.file_discovery` into the compact incident-bundle result so
  focused bundle scoping can be verified without opening the archive.
- Review evidence: Cursor, Hermes, Claude, and CI approved the final head.
  Local validation covered focused incident-bundle and event-query tests plus
  py_compile and `git diff --check`. No event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, order/risk logic,
  or trading behavior changed.
- VPS5 evidence: deployed to `v8` at `570875ab` without bot restart because
  the change is read-only tooling. Smoke stayed hard-green with all five
  configured bots running, clean tracked repository state, and focused OKX
  incident-bundle output showing `time_window.files_scanned=1` plus path-pruned
  discovery metadata.

### PR #919: Incident Bundle Time-Window Scope Filters

- Branch: `codex/v8-incident-window-scopes`.
- Scope: read-only incident bundle tooling and tests.
- Result: `passivbot tool live-incident-bundle` applies query scope
  filters to `time_window_report.json`, `timeline.txt`, and matched event
  segment selection, while preserving the existing behavior where a cycle-id
  bundle keeps surrounding time-window context instead of filtering the window
  down to that cycle only.
- Local validation: focused incident-bundle and event-query tests plus
  py_compile passed before opening review. No event producers, exchange calls,
  cache mutation, readiness gates, console routing, monitor writes, order/risk
  logic, or trading behavior changed.
- Review evidence: Hermes, Claude, Cursor, and CI approved the branch.
- VPS5 evidence: deployed at `989b81c9` without bot restart because this is
  read-only tooling. Smoke stayed hard-green with all five bots matched, and a
  focused OKX bundle verified scoped time-window paths, exchange/user values,
  timeline, and event-segment selection from the live monitor tree.

### PR #918: Incident Bundle Query Scope Filters

- Branch: `codex/v8-incident-query-scopes`.
- Scope: read-only incident bundle tooling and tests.
- Result: `passivbot tool live-incident-bundle` now exposes additional
  event-query scope filters for the bundled event reports:
  `--level`, `--exchange`, `--user`, `--bot-id`,
  `--remote-call-group-id`, `--side`, `--source`, `--component`, `--tag`, and
  `--data-eq`. The filters are passed to both `event_report.json` and
  `problem_event_report.json`, recorded in `manifest.json`, and preserve the
  existing smoke/process/log behavior.
- Local validation: focused incident-bundle and event-query tests plus
  py_compile passed before opening review. No event producers, exchange calls,
  cache mutation, readiness gates, console routing, monitor writes, order/risk
  logic, or trading behavior changed.
- Review evidence: Hermes, Claude, Cursor, and CI approved the branch.
- VPS5 evidence: deployed at `946d0757` without bot restart because this is
  read-only incident-bundle tooling. Smoke stayed hard-green with all five bots
  matched, and a focused OKX bundle verified scoped event/problem reports from
  the live monitor tree. That smoke also showed the remaining gap addressed by
  the next slice: the time-window report still scanned broader root context.

### PR #917: Incident Bundle Problem Event Report

- Branch: `codex/v8-incident-problem-query`.
- Scope: read-only incident bundle tooling and tests.
- Result: `passivbot tool live-incident-bundle` now embeds
  `problem_event_report.json` by default. The report is built with the shared
  `live-event-query --problem-events` predicate, honors the bundle's existing
  cycle/symbol/status/reason/time/tail filters, includes a trace summary, and
  can be disabled with `--no-problem-report` for compact bundles. Event segment
  selection now considers the problem-event report too, so bundles keep the raw
  segment needed to reconstruct smoke attention rows.
- Local validation: focused incident-bundle and event-query tests plus
  py_compile passed before opening review. No event producers, exchange calls,
  cache mutation, readiness gates, console routing, monitor writes, order/risk
  logic, or trading behavior changed.

### Foundation Before PR #619

### PR #619: Shutdown Progress

- Branch: `codex/v8-shutdown-progress`.
- Scope: adjacent operations improvement, not logging core.
- Result: improved shutdown progress and bounded shutdown cancel grace coverage.
- Follow-up: continue shutdown interruption work outside logging-only PRs.

### PR #621: Live Event Query Helper

- Branch: `codex/v8-live-event-query-helper`.
- Scope: shared live event query schema constants and initial query helper.
- Result: provided a stable base for later CLI filters and incident tooling.

### PR #622: Startup Warm Cache

- Branch: `codex/v8-startup-warm-cache`.
- Scope: adjacent operations improvement.
- Result: improved live startup warm-cache reuse.
- Follow-up: continue cache proof and warmup optimization separately from
  observability-only slices.

### PR #623: Live Event Query Scope

- Branch: `codex/v8-live-event-query-scope`.
- Scope: bounded live event query directory scans and rotated scan defaults.
- Result: query helper became safer on VPS-sized monitor trees.

### PR #624: EMA Console Noise

- Branch: `codex/v8-ema-console-noise`.
- Scope: console cleanup.
- Result: reduced forager EMA console noise while keeping diagnostics available
  through structured/debug paths.

### PR #625: Candle Tail Event

- Branch: `codex/v8-candle-tail-event`.
- Scope: candle/EMA readiness observability.
- Result: emitted structured candle tail projection events.

### PR #626: Event Query Filter

- Branch: `codex/v8-live-event-query-filter`.
- Scope: query tooling.
- Result: added event-type filtering to `passivbot tool live-event-query`.

### PR #627: Warmup Cache Decision Event

- Branch: `codex/v8-warmup-cache-event`.
- Scope: startup/warmup observability.
- Result: emitted structured warmup cache decision events.

### PR #628: Startup Timing Event

- Branch: `codex/v8-startup-timing-event`.
- Scope: startup timing observability.
- Result: emitted startup timing events.

### PR #629: Cache Load Events

- Branch: `codex/v8-cache-load-events`.
- Scope: cache instrumentation.
- Result: emitted candle cache load events and hardened payload building.

### PR #630: Cache Load Event Throttle

- Branch: `codex/v8-cache-load-event-throttle`.
- Scope: high-volume policy.
- Result: throttled cache load events to keep structured output bounded.

### PR #631: Cache Flush Events

- Branch: `codex/v8-cache-flush-events`.
- Scope: cache instrumentation.
- Result: emitted cache flush events.

### PR #633: Risk Mode Events

- Branch: `codex/v8-risk-mode-events`.
- Scope: risk mode observability.
- Result: emitted risk mode change events and covered halted HSL mode events.

### PR #634: Candle Coverage Events

- Branch: `codex/v8-candle-coverage-events`.
- Scope: candle coverage audit observability.
- Result: emitted candle coverage audit events.

### PR #635: Fill Refresh Events

- Branch: `codex/v8-fill-refresh-events`.
- Scope: fill refresh observability.
- Result: emitted fill refresh summary events and covered fill refresh resync
  summaries.

### PR #636: Rust Orchestrator Event Hardening

- Branch: `codex/v8-rust-orchestrator-event-hardening`.
- Scope: event emission safety.
- Result: hardened Rust orchestrator event emission and redacted orchestrator
  error events.

### PR #637: Live Ops Improvement Backlog

- Branch: `codex/v8-live-ops-improvement-backlog`.
- Scope: process tracking.
- Result: created the living operations improvement backlog and clarified live
  event query backlog work.

### PR #638: Live Event Query Filters

- Branch: `codex/v8-live-event-query-filters`.
- Scope: query tooling.
- Result: added additional live event query filters.

### PR #639: Live Smoke Report Tool

- Branch: `codex/v8-live-smoke-report-tool`.
- Scope: operator tooling.
- Result: added read-only live smoke report tooling for monitor/log inspection.

### PR #640: Health Summary Events

- Branch: `codex/v8-health-summary-events`.
- Scope: health observability.
- Result: emitted structured health summary events.

### PR #641: Live Incident Bundle

- Branch: `codex/v8-live-incident-bundle`.
- Scope: incident tooling.
- Result: added live incident bundle tool and redacted monitor snapshots.
- VPS5 evidence: bundle smoke created an archive successfully; tool returned
  attention because live GateIO HSL RED risk events were present, not because
  bundle generation failed.

### PR #642: Live Event Query ID Scopes

- Branch: `codex/v8-live-event-query-id-scopes`.
- Scope: query tooling.
- Result: added `bot_id`, `snapshot_id`, `plan_id`, `action_id`,
  `remote_call_group_id`, and related ID filters; timeline rendering now uses
  shared event ID keys.
- VPS5 evidence: deployed to VPS5 at `ad36d8ea`; `--remote-call-group-id`
  returned correlated Kucoin authoritative remote-call events.

### PR #643: Health Resource Pressure

- Branch: `codex/v8-health-resource-pressure`.
- Scope: health observability.
- Result: enriched structured `health.summary` events with resource pressure and
  live event pipeline counters.
- Review evidence: Cursor, Hermes, and Claude approved current head
  `d34241a4`; CI green; local targeted tests passed before merge.
- VPS5 evidence: pending pull/restart/smoke.

### PR #644: Logging And Ops Progress Tracking

- Branch: `codex/v8-live-logging-progress-tracker`.
- Scope: process tracking.
- Result: added this progress ledger and converted the live operations backlog
  into a living checklist with per-item statuses and a merged-work log.

### PR #645: Reason-Code Registry Slice

- Branch: `codex/v8-reason-code-registry-slice`.
- Scope: event taxonomy and drift prevention.
- Result: added shared `EventTags` and `ReasonCodes` registries for common live
  event tags/reason codes, migrated representative emitters without changing
  emitted strings, and documented the registry rule.

### PR #646: Console Event Summaries

- Branch: `codex/v8-console-event-summaries`.
- Scope: Phase 5 console/text projection.
- Result: improved `format_console_event()` with compact operator-facing tags
  and typed summaries for order waves, order writes, confirmation results, and
  Rust planning returns. Routes and console event volume were unchanged.

### PR #648: Live Event Trace Summaries

- Branch: `codex/v8-live-event-trace-summary`.
- Scope: operator query tooling.
- Result: added `passivbot tool live-event-query --trace-summary` to aggregate
  matched live events by event type, level, status, reason code, ID scopes,
  symbol/side, and order-wave/action coverage. Summary counts cover all matched
  events even when `--limit` truncates the returned event sample.

### PR #649: Startup Timing Baselines In Smoke Report

- Branch: `codex/v8-startup-phase-budgets`.
- Scope: adjacent operations observability.
- Result: `passivbot tool live-smoke-report` now summarizes existing
  `bot.startup_timing` monitor events into latest per-phase timings and rolling
  median/p95/min/max baselines. Latest details are redacted before smoke-report
  or incident-bundle output.

### PR #651: Live Event Order Trace View

- Branch: `codex/v8-live-event-order-trace`.
- Scope: operator query tooling.
- Result: added `passivbot tool live-event-query --order-trace` to reconstruct
  order-wave/action lifecycles from existing structured execution events. The
  view groups by `order_wave_id` and `action_id`, reports event/status/reason
  counts, confirmation events, symbol/pside/side sets, and bounded event
  samples with shortened order/client-order references.

### PR #652: Order Trace Progress Update

- Branch: `codex/v8-progress-after-order-trace`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #651 merged.

### PR #653: Live Event Registry Documentation

- Branch: `codex/v8-reason-code-registry-docs`.
- Scope: event taxonomy documentation and drift prevention.
- Result: added `docs/ai/live_event_registry.md` for stable event tags and
  reason codes, linked it from the AI docs router/logging guide, and added a
  doc drift test that compares documented values to `EventTags`/`ReasonCodes`.

### PR #654: Live Event Cycle Trace View

- Branch: `codex/v8-live-event-cycle-trace`.
- Scope: operator query tooling.
- Result: added `passivbot tool live-event-query --cycle-trace` to reconstruct
  matched events by `cycle_id`. Each cycle contains bounded timeline samples,
  aggregate trace summaries, and nested order traces using the existing order
  lifecycle reconstruction.

### PR #655: Cycle Trace Progress Update

- Branch: `codex/v8-progress-after-cycle-trace`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PRs #652-#654 merged.

### PR #656: Local Cache Integrity Doctor

- Branch: `codex/v8-cache-integrity-doctor-slice`.
- Scope: adjacent operations tooling.
- Result: added `passivbot tool cache-integrity-doctor`, a read-only local
  cache smoke report for root presence, aggregate file/size counts, empty
  files, and corrupt JSON/NDJSON/NPY artifacts. This is an initial cache-doctor
  slice; it does not yet prove warm-cache coverage or HSL/fill metadata
  compatibility.

### PR #658: Cache Doctor Progress Update

- Branch: `codex/v8-progress-after-cache-doctor`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #656 merged.

### PR #659: Incident Bundle Trace Reports

- Branch: `codex/v8-incident-bundle-traces`.
- Scope: incident tooling.
- Result: `passivbot tool live-incident-bundle` now embeds existing
  `live-event-query` trace-summary and order-trace reports in `event_report.json`
  by default, includes cycle-trace reconstruction when scoped to `--cycle-id`,
  and supports `--no-trace-report` for compact bundles.
- VPS5 evidence: deployed to VPS5 at `27931c81`; a read-only bundle smoke on
  monitor data produced a tarball containing `trace_summary`, `order_trace`, and
  `cycle_trace` sections. The tool returned attention because the embedded
  smoke report saw existing GateIO HSL RED and EMA readiness degradation.

### PR #661: Live Smoke Process Status

- Branch: `codex/v8-smoke-process-status`.
- Scope: operator tooling.
- Result: `passivbot tool live-smoke-report` can now include an optional
  read-only `processes` section. With `--supervisor-config`, tmuxp-style
  expected `passivbot live` commands are compared against running live
  processes and missing expected bots become smoke hard failures. Incident
  bundles pass the same process snapshot through `smoke_report.json` when
  requested.
- VPS5 evidence: deployed to VPS5 at `72b3d931`; read-only smoke using
  `/root/bots_vps5.yaml` matched all five expected bots and left them running.
  The overall smoke exit remained nonzero because Kucoin authoritative state
  fetches had recent `RequestTimeout` events, not because process liveness
  failed.

### PR #663: Remote-Call Failure Smoke Summary

- Branch: `codex/v8-smoke-remote-call-summary`.
- Scope: operator tooling.
- Result: `passivbot tool live-smoke-report` now includes a bounded
  `remote_call_failures` aggregate section built from existing
  `remote_call.failed` monitor events. Groups are keyed by
  bot/reason/surface/error type/component and include latest redacted failure
  context.
- VPS5 evidence: deployed to VPS5 at `45b0cf9e`; read-only smoke using
  `/root/bots_vps5.yaml` still matched all five expected bots and now exposed
  Kucoin authoritative endpoint timeouts directly in the smoke output:
  positions=9, open_orders=7, balance=7.

### PR #665: Live Smoke Report Time Window

- Branch: `codex/v8-smoke-report-time-window`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` can scope structured monitor
  events with `--since-ms`, `--until-ms`, or `--recent-minutes`, and reports
  explicit event-window counters. Incident bundles pass the same window into
  embedded smoke reports. Text-log scanning remains intentionally unchanged.

### PR #666: Create Filter/Defer Events

- Branch: `codex/v8-execution-defer-events`.
- Scope: execution/order lifecycle observability.
- Result: create-order pre-exchange filter/defer decisions now emit bounded
  structured events after the existing gates decide. The execution gates remain
  authoritative, event emission is best-effort, and the new routes stay off the
  default console/text projection.

### PR #667: Live Event Query Time Window

- Branch: `codex/v8-event-query-time-window`.
- Scope: operator query tooling.
- Result: `passivbot tool live-event-query` can scope matched structured events
  with `--since-ms`, `--until-ms`, or `--recent-minutes`. The same scoped event
  set feeds query output, timeline, trace summary, order trace, and cycle trace
  views, with explicit event-window counters.

### PR #668: Cache Doctor Family Summary

- Branch: `codex/v8-cache-doctor-family-summary`.
- Scope: adjacent operations tooling.
- Result: `passivbot tool cache-integrity-doctor` now includes per-root and
  aggregate cache-family summaries plus family tags on reported issues. This is
  still read-only diagnostics and does not decide whether live trading may reuse
  a warm cache.
- VPS5 evidence: deployed to VPS5 at `734c2de0`. A settled 2-minute smoke after
  restart reported all five configured bots running and no hard problem events.

### PR #670: Smoke Report Timestamped Log Window

- Branch: `codex/v8-smoke-report-log-window`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` applies `since_ms`/`until_ms` and
  `--recent-minutes` windows to parseable ISO-UTC text log lines as well as
  structured monitor events. Unparseable log lines remain visible and are
  counted in `logs.window.unparsed_ts`.
- VPS5 evidence: deployed to VPS5 at `b74d12be` without bot restart. Smoke
  showed `logs.window.lines_skipped_before=2000`, proving stale parseable log
  lines were excluded, while two current Kucoin websocket warning lines were
  still false-classified as hard due to the older traceback matcher.

### PR #671: Smoke Report Traceback Prose Filter

- Branch: `codex/v8-smoke-report-traceback-pattern`.
- Scope: operator smoke tooling.
- Result: smoke-report text-log matching now treats only real Python traceback
  headers (`Traceback (most recent call last):`) as traceback signals, avoiding
  hard/attention matches for operational prose such as "suppressing callback
  traceback".
- VPS5 evidence: deployed to VPS5 at `34f63799` without bot restart. A
  2-minute smoke with text logs enabled reported all five configured bots
  running, `logs.hard_matches=0`, and no hard problem events.

### PR #673: Live Smoke Risk Event Summary

- Branch: `codex/v8-live-smoke-risk-summary`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes a bounded
  `risk_events` aggregate built from existing structured HSL/risk events. It
  groups by bot, event type, symbol, pside, and reason, keeps the latest compact
  risk fields such as tier, mode, drawdown score, distance to red, and cooldown
  timing, and does not change smoke `ok`/`attention`/hard-failure policy.
- VPS5 evidence: deployed to VPS5 at `2697ff48` without bot restart. A
  5-minute smoke with text logs and `/root/bots_vps5.yaml` process matching
  reported all five configured bots running, no hard failures, no log hard
  matches, and exposed GateIO ZEC long HSL RED cooldown in `risk_events`.

### PR #675: Smoke Log Unparsed Policy

- Branch: `codex/v8-smoke-log-unparsed-policy`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` and embedded incident-bundle smoke
  reports now accept `--log-window-unparsed-policy keep|drop`. The default
  `keep` preserves prior behavior; opt-in `drop` suppresses only non-signal
  unparseable text-log lines when a log window is active. Signal-bearing
  unparseable lines, including Python traceback headers, remain visible and can
  still make smoke hard-fail.
- VPS5 evidence: deployed to VPS5 at `3aa1e7a7` without bot restart. A
  2-minute smoke with `--log-window-unparsed-policy drop` and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no log matches, no remote-call failures, and
  `logs.window.unparsed_policy=drop`.

### PR #676: Smoke Log Unparsed Policy Progress

- Branch: `codex/v8-progress-after-unparsed-policy`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #675 merged and VPS5 smoke confirmed the unparseable-log window policy.

### PR #677: Execution Loop Error Burst Event

- Branch: `codex/v8-execution-loop-error-burst-event`.
- Scope: Phase 5 text-log-to-event migration.
- Result: the existing execution-loop error burst warning now emits a bounded
  structured `health.summary` event with reason code
  `execution_loop_error_burst` before the existing stdlib warning. Emission is
  best-effort, uses the existing warning threshold, redacts/caps latest error
  text, and does not change restart/backoff/trading behavior or default console
  volume.
- Review evidence: Claude and Hermes approved head `409f5d8e`; focused pytest,
  compileall, and `git diff --check` passed before merge.
- VPS5 evidence: deployed to VPS5 at `eda7cb2f` with a full bot restart. Three
  smoke windows reported all five configured bots running, no hard failures, no
  log hard matches, and no missing expected processes. Settled windows still
  showed non-hard EMA readiness / staged-execution degradation and GateIO ZEC
  HSL cooldown, which remain separate operational signals.

### PR #678: Execution Burst Progress Update

- Branch: `codex/v8-progress-after-execution-burst-event`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #677 merged and VPS5 smoke confirmed the execution-loop error burst event
  migration.

### PR #679: Smoke Problem Event Context

- Branch: `codex/v8-smoke-problem-event-data`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes bounded,
  allowlisted `latest_data` for relevant problem-event groups such as
  `ema.unavailable` and `cycle.degraded`, with recursive redaction and payload
  bounds. The text-log scanner also uses the last parsed timestamp as context
  for unparseable continuation lines inside active windows, so stale traceback
  fragments after old errors are skipped while current traceback signals remain
  preserved.
- Review evidence: Hermes approved the original and amended delta; CI was
  green; focused smoke-report tests, compileall, and `git diff --check` passed
  before merge. Claude did not return during repeated polls, and Composer had
  been retired, so this docs/tooling-only slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `ff714b61` without bot restart. A
  2-minute smoke with `--log-window-unparsed-policy drop` and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no text-log hard or
  attention matches, and exposed useful `latest_data` for the remaining
  non-hard EMA/cycle readiness groups.

### PR #680: Smoke Problem Context Progress

- Branch: `codex/v8-progress-after-smoke-problem-context`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #679 merged and VPS5 smoke confirmed the bounded problem-event context.

### PR #682: Smoke Problem Event Groups

- Branch: `codex/v8-smoke-problem-event-groups`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now reports top-level
  `problem_event_count` and bounded `problem_event_groups` aggregates grouped
  by bot, event type, reason code, status, hard flag, symbol, and position
  side. Existing bounded `problem_events` samples remain available for detail.
- Review evidence: Hermes approved head `048e8595c`; CI was green; focused
  smoke-report tests, compileall, and `git diff --check` passed before merge.
  Claude did not return during repeated polls, and Composer had been retired,
  so this read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `998d7c9c` without bot restart. The first
  smoke surfaced a real transient Kucoin authoritative account-refresh timeout;
  a later settled 2-minute smoke reported all five configured bots running, no
  hard failures, no hard problem events, no log matches, no remote-call
  failures, and grouped the remaining known non-hard EMA/cycle/HSL attention in
  `problem_event_groups`.

### PR #683: Smoke Grouping Progress Update

- Branch: `codex/v8-progress-after-smoke-grouping`.
- Scope: process tracking.
- Result: updated this progress ledger and the live operations backlog after
  PR #682 merged and VPS5 smoke confirmed grouped problem-event summaries.
- Review evidence: Hermes approved head `3fa8d819`; CI was green; `git diff
  --check` passed before merge. Claude did not return during repeated polls, and
  Composer had been retired, so this docs-only slice used the degraded gate.

### PR #684: Contextless Traceback Smoke Filter

- Branch: `codex/v8-smoke-drop-contextless-traceback`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report --log-window-unparsed-policy drop`
  now skips unparseable log lines without in-window timestamp context, avoiding
  stale hard failures when the inspected log tail begins in the middle of an old
  traceback. The direct smoke-report and incident-bundle help text now describes
  the context-based drop behavior.
- Review evidence: Hermes approved original head `f01996ec` with one minor
  help-text mismatch, then approved the fixed head `04ca717`; CI was green;
  focused smoke-report tests, compileall, and `git diff --check` passed before
  merge. Claude did not return during repeated polls, and Composer had been
  retired, so this read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `dff40001` without bot restart. A
  1-minute smoke with `--log-window-unparsed-policy drop` and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no log hard or attention
  matches, and no remote-call failures.

### PR #686: Smoke Repository Metadata

- Branch: `codex/v8-smoke-repo-metadata`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes a best-effort
  `repository` block with worktree root, branch, head, full head, tracked-only
  dirty status, and tracked change count. Git lookup failures remain
  observational and do not affect smoke hard-failure accounting.
- Review evidence: Hermes approved head `b03f4139`; CI was green; focused
  smoke-report and incident-bundle tests, compileall, and `git diff --check`
  passed before merge. Claude did not return during repeated polls, and
  Composer had been retired, so this read-only tooling slice used the degraded
  gate.
- VPS5 evidence: deployed to VPS5 at `9e898019` without bot restart. The smoke
  report confirmed `repository.branch=v8`, `repository.head=9e898019`,
  `repository.dirty=false`, `tracked_changes=0`, and all five configured bots
  running. The same smoke surfaced a separate Kucoin operational issue:
  repeated authoritative balance/positions/open-orders `RequestTimeout` events
  with 98-140s staged refresh wall times and websocket ping timeouts.

### PR #688: Remote-Call Timing Smoke Summary

- Branch: `codex/v8-smoke-remote-call-timings`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes bounded
  `remote_call_timings` groups for terminal remote calls that expose elapsed
  median, p95, min, max, latest ids, and latest elapsed time. The section is
  observational only and does not affect `ok`, `attention`, or trading
  behavior.
- Review evidence: Hermes approved head `9945a3d3`; CI was green; focused
  smoke-report and incident-bundle tests, compileall, and `git diff --check`
  passed before merge. Claude did not return during repeated polls, and
  Composer had been retired, so this read-only tooling slice used the degraded
  gate.
- VPS5 evidence: deployed to VPS5 at `11f7d142` without bot restart. A
  5-minute smoke with text logs and `/root/bots_vps5.yaml` process matching
  reported all five configured bots running, no hard failures, no hard problem
  events, no log hard or attention matches, no remote-call failures, and
  `remote_call_timings.total=637`.

### PR #690: Remote-Call Health Smoke Summary

- Branch: `codex/v8-smoke-remote-call-health`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes bounded
  `remote_call_health` groups that roll up terminal remote calls by
  bot/component/kind/surface, with success/failure/throttle counts, failure and
  throttle percentages, latency summaries, reason/error counts, and bounded
  affected-symbol samples. Throttled terminal buckets are derived from
  `event_type`, while differing raw statuses remain auxiliary context.
- Review evidence: Hermes first found that `remote_call.throttled` events with
  raw `status="deferred"` were not counted as throttles, then approved fixed
  head `dc99378a`; CI was green; focused smoke-report and incident-bundle
  tests, compileall, and `git diff --check` passed before merge. Claude did
  not return during repeated polls, and Composer had been retired, so this
  read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `b150176f` without bot restart. A
  5-minute smoke with text logs, `--log-window-unparsed-policy drop`, and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no log hard or attention
  matches, no remote-call failures, and `remote_call_health.total=445`.

### PR #692: Remote-Call Health Top-Level Totals

- Branch: `codex/v8-smoke-remote-call-health-totals`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes top-level
  `remote_call_health` success, failure, throttle, failure-percent, and
  throttle-percent totals in addition to the existing bounded per-group health
  details. This keeps operator smoke summaries scannable without changing
  `ok`, `attention`, event producers, or trading behavior.
- Review evidence: Hermes first found that the new aggregate failure/throttle
  counters could be overwritten by per-group counters, then approved fixed head
  `ac4afe3f`; CI was green; focused smoke-report and incident-bundle tests,
  compileall, and `git diff --check` passed before merge. Claude did not return
  during repeated polls, and Composer had been retired, so this read-only
  tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `c8ce4880` without bot restart. A
  5-minute smoke with text logs, `--log-window-unparsed-policy drop`, and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no log hard or attention
  matches, and top-level `remote_call_health` totals:
  `total=390`, `succeeded=389`, `failed=1`, `throttled=0`, `failure_pct=0`,
  and `throttled_pct=0`.

### PR #694: Account-Critical Remote-Call Health

- Branch: `codex/v8-smoke-authoritative-health`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report` now includes
  `account_critical_remote_call_health`, a filtered view of terminal
  authoritative balance, position, open-order, and Hyperliquid split
  account-state surfaces. It reuses the existing bounded remote-call health
  summarizer while excluding broader candle/fill traffic.
- Review evidence: Hermes approved head `bebbb3f6`; CI was green; focused
  smoke-report and incident-bundle tests, compileall, `git diff --check`, and
  the touched-file silent-handling audit passed before merge. Claude did not
  return during repeated polls, and Composer had been retired, so this
  read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `3299c1ca` without bot restart. A
  5-minute smoke with text logs, `--log-window-unparsed-policy drop`, and
  `/root/bots_vps5.yaml` process matching reported all five configured bots
  running, no hard failures, no hard problem events, no log hard or attention
  matches, and account-critical health totals:
  `total=126`, `succeeded=126`, `failed=0`, `throttled=0`, `failure_pct=0`,
  and `throttled_pct=0`.

### PR #696: Concise Live Smoke Summary

- Branch: `codex/v8-smoke-report-summary`.
- Scope: operator smoke tooling.
- Result: `passivbot tool live-smoke-report --summary` now projects the full
  report down to high-signal smoke fields: health booleans/counters,
  repository state, monitor totals, event/log windows, process summary,
  bounded problem groups, remote-call/account-critical health, and risk events.
  `--compact` can be combined with `--summary` for short machine-readable
  output. Full report generation and exit-code behavior are unchanged.
- Review evidence: Hermes approved head `f1efbe45`; CI was green; focused
  smoke-report tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge. Claude did not return during
  repeated polls, and Composer had been retired, so this read-only tooling
  slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `d850daf5` without bot restart. A
  2-minute compact summary smoke with text logs,
  `--log-window-unparsed-policy drop`, and `/root/bots_vps5.yaml` process
  matching reported all five configured bots running, no hard failures, no log
  hard or attention matches, account-critical calls `total=58`,
  `succeeded=58`, and all terminal remote calls `total=169`,
  `succeeded=169`.

### PR #681: Staged Refresh Timing Events

### PR #698: Smoke Repository Root Redaction

- Branch: `codex/v8-smoke-report-redact-repo-root`.
- Scope: operator smoke tooling privacy.
- Result: `live-smoke-report` now redacts current-home, `/root`,
  `/home/<user>`, and `/Users/<user>` prefixes from the serialized
  `repository.root` field while continuing to run git commands against the real
  resolved repository path. Incident-bundle `smoke_report.json` inherits the
  same safer display field.
- Review evidence: current-head Claude and Hermes approved head `7c7368f3`;
  CI was green; focused smoke-report/incident-bundle tests, compileall, and
  `git diff --check` passed before merge.
- VPS5 evidence: deployed to VPS5 as part of the later `d5639813` pull without
  bot restart. The settled smoke reported clean repository state on `v8`, all
  five configured bots running, and no hard failures.

### PR #699: Dropped Unparsed Smoke Log Signal Counters

- Branch: `codex/v8-smoke-report-dropped-unparsed-counters`.
- Scope: operator smoke tooling.
- Result: when `--log-window-unparsed-policy drop` suppresses a contextless
  unparseable log line that still matches attention/hard patterns, smoke
  reports now expose dropped attention/hard counters and include dropped
  attention signals in `attention_count`. Dropped contextless fragments remain
  excluded from `hard_failures`, preserving the stale-tail false-positive
  suppression from PR #684.
- Review evidence: current-head Claude and Hermes approved rebased head
  `4e2fcee7`; CI was green; full smoke-report and incident-bundle tests,
  compileall, and `git diff --check` passed before merge.
- VPS5 evidence: deployed to VPS5 at `d5639813` without bot restart. Two
  immediate smokes caught real Kucoin authoritative `RequestTimeout` bursts; a
  later settled 2-minute compact summary smoke reported `ok=true`, no hard
  failures, no log hard/attention matches, all five configured bots running,
  account-critical calls `total=81`, `succeeded=81`, and all terminal remote
  calls `total=230`, `succeeded=230`.

### PR #709: Fill Cache Ready Event

- Branch: `codex/v8-fills-cache-ready-event`.
- Scope: Phase 5 startup fill-cache observability.
- Result: startup fill-cache readiness now emits a structured
  `fills.refresh_summary` event with reason code `fill_cache_ready`, source
  `startup`, refresh mode `cache_load`, elapsed time, event count, and optional
  history scope. The existing console line remains unchanged and the structured
  event route stays off console/text.
- Review evidence: current-head Claude and Hermes approved head `f5838dfea`,
  and CI was green. Focused live-event, fill-cache init/update, monitor emitter,
  compileall, and `git diff --check` checks passed before merge.
- VPS5 evidence: deployed to VPS5 at `71479c61` with a bot restart. A settled
  5-minute brief smoke reported `ok=true`, no hard failures, no log hard
  matches, all five configured bots running, no failed remote calls, and no
  failed account-critical remote calls. Direct monitor-file checks found
  `fill_cache_ready` events for Binance, GateIO, Hyperliquid, Kucoin, and OKX.

### PR #711: Exchange Time-Sync Event

- Branch: `codex/v8-exchange-time-sync-event`.
- Scope: Phase 5 exchange timestamp/nonce recovery observability.
- Result: CCXT timestamp/nonce recovery now emits bounded
  `exchange.time_sync` events for recovery success or unavailable exchange
  hooks. The event route stays off console/text, and event emission is
  best-effort so it cannot mask the original recovery path.
- Review evidence: current-head Claude and Hermes approved head `225a0e2b8`,
  and CI was green. Focused live-event, exchange time-sync recovery, execution
  loop timestamp-error, compileall, and `git diff --check` checks passed before
  merge.
- VPS5 evidence: deployed to VPS5 at `0fa6269b` with a bot restart. A settled
  5-minute brief smoke reported `ok=true`, no hard failures, no log hard
  matches, all five configured bots running, no failed remote calls, and no
  failed account-critical remote calls.

### PR #712: Supervisor Process Diagnostics

- Branch: `codex/v8-smoke-supervisor-process-diagnostics`.
- Scope: operator smoke tooling.
- Result: `live-smoke-report --supervisor-config` now classifies expected
  process matches, duplicate configured-command matches, and extra/orphan-like
  `passivbot live` processes from bounded local process-table metadata. The
  report explicitly states that tmux pane ownership is not available from this
  read-only process-table classifier.
- Review evidence: Claude first found the no-RSS `ps` fallback row parser
  dropped valid process rows, then approved fixed head `da39c8af`; Hermes
  approved the same fixed head, CI was green, and focused smoke-report and
  incident-bundle tests plus `git diff --check` passed before merge.
- VPS5 evidence: pulled to VPS5 at `51ba92a3` without bot restart. A settled
  5-minute brief/summary smoke reported `ok=true`, no hard failures, no log hard
  matches, all five configured bots running, no failed remote calls, no failed
  account-critical remote calls, and zero duplicate/extra live process matches.

### PR #714: Live Config Preflight Tool

- Branch: `codex/v8-live-config-preflight`.
- Scope: adjacent operator tooling.
- Result: added `passivbot tool live-config-preflight`, a read-only offline JSON
  report for one live config covering identity hints, HSL settings, approved and
  ignored universe counts with bounded samples, forager slots/staleness, and
  cache-related live settings. The tool does not load credentials, contact
  exchanges, or enforce startup policy.
- Review evidence: Hermes approved the original head `b2557db0`; after PR #713
  merged, the branch was rebased to `b51a15a2` with the same tool code plus
  resolved progress-doc context. CI was green, local focused preflight tests
  passed, and `git diff --check` passed. Claude did not return during repeated
  polls, so this read-only tooling slice used the degraded gate.
- VPS5 evidence: pulled to VPS5 at `7b12d4b2` without bot restart. A
  `live-config-preflight --compact` smoke against
  `configs/forager_3pos_hsl_2026-06-26.json` returned `ok=true`, reported
  bounded approved/ignored coin samples and HSL/forager/cache settings, and
  surfaced one warning for missing short-side bot config.

### PR #715: Shutdown Event Smoke Summary

- Branch: `codex/v8-smoke-shutdown-summary`.
- Scope: operator smoke tooling.
- Result: existing `bot.stopping`, `bot.shutdown.stage`, and `bot.stopped`
  structured events are now summarized as `shutdown_events` in the full,
  `--summary`, and `--brief` smoke-report projections. The change is passive and
  does not add shutdown control, process signaling, or trading behavior.
- Review evidence: Hermes approved heads `01574b4d` and `7dfa6c4a`; after
  PR #714 merged, the branch was rebased to `befade50d` with only current-base
  changelog/tool-doc context added underneath. CI was green, local
  smoke-report/incident-bundle tests and compileall passed, and `git diff
  --check` passed. Claude did not return during repeated polls, so this
  read-only tooling slice used the degraded gate.
- VPS5 evidence: pulled to VPS5 at `7b12d4b2` without bot restart. A settled
  5-minute brief/summary smoke reported no hard failures, no log hard matches,
  all five configured bots running, no failed remote/account-critical calls,
  clean tracked repository state, and `shutdown_events.total=0` because no
  restart occurred.

### PR #719: Live Restart Smoke Plan Tool

- Branch: `codex/v8-live-restart-smoke-plan`.
- Scope: adjacent operator restart/smoke tooling.
- Result: added `passivbot tool live-restart-smoke-plan`, a read-only dry-run
  planner for the repeated live restart/smoke routine. The tool parses a
  tmuxp-style supervisor config through the existing sanitized smoke-report
  parser and emits structured plan metadata, per-bot phases, repo checks, smoke
  command wiring, timeout/escalation guidance, and explicit non-execution
  policy. `--execute` is rejected; the tool does not SSH, invoke tmux, signal
  processes, pull code, start bots, contact exchanges, or load credentials.
- Review evidence: Hermes approved head `c7c4ec09`; CI was green; local
  focused restart-plan, smoke-report, CLI dispatch, compileall, and `git diff
  --check` validation passed before merge. Claude did not return before merge,
  so this plan-only tooling slice used the degraded gate.
- VPS5 evidence: pulled to VPS5 at `27f597be` without bot restart as part of
  the PR #719/#720 deploy. A dry-run plan against `/root/bots_vps5.yaml`
  reported `ok=true`, `dry_run=true`, `execution_available=false`, five planned
  bots, no issues, and the expected rejected operations.

### PR #720: EMA Live Event Debug Profile

- Branch: `codex/v8-ema-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `ema` as a `logging.live_event_debug_profiles` /
  `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES` profile, including `ema-readiness`
  aliases. When `ema` is enabled, existing `ema.unavailable` events include
  bounded parsed EMA type, span, and inner reason summaries.
  Default events remain compact, console output is unchanged, and no
  exchange/order/risk behavior changed.
- Review evidence: Hermes approved original head `1a9a53218`; after PR #719
  merged, the branch was rebased to `b8be04654` with the same code delta, CI was
  green, and local `tests/test_live_event_bus.py`,
  `tests/test_passivbot_monitor.py`, compileall, and `git diff --check`
  validation passed. Claude did not return before merge, so this opt-in
  observability slice used the degraded gate.
- VPS5 evidence: pulled to VPS5 at `27f597be` without bot restart because the
  profile is opt-in and no VPS config enabled it. A 10-minute brief smoke
  reported all five configured bots running, no hard failures, no log hard
  matches, no failed remote calls, no failed account-critical remote calls, and
  clean tracked repository state.

### PR #722: Cache Doctor Candle Coverage Evidence

- Branch: `codex/maxwell-cache-integrity-doctor-13a`.
- Scope: adjacent cache/warmup diagnostics.
- Result: `passivbot tool cache-integrity-doctor` now derives v2 candle
  coverage windows, valid row counts, suspicious interior gap samples, and
  non-enforcing warm-cache evidence labels from local `.valid.npy` artifacts.
  It remains read-only and does not change cache materialization, startup, or
  trading behavior.
- Review evidence: Hermes approved head `43f6d17b`; CI was green; Maxwell ran
  focused cache-doctor tests, compileall, `git diff --check`, and a touched-file
  silent-handling audit before opening the PR. Claude did not return during the
  merge window, so this read-only tooling slice used the degraded gate.
- VPS5 evidence: deployed as part of the `09ae3773` pull without bot restart.
  The 10-minute brief smoke reported all five configured bots running, clean
  tracked repository state, no hard failures, no log hard matches, no failed
  remote calls, and no failed account-critical remote calls.

### PR #727: Cache Doctor Warm-Cache Readiness Evidence

- Branch: `codex/maxwell-cache-warm-readiness`.
- Scope: adjacent cache/warmup diagnostics.
- Result: `passivbot tool cache-integrity-doctor` now adds report-only
  `warm_cache_readiness` summaries derived from already-scanned candle, fill,
  and HSL/risk cache metadata. The readiness projection is explicitly
  non-enforcing and does not change startup or trading behavior.
- Review evidence: Claude and Hermes approved head `155e3640`; CI was green;
  focused cache-doctor tests, compileall, `git diff --check`, and
  touched-file silent-handling audit passed before merge. A parent-side
  temporary-worktree validation also passed the focused test/check set.

### PR #723: Remote-Call Debug Profile

- Branch: `codex/v8-remote-call-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `remote_calls` debug-profile enrichment for candle remote-fetch
  and authoritative state-fetch events. Enrichment is bounded to key shape,
  selected timing/correlation fields, status/surface/kind, and counts; default
  events remain unchanged, console output is unchanged, and no raw payloads are
  added.
- Review evidence: Hermes approved head `e78d79b0`; CI was green; focused
  remote-call profile tests, the broader live-event/monitor suite, compileall,
  `git diff --check`, and touched-file silent-handling audit passed before
  merge. Claude did not return during the merge window, so this opt-in
  observability slice used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `09ae3773` without bot restart because the
  profile is opt-in and no VPS config enabled it. The same 10-minute brief
  smoke reported no hard failures, all expected bots matched, and zero failed
  remote/account-critical calls. Remaining attention came from known non-hard
  EMA/staged-readiness and HSL cooldown events.

### PR #724: Candle Debug Profile

- Branch: `codex/v8-candle-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `candles` debug-profile enrichment to existing
  `candle.tail_projected` and `candle.coverage_checked` events, exposing bounded
  key-shape, timeframe/window, and missing-coverage counters without raw candle
  arrays or default console/log changes.
- Review evidence: Hermes approved head `4454fd09`; CI was green; focused
  candle debug-profile tests, the broader live-event/monitor suite, compileall,
  `git diff --check`, and touched-file silent-handling audit passed before
  merge. Claude did not return before merge, so this opt-in observability slice
  used the degraded gate.
- VPS5 evidence: deployed to VPS5 at `f3969dbc` without bot restart because the
  profile is opt-in and no VPS config enabled it. A 10-minute brief smoke
  reported `ok=true`, no hard failures, all five expected bots matched, clean
  tracked repository state, no failed remote calls, and no failed
  account-critical remote calls.

### PR #726: Reviewer Follow-Ups

- Branch: `codex/v8-reviewer-followups`.
- Scope: Claude retrospective follow-up for already-merged low-risk
  observability/tooling slices.
- Result: redacted shareable live-ops path fields consistently, removed
  shutdown event message echo from smoke summaries, scopes EMA debug enrichment
  to the `ema` profile only, and keeps Rust debug sample construction
  best-effort inside the event-emitter path.
- Review evidence: Claude and Hermes approved head `8f4465a9`, CI was green,
  and local focused tests plus the broader live-event/monitor/smoke/preflight
  suite, compileall, `git diff --check`, and touched-file silent-handling audit
  passed before merge.
- VPS5 evidence: deployed to VPS5 at `41961266` without bot restart because the
  slice is observability/tooling-only. A 10-minute brief smoke reported
  `ok=true`, no hard failures, all five expected bots matched, clean tracked
  repository state, no failed remote calls, and no failed account-critical
  remote calls. Remaining attention came from known non-hard EMA/staged
  readiness and HSL status/cooldown events.

### PR #728: Fills Debug Profile

- Branch: `codex/v8-fills-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `fills` debug-profile enrichment to existing
  `fills.refresh_summary` and `fill.ingested` events with bounded coverage,
  count, and key-shape metadata. Default events and console output stay
  unchanged, and no raw fill IDs, source IDs, or payload values are emitted.
- Review evidence: Claude and Hermes approved the original head `4c58a718`;
  after PR #727 merged, the branch was rebased to `8ba083f6`, CI was green, and
  both reviewers confirmed the rebased patch was unchanged. Local focused tests,
  the broader live-event/monitor suite, compileall, and `git diff --check`
  passed before merge.
- VPS5 evidence: deployed to VPS5 at `5714d36d` without bot restart because the
  slice is opt-in and no VPS config enabled it. The first 10-minute smoke was
  red only because text logs contained a fresh OKX ccxt-pro websocket callback
  traceback after a reconnect; structured monitor events had no hard failures,
  all five expected bots matched, the repository was clean, and remote-call /
  account-critical failures were zero. A settled 2-minute follow-up smoke
  reported `ok=true`, no hard failures, no log hard matches, all five expected
  bots matched, clean tracked repository state, no failed remote calls, and no
  failed account-critical remote calls.

### PR #732: Execution Debug Profile

- Branch: `codex/v8-execution-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment.
- Result: added `execution` debug-profile enrichment to existing
  order-wave, order-write, create-filter, and confirmation events with bounded
  key-shape/counter metadata. Default execution event payloads and console
  output remain unchanged, and raw order payload values are not added.
- Review evidence: Hermes found and then approved the raw-order-value leak fix
  at `cb1f2c23`; after rebasing onto current `v8`, CI was green and focused
  execution/live-event tests, compileall, and `git diff --check` passed. Claude
  did not return before merge, so this opt-in observability slice used the
  degraded gate.
- VPS5 evidence: deployed together with PRs #730 and #731 at `9bc2c37f`.
  Settled 5-minute smoke returned `ok=true`, no hard failures, no log hard
  matches, all five expected bots matched, clean tracked repository state, no
  failed remote calls, no failed account-critical remote calls, and no monitor
  errors or warnings.

### PR #734: Retrospective Tool Hygiene

- Branch: `codex/v8-retro-tool-hygiene`.
- Scope: Claude retrospective follow-up for already-merged operator tooling.
- Result: collapsed user/deploy prefixes in shareable restart/preflight/HSL
  preview output, resolved grouped and flat bot-side config keys in preflight
  reports, and kept HSL startup-preview event data scalar and allowlisted.
- Review evidence: Claude and Hermes approved; CI was green; focused
  preflight/HSL preview/restart-plan tests, compileall, `git diff --check`, and
  touched-file audits passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  operator tooling. Existing bots were left running.

### PR #735: Startup Budget Smoke Projections

- Branch: `codex/maxwell-startup-budget-smoke`.
- Scope: adjacent startup/warmup observability.
- Result: `live-smoke-report` now projects existing `bot.startup_timing` events
  into report-only elapsed and per-phase budget evidence using recent local p95
  baselines. The slice does not enforce budgets or change startup behavior.
- Review evidence: Claude and Hermes approved; CI was green; focused
  smoke-report tests, compileall, and `git diff --check` passed before merge.
- VPS5 evidence: deployed without bot restart. A settled follow-up smoke
  reported all five expected bots running, no hard failures, no log hard
  matches, no failed remote calls, and no failed account-critical calls.

### PR #736: EMA Readiness Handoff Backlog

- Branch: `codex/v8-backlog-ema-readiness-race`.
- Scope: living backlog / progress tracking.
- Result: recorded the OKX/AAVE active-symbol forager EMA readiness handoff gap
  as live-ops backlog item #17 and noted that the first remediation must
  preserve the no-fabricated-ranking-values contract.
- Review evidence: Claude and Hermes approved; CI was green before merge.
- VPS5 evidence: pulled to VPS5 without bot restart; all five configured bots
  remained running.

### PR #737: Active Forager EMA Carry-Forward

- Branch: `codex/v8-forager-active-ema-projection`.
- Scope: live forager EMA readiness hardening for active/normal symbols.
- Result: active/normal forager symbols may use bounded cached real-candle
  quote-volume and log-range EMA values when the current EMA read is transiently
  unavailable, while candidate-only symbols still become unavailable and
  active/normal symbols without cached values still fail loudly. The fix keeps
  open-tail projection valid for close EMA readiness only in forager mode, so
  qv/log-range ranking inputs do not come from projected open-tail values.
- Review evidence: Hermes first found that cached metric fallback and open-tail
  projection shared one eligibility map; fixed head `6965783e` split cached
  metric and projection eligibility and added a regression covering projected
  qv/log-range values differing from cached real-candle values. Claude and
  Hermes approved the fixed head; CI was green; focused EMA tests, compileall,
  `git diff --check`, and a diff-only silent-handling audit passed before
  merge.
- VPS5 evidence: deployed to VPS5 at `e3429ee9` with a bot restart. Settled
  2-minute and 5-minute smokes reported all five expected bots running, no hard
  failures, no log hard matches, no failed remote calls, no failed
  account-critical remote calls, and clean tracked repository state.

### PR #739: Restart Plan Process Signal Safety

- Branch: `codex/v8-restart-plan-process-safety`.
- Scope: adjacent operator restart/smoke planning.
- Result: `passivbot tool live-restart-smoke-plan` now includes a
  `process_signal_safety` contract that warns future restart automation away
  from broad `pkill -f` / `pgrep -f` live-bot process matches and toward exact
  tmux panes or exact canonical process rows. The plan remains read-only and
  execution-unavailable.
- Review evidence: Claude and Hermes approved; CI was green; focused
  restart-plan tests, compileall, `git diff --check`, and the diff-only
  touched-file silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is plan-only
  operator tooling. A settled 5-minute smoke at `87e53dac` reported all five
  expected bots running, no hard failures, no log hard matches, no failed
  remote calls, no failed account-critical remote calls, and clean tracked
  repository state.

### PR #741: Ticker Probe Time Sync Health

- Branch: `codex/v8-ticker-probe-clock-skew`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` now records bounded
  `fetch_time` clock-skew evidence in each repeat and summarizes it as
  per-user and collection-level `time_sync_health`. Unsupported exchanges are
  counted separately from true failures, and `--skip-time-sync` omits the extra
  read-only time call.
- Review evidence: Hermes first found that inherited CCXT `fetch_time` methods
  can exist when `has["fetchTime"]` is false or missing. The fixed head
  `6d24b8b7` gates on `has["fetchTime"] is True`, remaps `NotSupported` to
  unsupported/skipped, and adds a regression proving the unsupported inherited
  method is not called. Claude and Hermes approved the fixed head; CI was
  green; focused ticker-probe tests, compileall, `git diff --check`, and the
  touched-file silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A compact smoke at `d4c28058` reported all five expected bots
  running, no hard failures, no log hard matches, no failed account-critical
  remote calls, and clean tracked repository state. A real Binance
  `--account-only` probe produced `time_sync_health.total=1`,
  `succeeded=1`, `failed=0`, `unsupported=0`, and `max_abs_clock_skew_ms=14`.

### PR #743: Ticker Probe Candle Freshness Health

- Branch: `codex/v8-ticker-probe-candle-freshness`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` now derives
  `candle_freshness_health` from the existing 1m OHLCV tail probe results. The
  summary reports symbol success/failure counts, current-incomplete candle
  counts, last-candle age statistics, and the worst-age symbol without making
  additional exchange calls.
- Review evidence: Claude and Hermes approved; CI was green; focused
  ticker-probe tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A compact smoke at `1fe1292b` reported all five expected bots
  running, no hard failures, no log hard matches, no failed remote calls, no
  failed account-critical remote calls, and clean tracked repository state. A
  real Binance public-only probe for `BTC/USDT:USDT` produced
  `candle_freshness_health.total_symbols=1`, `succeeded_symbols=1`,
  `failed_symbols=0`, `current_incomplete_symbols=1`, and
  `worst_symbol=BTC/USDT:USDT`.

### PR #745: Ticker Probe Fill History Health

- Branch: `codex/v8-ticker-probe-fill-history-health`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` now derives
  `fill_history_health` from the existing first-symbol `fetch_my_trades`
  sample. The summary reports success/failure counts, latency, trade count,
  newest timestamp, side/symbol shape, and id/order presence counts without raw
  trade/order ids or raw fill payloads. It intentionally does not add fill
  pagination calls.
- Review evidence: Claude and Hermes approved; CI was green; focused
  ticker-probe tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A compact smoke at `4130155e` reported all five expected bots
  running, no hard failures, no log hard matches, no failed remote calls, no
  failed account-critical remote calls, and clean tracked repository state. A
  one-repeat authenticated Binance probe for `BTC/USDT:USDT` validated
  `fill_history_health.total=1`, `succeeded=1`, `failed=0`,
  `latest_symbol=BTC/USDT:USDT`, and `latest_trade_count=0`.

### PR #747: Ticker Probe Rate Limit Health

- Branch: `codex/v8-ticker-probe-rate-limit-health`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` now derives
  `rate_limit_health` from existing probe outcomes and CCXT
  `rateLimit`/`enableRateLimit` metadata. The summary reports observed
  public/private/concurrent call counts, endpoint counts, configured sleep, and
  an estimated minimum serial duration without adding exchange calls or
  enforcing throttles.
- Review evidence: Claude and Hermes approved; CI was green; focused
  ticker-probe tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A brief smoke at `74270454` reported all five expected bots
  running, no hard failures, no log hard matches, no failed remote calls, no
  failed account-critical remote calls, and clean tracked repository state. A
  one-repeat authenticated Binance probe for `BTC/USDT:USDT` validated
  `rate_limit_health.observed_call_count=12`, `public_call_count=6`,
  `private_call_count=5`, `concurrent_request_count=1`,
  `exchange_rate_limit_ms=50`, and `estimated_min_serial_ms=600`.

### PR #749: Ticker Probe Fill Pagination Sample

- Branch: `codex/v8-ticker-probe-fill-pagination-sample`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` keeps the default
  one-call first-symbol `fetch_my_trades` sample, and adds opt-in bounded
  pagination through `--fill-history-pages` and `--fill-history-page-limit`.
  The probe records only page/count/timestamp/latency summaries and terminal
  pagination reason, with no raw trade/order ids.
- Review evidence: Claude and Hermes approved; CI was green; focused
  ticker-probe tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A 5-minute smoke and a settled 3-minute smoke at `16c25149`
  reported all five expected bots running, no hard failures, no log hard
  matches, no failed remote calls, no failed account-critical remote calls, and
  clean tracked repository state. A one-repeat authenticated Binance probe for
  `BTC/USDT:USDT` with `--fill-history-pages 2 --fill-history-page-limit 2`
  validated `fill_history_health.total=1`, `succeeded=1`, `failed=0`,
  `latest_call_count=1`, `latest_page_count=1`,
  `latest_terminal_reason=short_page`, and
  `rate_limit_health.endpoint_counts.fetch_my_trades_first_symbol=1`.

### PR #751: Ticker Probe Endpoint Latency Health

- Branch: `codex/v8-ticker-probe-endpoint-latency-health`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` now derives
  `endpoint_latency_health` from already-recorded outcomes. The summary groups
  endpoint attempts, including open-orders fallback attempts and fill-history
  pages, by endpoint/category with success/failure counts, latency summaries,
  error-type counts, and slowest endpoint metadata.
- Review evidence: Claude and Hermes approved; CI was green; focused
  ticker-probe tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A settled smoke at `4eef3572` reported all five expected bots
  running, no hard failures, no log hard matches, no failed remote calls, no
  failed account-critical remote calls, and clean tracked repository state. A
  one-repeat authenticated Binance probe for `BTC/USDT:USDT` validated
  `endpoint_latency_health.endpoint_count=11`, `total=12`, `succeeded=11`,
  `failed=1`, `slowest.endpoint=load_markets`, and the expected Binance
  all-symbol open-orders warning as one `fetch_open_orders` failure while
  account-critical health remained successful through symbol fallback.

### PR #753: Ticker Probe Exchange Surface Health

- Branch: `codex/v8-ticker-probe-exchange-surface-health`.
- Scope: read-only active exchange health probe.
- Result: `passivbot tool ticker-endpoint-probe` now derives
  `exchange_surface_health` from already-recorded open-orders, time-sync,
  fill-history, and OHLCV-tail outcomes. The summary adds exchange/user notes
  for surface quirks such as open-orders symbol fallback, unsupported time sync,
  fill-history terminal pagination reason, and OHLCV tail shape without adding
  exchange calls.
- Review evidence: Claude and Hermes approved; CI was green; focused
  ticker-probe tests, compileall, `git diff --check`, and the touched-file
  silent-handling audit passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  probe tooling. A one-repeat authenticated Binance probe for `BTC/USDT:USDT`
  validated `exchange_surface_health.notes=[fill_history_short_page,
  open_orders_all_symbols_failed, open_orders_symbol_fallback_required]`,
  `open_orders.mode_counts.symbol_fallback=1`,
  `fill_history.terminal_reasons.short_page=1`, and collection-level exchange
  note counts. A settled 5-minute smoke at `0f1afc49` reported all five
  expected bots running, no hard failures, no log hard/attention matches, no
  failed remote calls, no failed account-critical remote calls, and clean
  tracked repository state.

### PR #755: Live Smoke EMA Readiness Health

- Branch: `codex/v8-smoke-ema-readiness-health`.
- Scope: read-only smoke-report tooling.
- Result: `passivbot tool live-smoke-report` now derives bounded
  `ema_readiness_health` full/summary groups and brief `ema_readiness`
  counters from existing `ema.unavailable` events. The new projection reports
  event count, affected bot count, latest candidate/unavailable totals, bounded
  reason counts, error-type counts, latest cycle IDs, and compact allowlisted
  EMA event data without changing smoke verdict logic.
- Review evidence: Claude and Hermes approved; CI was green; focused
  smoke-report tests, compileall, `git diff --check`, and local real-data
  summary/brief smoke checks passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  smoke-report tooling. The first 10-minute smoke after deploy caught a real
  Binance `InvalidNonce`/timestamp-window recovery in authoritative positions
  and the related text-log warning; later authoritative calls succeeded. A
  settled 2-minute brief smoke at `5d9f3a5f` reported all five expected bots
  running, no hard failures, no log hard/attention matches, no failed remote
  calls, no failed account-critical remote calls, and clean tracked repository
  state. The new `ema_readiness` counters reported `total=11`, `bots=4`,
  `latest_candidate_unavailable_total=31`, and `latest_unavailable_total=112`,
  making persistent non-hard EMA readiness degradation visible without full
  problem-event inspection.

### PR #756: EMA Gate Modes For Trailing Martingale Entries

- Branch: `codex/v8-ema-entry-gate-mode`.
- Scope: adjacent Rust strategy/runtime behavior, not a logging-overhaul slice.
- Result: trailing-martingale entry EMA gating is now a fixed config policy
  with `disabled`, `initial`, `reentry`, and `all` modes. The default remains
  `initial`; unstuck EMA gating also gained an explicit fixed toggle defaulting
  to enabled.
- Review evidence: Claude and Hermes approved; CI was green; the PR author ran
  `cargo check`, rebuilt the Rust extension, ran targeted Python suites, and
  added real backtest/optimize smoke evidence. My final read agreed with the
  reviewers that Claude's remaining notes were future-maintenance concerns, not
  current merge blockers.
- VPS5 evidence: deployed after PR #757. The VPS checkout pulled to
  `19b34138`; the Rust extension was rebuilt in `/root/passivbot/venv` with
  `PATH=/root/.cargo/bin:/root/passivbot/venv/bin:$PATH` and
  `VIRTUAL_ENV=/root/passivbot/venv`. Bots were then restarted from
  `/root/bots_vps5.yaml` and left running. An immediate 3-minute smoke and a
  settled 5-minute smoke both reported all five expected bots matched, no hard
  failures, no log hard/attention matches, no failed remote calls, no failed
  account-critical calls, and clean tracked repository state. The settled smoke
  at `19b34138` reported `remote_calls.total=336`,
  `account_critical_remote_calls.total=46`, and only non-hard EMA readiness
  attention (`ema_readiness.total=4`, `bots=1`,
  `latest_candidate_unavailable_total=0`).

### PR #759: Live Smoke Staged Readiness Health

- Branch: `codex/v8-smoke-staged-readiness-health`.
- Scope: read-only smoke-report tooling.
- Result: `passivbot tool live-smoke-report` now derives bounded
  `staged_readiness_health` full/summary groups and brief `staged_readiness`
  counters from existing staged `cycle.degraded` events. The new projection
  reports affected bot count, latest missing/invalid staged-surface totals,
  missing/invalid surface groups, completed-candle mismatch counts, latest
  cycle IDs, and compact allowlisted cycle-degraded event data without changing
  smoke verdict logic.
- Review evidence: Claude and Hermes approved; CI was green; focused
  smoke-report tests, compileall, `git diff --check`, and a local real-data
  brief smoke projection passed before merge.
- VPS5 evidence: deployed without bot restart because the slice is read-only
  smoke-report tooling. A 5-minute brief smoke at `74a52ede` reported all five
  expected bots running, no hard failures, no log hard/attention matches, no
  failed remote calls, no failed account-critical calls, and clean tracked
  repository state. The new `staged_readiness` counters reported `total=4`,
  `bots=1`, `latest_missing_surface_total=1`, and
  `latest_invalid_surface_total=1`, making staged `completed_candles` style
  readiness degradation visible without full problem-event inspection.

### PR #760: Staged Readiness Deploy Progress

- Branch: `codex/v8-progress-after-staged-readiness-smoke`.
- Scope: docs-only progress update.
- Result: recorded PR #759 review, merge, deploy, and VPS5 smoke evidence in
  this progress ledger and the live-ops backlog.
- Review evidence: Claude and Hermes approved; CI was green.
- VPS5 evidence: pulled to `31d42ea3` without bot restart because the slice is
  docs-only. The first 5-minute smoke was red from real HSL ZEC long RED
  finalizations on OKX, GateIO, and Binance, not from the docs change; the
  structured `risk_events` section and text-log hard matches both surfaced the
  RED finalizations. A settled 2-minute follow-up smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `logs.attention_matches=0`,
  `matched_expected=5`, `missing_expected=[]`, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`. Remaining non-hard attention was
  visible through `ema_readiness` and `staged_readiness`; the settled
  `staged_readiness` projection reported `total=5`, `bots=3`,
  `latest_missing_surface_total=3`, and `latest_invalid_surface_total=3`.
  A later 5-minute smoke still reported `ok=true`, all five bots matched, no
  log matches, no failed remote/account-critical calls, and clean repository
  state, while `staged_readiness` had grown to `total=17`, `bots=4`,
  `latest_missing_surface_total=5`, and `latest_invalid_surface_total=5`.

### PR #762: Completed Candle Fallback Shape Recovery

- Branch: `codex/v8-staged-readiness-target-change`.
- Scope: narrow staged-readiness runtime fix driven by the PR #759/#760 smoke
  signal.
- Result: completed-candle preconditions now compare canonical
  `(symbol, completed_timestamp)` targets instead of exact signature tuple
  shape, so a stamped bounded `tail_gap_fallback` signature may recover to
  normal cache coverage without causing a spurious staged-planner defer. Symbol
  set changes, target timestamp changes, and genuinely missing/stale completed
  candles still defer.
- Review evidence: Claude and Hermes approved; CI was green; focused staged
  planner tests, `tests/test_live_smoke_report.py -k staged_readiness`,
  compileall, `git diff --check`, and the full
  `tests/test_passivbot_balance_split.py` file passed locally.
- VPS5 evidence: deployed to `d9188b64` with a bot restart because this changed
  live Python runtime code. Four bots stopped after the first exact-pane
  Ctrl+C; Kucoin needed a second exact-pane Ctrl+C before the stale `passivbot`
  tmux session was killed and reloaded from `/root/bots_vps5.yaml`. Immediate
  2-minute smoke reported `ok=true`, all five bots matched, no hard/log
  failures, no failed remote/account-critical calls, and
  `staged_readiness.total=0`. A settled 5-minute smoke also reported `ok=true`,
  all five bots matched, clean repository state, no hard/log failures, no
  failed remote/account-critical calls, and `staged_readiness.total=0`.

### PR #764: Cache Doctor Metadata Compatibility Evidence

- Branch: `codex/v8-cache-doctor-compat`.
- Scope: adjacent read-only cache diagnostics from the live-ops backlog.
- Result: `passivbot tool cache-integrity-doctor` now reports deeper
  metadata compatibility evidence for local candle/fill/HSL caches. Candle
  `index.json` known-gap metadata is classified by no-trade vs unclassified
  reasons, fill current-contract evidence distinguishes proven coverage from
  current-contract-but-unproven coverage, and HSL/risk metadata reports HSL
  artifact/timestamp compatibility fields. The final follow-up commit made
  mixed no-trade/unclassified candle gaps explicitly partial and still
  `candle_synthetic_no_trade_evidence_unproven`.
- Review evidence: Claude and Hermes approved the final head; CI was green;
  `tests/test_cache_integrity_doctor.py`, compileall, and `git diff --check`
  passed locally for the follow-up.
- VPS5 evidence: deployed as part of merged `v8` `5275ab75` without bot
  restart because this is read-only local tooling.

### PR #765: Live Smoke Event Pipeline Health

- Branch: `codex/v8-smoke-pipeline-health`.
- Scope: read-only smoke-report tooling.
- Result: `passivbot tool live-smoke-report` now derives bounded
  `event_pipeline_health` full/summary groups and brief `event_pipeline`
  counters from existing `health.summary` event-pipeline counters. The new
  projection reports latest queue depth, unfinished queue work, dropped event
  counts, sink-error counts, degraded count, worker-not-alive count, and
  stopping count without changing smoke verdict logic.
- Review evidence: Claude and Hermes approved; CI was green; full
  `tests/test_live_smoke_report.py`, compileall, and `git diff --check`
  passed locally.
- VPS5 evidence: deployed at `5275ab75` without bot restart because this is
  read-only smoke-report tooling. A 5-minute smoke confirmed the new brief
  `event_pipeline` field was present, but no recent matching health-summary
  sample was in that short window. A 30-minute smoke reported
  `event_pipeline.total=1`, `bots=1`, `latest_dropped_total=0`,
  `latest_sink_error_total=0`, `latest_worker_not_alive_count=0`, and clean
  tracked repository state. The same smoke was red from live risk state and
  unrelated runtime events: HSL red/cooldown events, CRITICAL HSL text logs,
  and an earlier Hyperliquid fill-refresh timeout. All five expected bots were
  still running, and remote/account-critical call summaries reported no
  failures.

### PR #767: Live Smoke Risk Log Classification

- Branch: `codex/v8-smoke-risk-log-classification`.
- Scope: read-only smoke-report tooling.
- Result: `passivbot tool live-smoke-report` now splits text-log attention and
  hard matches into risk/HSL-related and non-risk buckets. Full, summary, and
  brief reports include `risk_attention_matches`, `risk_hard_matches`,
  `non_risk_attention_matches`, and `non_risk_hard_matches`; bounded log match
  samples now include `category=risk|general`. Smoke verdict logic is unchanged:
  risk/HSL CRITICAL lines still count in `hard_matches` and still make smoke
  red.
- Review evidence: Claude and Hermes approved; CI was green; full
  `tests/test_live_smoke_report.py`, compileall, `git diff --check`, and the
  touched-file silent-handling audit passed locally.
- VPS5 evidence: deployed at `b07d5166` without bot restart because this is
  read-only smoke-report tooling. A 5-minute brief smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`,
  `logs.risk_hard_matches=0`, `logs.non_risk_hard_matches=0`, all five
  expected bots matched, clean tracked repository state, no failed remote
  calls, and no failed account-critical remote calls. Remaining attention was
  non-hard EMA readiness and HSL status.

### PR #769: Live Smoke Verdict Source Breakdown

- Branch: `codex/v8-smoke-source-breakdown`.
- Scope: read-only smoke-report tooling.
- Result: `passivbot tool live-smoke-report` now exposes
  `hard_failure_sources` and `attention_sources` in full, summary, and brief
  reports. The source maps identify monitor parse errors, invalid event rows,
  structured hard/problem events, text-log matches, dropped unparsed attention
  matches, and process hard failures. Smoke verdict logic is unchanged:
  `hard_failures`, `attention_count`, and `ok` use the same accounting as
  before.
- Review evidence: Claude and Hermes approved; CI was green; full
  `tests/test_live_smoke_report.py`, compileall, `git diff --check`, and the
  touched-file silent-handling audit passed locally.
- VPS5 evidence: deployed at `b789e146` without bot restart because this is
  read-only smoke-report tooling. A 5-minute brief smoke reported `ok=true`,
  `hard_failures=0`, `hard_failure_sources.total=0`, all five expected bots
  matched, clean tracked repository state, no failed remote calls, no failed
  account-critical remote calls, and no text-log attention or hard matches.
  The remaining attention was explicitly attributed to
  `attention_sources.problem_events=101`, with non-hard EMA readiness and HSL
  status events visible in the existing summaries.

### PR #772: Live Config Cache Readiness Preflight

- Branch: `codex/v8-live-config-cache-readiness`.
- Scope: read-only config preflight tooling.
- Result: `passivbot tool live-config-preflight` now includes config-only
  cache readiness/root-hint reporting for candles, fills, and HSL/risk surfaces,
  plus bounded compare deltas. The report explicitly marks cache artifacts as
  not scanned and startup policy as not enforced, so it does not claim coverage,
  touch local caches, contact exchanges, or change live startup/trading
  behavior.
- Review evidence: Claude and Hermes approved; CI was green; targeted
  `tests/test_live_config_preflight.py`, py_compile/compileall, a compact CLI
  smoke, and `git diff --check` passed.
- VPS5 evidence: deployed at `5fcb39cd` without bot restart because this is
  read-only preflight tooling. A 5-minute summary smoke reported `ok=true`,
  `hard_failures=0`, `hard_failure_sources.total=0`,
  `logs.hard_matches=0`, `logs.attention_matches=0`, all five expected bots
  matched, clean tracked repository state, no failed remote calls, and no
  failed account-critical remote calls. Remaining attention came from known
  non-hard EMA readiness and HSL cooldown/status groups.

### PR #775: Event Pipeline Health Aggregation Regression

- Branch: `codex/v8-fake-live-observability-test`.
- Scope: offline regression test and backlog ledger only.
- Result: `tests/test_live_smoke_report.py` now covers multi-bot
  `event_pipeline_health` aggregation from existing `health.summary` events,
  including queue depth, dropped events, sink errors, degraded counts, and
  worker-liveness. The PR changed no production runtime files and does not alter
  event routing, live bots, exchange calls, or trading behavior.
- Review evidence: Claude and Hermes approved; CI was green; full
  `tests/test_live_smoke_report.py`, targeted event-pipeline aggregation tests,
  py_compile/compileall, and `git diff --check` passed.
- VPS5 evidence: not deployed or smoke-tested separately because the merged
  slice changed only tests and `docs/plans/live_ops_improvement_backlog.md`.
  The latest runtime-bearing deploy/smoke evidence remains PR #772 at
  `5fcb39cd`.

### PR #773: Cache Doctor Candle Boundary Gap Summary

- Branch: `codex/v8-cache-doctor-boundary-summary`.
- Scope: read-only cache-integrity tooling and tests.
- Result: `passivbot tool cache-integrity-doctor` now exposes bounded candle
  boundary-gap summaries so operators can distinguish full coverage from
  edge-window gaps without opening cache metadata manually. The slice does not
  change live trading behavior, candle loading behavior, or exchange calls.
- Review evidence: Claude and Hermes approved after the boundary summary fix;
  CI was green.
- VPS5 evidence: deployed as part of merged `v8` `54a909b`. Bots were
  restarted and left running. A 5-minute compact smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, no failed remote or
  account-critical remote calls, `matched_expected=5`, and
  `missing_expected=[]`.

### PR #774: Structured Unstuck Events

- Branch: `codex/v8-unstuck-events`.
- Scope: live event producers for unstuck state transitions and tests.
- Result: unstuck-related live state now emits structured events through the
  event pipeline, preserving best-effort observability semantics and avoiding
  trading-behavior changes. Smoke-report value-safety was fixed before merge.
- Review evidence: Claude and Hermes approved after the value-safety fix; CI
  was green.
- VPS5 evidence: deployed as part of merged `v8` `54a909b`. The same
  post-restart smoke showed all five configured bots running with no hard
  failures, no text-log hard matches, and no failed account-critical remote
  calls. Shutdown/restart events from the deployment were visible through the
  structured smoke summaries.

### PR #778: Live Performance Report

### PR #779: Live Performance Report Summary Filters

- Branch: `codex/v8-live-performance-report-summary`.
- Scope: read-only performance-report filtering, summary projection, tests, and
  docs.
- Result: `passivbot tool live-performance-report` now supports
  `--summary`, `--bot`, `--exchange`, and `--user`, with explicit
  skipped-event filter accounting and bounded summary output. This keeps
  repeated operator performance checks concise without changing event emission,
  exchange calls, caches, or trading behavior.
- Review evidence: Claude and Hermes approved with no findings; CI was green;
  targeted performance-report, event-query, and smoke-report tests,
  py_compile, `git diff --check`, and a local compact filtered CLI smoke
  passed.
- VPS5 evidence: deployed at `0d742a4f` without bot restart because this is
  read-only tooling. A filtered Binance performance summary returned `ok=true`
  and showed HSL/startup latency as the dominant Binance performance cost. A
  5-minute summary smoke reported all five expected bots running with no hard
  failures, no text-log hard matches, and no failed account-critical remote
  calls.

### PR #780: Live Decision Boundary Lag Report

- Branch: `codex/v8-live-performance-decision-lag`.
- Scope: read-only performance-report decision-boundary lag aggregation, tests,
  and docs.
- Result: `passivbot tool live-performance-report` now includes
  `decision_boundary_lag`, aggregating per-bot lag from the relevant whole
  minute boundary to cycle start, Rust call/return, action planning,
  order-wave/write/confirmation events when present, and cycle completion. The
  report keeps cycle ids internal and surfaces only aggregate timing groups.
- Review evidence: Claude and Hermes approved with no findings; CI was green;
  targeted performance-report, event-query, and smoke-report tests,
  py_compile, `git diff --check`, and a local compact filtered CLI smoke
  passed.
- VPS5 evidence: deployed at `f70434f3` without bot restart because this is
  read-only tooling. A filtered Binance performance summary returned `ok=true`
  and showed decision-boundary lag groups directly; the same smoke pass kept
  all five configured bots running with no hard failures and no failed
  account-critical remote calls.

### PR #781: Live Input Staleness Report

- Branch: `codex/v8-live-performance-input-staleness`.
- Scope: read-only performance-report input-staleness aggregation, tests, and
  docs.
- Result: `passivbot tool live-performance-report` now includes
  `input_staleness`, aggregating account packet age at snapshot build plus
  snapshot/EMA-bundle age at the Rust call boundary when existing monitor
  events provide enough proof. Joins are keyed by bot plus cycle generation so
  reused cycle IDs after restart do not cross-link old snapshot/EMA state.
- Review evidence: Claude and Hermes approved with no findings; CI was green;
  targeted performance-report, event-query, and smoke-report tests,
  py_compile, `git diff --check`, and a local compact CLI smoke passed.
- VPS5 evidence: deployed at `cdb2f381` without bot restart because this is
  read-only tooling. A filtered Binance performance summary returned
  `ok=true` with `input_staleness` groups visible; a settled smoke after a
  transient GateIO timeout reported all five expected bots running with no hard
  failures and no failed account-critical remote calls.

### PR #784: Startup Readiness Performance Summary

### PR #787: Live Performance Slowest Blockers View

- Branch: `codex/v8-live-performance-slowest-blockers`.
- Scope: read-only live performance report projection, docs, and tests.
- Result: `passivbot tool live-performance-report` now includes
  `slowest_blockers`, a bounded cross-section ranking derived from existing
  performance, decision-boundary, and input-staleness metric groups. The view
  excludes diagnostics-only/observability groups and copies only the existing
  bounded metric fields plus `source_section` and `blocking_scope`.
- Review evidence: CI was green. Claude approved the current head with no
  findings. Hermes approved the equivalent pre-rebase code delta; the final
  rebase only moved the branch over the already-merged progress-doc update.
  Local validation covered performance-report, event-query, and smoke-report
  tests, py_compile, `git diff --check`, silent-handling audit, and a compact
  filtered CLI smoke.
- VPS5 evidence: deployed at `002fb965` without bot restart because this is
  read-only report tooling. A focused Binance performance report returned
  `ok=true` and showed `slowest_blockers` populated, with startup warmup and
  HSL-related timings ranked above lower-impact groups. A 2-minute smoke
  reported `ok=true`, `hard_failures=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected=[]`.

### PR #789: Live Performance Resource Pressure Report

- Branch: `codex/v8-live-performance-resource-pressure`.
- Scope: read-only live performance report projection, docs, and tests.
- Result: `passivbot tool live-performance-report` now includes
  `resource_pressure`, derived only from existing `health.summary` events. The
  section aggregates whitelisted process and event-pipeline fields such as RSS,
  memory percent, open file descriptors, load averages, loop duration, event
  queue depth, dropped-event counters, sink-error counters, degraded count, and
  event-pipeline worker state. It does not surface raw account, balance,
  equity, PnL, or other financial health payload fields.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Claude noted one optional non-blocking efficiency nit: the report accumulator
  stores value lists for min/max/mean, which is acceptable for this one-shot
  offline report and consistent with nearby report accumulators. Local
  validation covered performance-report, event-query, and smoke-report tests,
  py_compile, `git diff --check`, silent-handling audit, and a compact
  filtered CLI smoke.
- VPS5 evidence: deployed at `bc6e7f1d` without bot restart because this is
  read-only report tooling. A compact smoke reported `ok=true`,
  `hard_failures=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected=[]`. A focused performance summary showed
  `resource_pressure` populated for GateIO and Hyperliquid with RSS, load
  average, loop duration, event queue, sink-error, and worker-state fields.

### PR #791: Live Performance Shutdown Latency Report

- Branch: `codex/v8-live-performance-shutdown-latency`.
- Scope: read-only live performance report projection, docs, and tests.
- Result: `passivbot tool live-performance-report` now includes
  `shutdown_latency`, derived only from existing `bot.stopping`,
  `bot.shutdown.stage`, and `bot.stopped` events. The section summarizes
  per-stage cumulative shutdown elapsed time and final total shutdown duration
  while keeping the data out of trading blocker rankings and without copying
  shutdown error text.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Local validation covered performance-report, event-query, and smoke-report
  tests, py_compile, `git diff --check`, silent-handling audit, and a compact
  CLI performance-report smoke.
- VPS5 evidence: deployed at `a04bc1ed` without bot restart because this is
  read-only report tooling. A compact smoke reported `ok=true`,
  `hard_failures=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected=[]`. A focused performance summary showed
  `shutdown_latency` present; it was empty in the recent window because no
  shutdown lifecycle events occurred during that window.

### PR #793: Live Performance Execution Timing Report

- Branch: `codex/v8-live-performance-execution-timing`.
- Scope: read-only live performance report projection, docs, and tests.
- Result: `passivbot tool live-performance-report` now includes
  `execution_timing`, derived only from existing order-wave, order create/cancel,
  and confirmation events. The section reports bounded exchange-action latency
  groups plus `starts_seen`, `terminals_seen`, `timing_observations`,
  `missing_id_counts`, `unpaired_terminal_counts`, and `pending_start_counts`.
  Pairing keys are used only internally; raw order payloads, action ids, and
  client-order ids are not surfaced.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Local validation covered performance-report, event-query, and smoke-report
  tests, py_compile, `git diff --check`, and a compact synthetic CLI
  performance-report smoke.
- VPS5 evidence: deployed at `b5fc245b` without bot restart because this is
  read-only report tooling. All five configured `passivbot live` processes
  remained running after pull. A 5-minute smoke reported `hard_failures=0`,
  `remote_calls.failed=0`, `account_critical_remote_calls.failed=0`,
  `matched_expected=5`, and `missing_expected=[]`. A 180-minute performance
  summary returned `ok=true` and showed `execution_timing` present but empty
  because no order-wave/write events occurred in that sampled window. Existing
  slowest blockers were input-staleness and cycle-boundary lag groups, not
  execution writes.

### PR #796: Live Performance Readiness Checklist

### PR #799: Cache Warmup Performance Report

- Branch: `codex/v8-live-performance-cache-warmup`.
- Scope: read-only live performance report cache/warmup projection, docs, and
  tests.
- Result: `passivbot tool live-performance-report` now includes
  `cache_warmup`, derived only from existing `cache.warmup_decision`,
  `cache.load.completed`, and `cache.flush.completed` events. The section
  summarizes bounded warm-cache reuse/cold-path decisions, candle cache
  load/flush rows, reason/source counters, symbol/timeframe samples, and
  elapsed timing where present. Trading behavior, exchange calls, cache
  mutation, event producers, and console routing are unchanged.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Both reviews verified that the section is report-only and uses explicit
  scalar/counter whitelists that exclude raw cache paths, raw payloads, account
  values, and secrets. Local validation covered performance-report,
  event-query, and smoke-report tests, py_compile, `git diff --check`, local
  compact CLI performance-report smoke, and a silent-handling scan of touched
  report/test files.
- VPS5 evidence: deployed at `cb034e82` without bot restart because this is
  read-only report tooling. A 2-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  `missing_expected=[]`. A focused 5-minute performance report returned
  `ok=true` and showed `cache_warmup` populated for all five bots, with
  `total_events=614`, `cache.load.completed=469`,
  `cache.flush.completed=140`, and `cache.warmup_decision=5`. Sample groups
  showed OKX, Binance, and GateIO warmup cold-path decisions plus bounded
  candle load/flush row counts and elapsed summaries.

### PR #801: Forager EMA Readiness Performance Report

- Branch: `codex/v8-live-performance-forager-ema-readiness`.
- Scope: read-only live performance report forager/EMA readiness projection,
  docs, and tests.
- Result: `passivbot tool live-performance-report` now includes
  `forager_ema_readiness`, derived only from existing `forager.selection`,
  `forager.feature_unavailable`, `ema.unavailable`, and `ema.fallback_used`
  events. The section summarizes bounded forager selection counts,
  feature-unavailable counts, EMA unavailable reason/error-type counters, EMA
  fallback counters, pside/status/reason-code counters, configured age/budget
  fields where present, and bounded symbol samples. Trading behavior, exchange
  calls, cache mutation, event producers, and console routing are unchanged.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Both reviews verified the section is report-only, standalone, and value-safe.
  Local validation covered performance-report, event-query, and smoke-report
  tests, py_compile, `git diff --check`, a local compact CLI
  performance-report smoke, and a silent-handling scan of touched report/test
  files. Tests explicitly inject and reject raw top scores, raw EMA error text,
  API-key markers, balance/equity fields, raw payload markers, and local paths.
- VPS5 evidence: deployed at `1fc77413` without bot restart because this is
  read-only report tooling. A 10-minute time-windowed smoke reported
  `ok=true`, `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected=[]`, clean tracked repository state, and
  `account_critical_remote_calls.failed=0`. A focused 10-minute performance
  report returned `ok=true` and showed `forager_ema_readiness` populated with
  `total_events=140`, including `ema.fallback_used=41`, `ema.unavailable=56`,
  and `forager.selection=43`. The section grouped current readiness evidence
  across Binance, GateIO, OKX, and Hyperliquid.

### PR #803: Resource Pressure Percentiles

- Branch: `codex/v8-resource-pressure-percentiles`.
- Scope: read-only live performance report resource-pressure projection, docs,
  and tests.
- Result: `passivbot tool live-performance-report` `resource_pressure` field
  stats now include `count`, `median`, and `p95` in addition to the prior
  latest/min/max/mean values. Integer-only health series remain integer-valued,
  while fractional fields such as load averages and memory percentage keep
  bounded decimal precision. The section continues to derive only from existing
  `health.summary` events and continues to use the existing whitelist of
  process and event-pipeline fields.
- Review evidence: CI was green. Hermes approved current head `80bc42fd` with
  no findings. Claude did not return after repeated polling, so the PR was
  merged under the documented degraded low-risk tooling gate. Local validation
  covered performance-report, event-query, and smoke-report tests, py_compile,
  `git diff --check`, a local compact CLI performance-report smoke, and a
  silent-handling scan of touched report/test files. No event producers,
  exchange calls, cache mutation, readiness gates, console routing, or trading
  behavior changed.
- VPS5 evidence: deployed at `07f8e759` without bot restart because this is
  read-only report tooling. A 5-minute time-windowed smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected=[]`, clean tracked repository state, and
  `account_critical_remote_calls.failed=0`. A focused 30-minute performance
  report returned `ok=true` and showed `resource_pressure` populated with
  `total=8` health summary events across four bots. Sample groups confirmed
  resource fields now include count/latest/min/mean/median/p95/max values
  without surfacing raw account or financial payload fields.

### PR #805: Operation Duration Performance Summary

### PR #807: Snapshot-to-Rust Correlation Fix

- Branch: `codex/v8-live-performance-snapshot-correlation`.
- Scope: read-only live performance report correlation fix, docs, and tests.
- Result: `passivbot tool live-performance-report` now correlates
  `snapshot_to_rust` timing from `snapshot.built` to
  `rust_orchestrator.called` by exact live-event envelope cycle ID when
  available, and otherwise falls back to the latest preceding snapshot in the
  same bot/restart scope. It also reports exact-match, latest-snapshot-match,
  missing, and ambiguous counters so report consumers can see when fallback
  correlation was used. No event producers, exchange calls, cache mutation,
  readiness gates, console routing, or trading behavior changed.
- Review evidence: CI was green. Claude and Hermes approved with no findings.
  Local validation covered the focused live performance report tests,
  py_compile, `git diff --check`, and a compact local report smoke.
- VPS5 evidence: deployed at `2bb89cfa` without bot restart because this is
  read-only report tooling. A 10-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected=[]`, clean tracked repository state, and
  `account_critical_remote_calls.failed=0`. A focused 30-minute performance
  report confirmed corrected `snapshot_to_rust` values on VPS5: Binance p95
  `1449ms`, GateIO p95 `1477ms`, OKX p95 `1712ms`, and Hyperliquid p95
  `564ms`, with `snapshot_to_rust_latest_snapshot_matches=154`,
  `snapshot_to_rust_exact_matches=0`, and one missing snapshot match at the
  time-window boundary.

### PR #811: Legacy Snapshot ID Query Fallback

- Branch: `codex/v8-event-query-snapshot-data-fallback`.
- Scope: read-only `live-event-query` compatibility for legacy
  `snapshot.built` rows written before snapshot IDs were promoted into the
  structured event envelope.
- Result: `_event_ids()` now derives `snapshot_id` from
  `_live_event.data.snapshot_id` when `ids.snapshot_id` is absent, so existing
  ID filters, compact output, timelines, trace summaries, order traces, and
  cycle traces can match older snapshot events. The fallback intentionally does
  not promote `data.cycle_id`, because legacy `snapshot.built.data.cycle_id`
  is a planning snapshot epoch rather than a live cycle ID.
- Review evidence: CI was green. Hermes approved current head `6ca8a602` with
  no findings. Claude did not return after repeated polling, so the PR was
  merged under the documented degraded low-risk tooling gate. Local validation
  covered event-query tests, the adjacent CLI dispatch test, py_compile, and
  `git diff --check`. No event producers, exchange calls, cache mutation,
  readiness gates, console routing, or trading behavior changed.
- VPS5 evidence: deployed at `52f85e08` without bot restart because this is
  read-only query tooling. A 5-minute time-windowed smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`,
  `missing_expected_count=0`, clean tracked repository state,
  `remote_calls.failed=0`, and `account_critical_remote_calls.failed=0`.
  Remaining attention came from known non-hard EMA readiness problem events.

### PR #812: Event-Query Snapshot Fallback Progress

- Branch: `codex/v8-progress-after-event-query-fallback`.
- Scope: docs-only progress-ledger update for PR #811.
- Result: recorded the legacy snapshot-ID query fallback, its degraded
  low-risk merge gate, and VPS5 smoke evidence in this ledger. No code,
  tooling, event producers, exchange calls, cache mutation, readiness gates,
  console routing, or trading behavior changed.
- Review evidence: CI was green. Hermes approved current head `38da2ccf` with
  no findings. Claude did not return after repeated polling, so the PR was
  merged under the documented degraded low-risk docs gate.
- VPS5 evidence: deployed at `8e4712f6` without bot restart because this is
  docs-only. A compact smoke after the pull reported `ok=true`,
  `hard_failures=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, and
  remaining attention only from known non-hard EMA readiness events.

### PR #813: Market Snapshot Staleness Performance Report

- Branch: `codex/v8-market-snapshot-staleness-report`.
- Scope: read-only live performance report input-staleness projection and
  tests.
- Result: `passivbot tool live-performance-report` now derives
  `input_staleness.snapshot_market_stale_count` from existing
  `snapshot.built.data.market_snapshot_summary` rows and adds
  `input_staleness.market_snapshot.configured_excess` timing groups when a
  symbol's observed `max_age_ms` exceeds configured `configured_max_age_ms`.
  The bounded summary includes the stale-count field. No event producers,
  exchange calls, cache mutation, readiness gates, console routing, or trading
  behavior changed.
- Review evidence: CI was green. Hermes approved current head `a98c168ca` with
  no findings. Claude did not return after repeated polling, so the PR was
  merged under the documented degraded low-risk tooling gate. Local validation
  covered the full `tests/test_live_performance_report.py` suite, py_compile,
  `git diff --check`, and a focused silent-handling scan of touched report/test
  files.
- VPS5 evidence: deployed at `11c1a847` without bot restart because this is
  read-only report tooling. A 5-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state, `account_critical_remote_calls.failed=0`, and one non-hard
  general `remote_calls.failed=1`. A focused 30-minute performance report
  returned `ok=true` and showed `snapshots_seen=151`,
  `snapshot_surface_age_rows=906`, `snapshot_market_summaries_seen=151`,
  `snapshot_market_stale_count=0`, and `total_groups=52`; no market snapshot
  excess-age group was present in that window.

### PR #815: Event-Query Trace Taxonomy Summary

- Branch: `codex/v8-event-query-trace-taxonomy`.
- Scope: read-only `live-event-query` trace-summary taxonomy projection and
  tests.
- Result: `passivbot tool live-event-query --trace-summary` now includes
  source, component, tag, exchange, and user counters for matched structured
  live events. Tags are read from existing monitor rows and optional embedded
  live-event tags, deduplicated per event before counting. No event producers,
  exchange calls, cache mutation, readiness gates, console routing, monitor
  writes, or trading behavior changed.
- Review evidence: CI was green. Hermes approved current head `fe9f0fdc` with
  no findings. Claude did not return after repeated polling, so the PR was
  merged under the documented degraded low-risk tooling gate. Local validation
  covered `tests/test_live_event_query.py`, py_compile for touched files,
  `git diff --check`, and a silent-handling scan of touched files.
- VPS5 evidence: deployed at `404063c6` without bot restart because this is
  read-only query tooling. A 3-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state, `remote_calls.failed=0`, and
  `account_critical_remote_calls.failed=0`.

### PR #810: Snapshot IDs in Diagnostic Events

- Branch: `codex/v8-snapshot-built-envelope-ids`.
- Scope: live diagnostic event correlation IDs and tests.
- Result: `DiagnosticEvent` can now carry `cycle_id` and `snapshot_id`, and the
  live planning path passes the current live cycle ID plus planning snapshot ID
  into `snapshot.built` diagnostics. This narrows the gap between legacy
  diagnostic producers and the structured live-event envelope without changing
  trading behavior.
- Review evidence: CI was green. Local validation covered focused event-bus and
  planning snapshot tests, the adjacent event-bus, balance-split, and
  performance-report suites, py_compile, `git diff --check`, and a source-stamp
  verification of the shared local Rust extension. No order/risk/cache/exchange
  behavior changed.

### PR #817: Event-Query Tag Filtering

- Branch: `codex/v8-event-query-tag-filter`.
- Scope: read-only `live-event-query` tag filtering and tests.
- Result: `passivbot tool live-event-query` and `build_event_report()` now
  accept tag filters so event, timeline, trace-summary, order-trace, and
  cycle-trace views can be scoped by structured live-event tags. The tag filter
  applies to already-persisted event rows only.
- Review evidence: CI was green. Local validation covered
  `tests/test_live_event_query.py`, adjacent CLI dispatch tests, py_compile for
  touched files, `git diff --check`, and a silent-handling scan with no matches.
  No event producers, exchange calls, cache mutation, readiness gates, console
  routing, monitor writes, or trading behavior changed.

### PR #821: Event Type Registry Docs

### PR #822: Debug Profile Registry Docs

### PR #824: Startup Debug Profile Performance Report

### PR #827: Execution Terminal Outcome Report

- Branch: `codex/v8-execution-outcome-report`.
- Scope: read-only live performance report execution-timing projection and
  tests.
- Result: `passivbot tool live-performance-report` `execution_timing` now
  includes `terminal_outcome_counts`, derived only from fixed existing
  execution terminal event types. The counters report bounded labels such as
  `create.succeeded`, `cancel.ambiguous_terminal`, and
  `confirmation.satisfied` even when timing correlation is missing or unpaired.
  No order/action ids, raw order payloads, exchange data, account values, event
  producers, exchange calls, cache mutation, readiness gates, order/risk logic,
  console routing, monitor writes, or trading behavior changed.
- Review evidence: CI was green. Claude approved and Hermes approved with no
  findings. Local validation covered the full
  `tests/test_live_performance_report.py` suite, py_compile for touched files,
  `git diff --check`, and a silent-handling scan of touched report/test files.
- VPS5 evidence: deployed at `38f3a9e3` without bot restart because this is
  read-only report tooling. A 5-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state, and `account_critical_remote_calls.failed=0`.

### PR #826: Debug Profile Logging Guide

- Branch: `codex/v8-debug-profile-guide-docs`.
- Scope: docs-only logging-guide alignment.
- Result: `docs/ai/logging_guide.md` now points supported live-event debug
  profile names to `docs/ai/live_event_registry.md`, describes current
  profile-family behavior, and states that debug summaries must not copy raw
  exchange/account payloads, credentials, or unbounded row data. The final
  reviewed patch includes `forager` in the bounded profile-family list.
- Review evidence: CI was green. Claude approved the current rebased head.
  Hermes approved the docs patch after the `forager` reviewer nit was fixed and
  dry-ran the patch into current `origin/v8`. Local validation covered
  `git diff --check`. No runtime code, event producers, exchange calls, cache
  mutation, readiness gates, console routing, monitor writes, order/risk logic,
  or trading behavior changed.
- VPS5 evidence: deployed at `eec38e60` without bot restart because this is
  docs-only. A 5-minute smoke reported `ok=true`, `hard_failures=0`,
  `logs.hard_matches=0`, `matched_expected=5`, clean tracked repository state,
  and all hard failure sources at zero.

### PR #829: Account State Change Performance Report

- Branch: `codex/v8-live-performance-state-changes`.
- Scope: read-only live performance report account-state activity projection
  and tests.
- Result: `passivbot tool live-performance-report` now includes
  `account_state_changes`, derived only from existing `fill.ingested`,
  `position.changed`, and `balance.changed` events. The section summarizes
  event counts by bot and event type plus bounded status, reason, symbol,
  pside, side, and component counters. It deliberately ignores event `data`,
  so balances, equity, sizes, prices, PnL, fees, order ids, fill ids, raw
  payloads, and client-order ids are not surfaced.
- Review evidence: CI was green. Claude approved and Hermes approved with no
  findings. Local validation covered the full
  `tests/test_live_performance_report.py` suite, py_compile for touched files,
  `git diff --check`, and a silent-handling scan of touched report/test files.
  No event producers, exchange calls, cache mutation, readiness gates,
  order/risk logic, console routing, monitor writes, or trading behavior
  changed.
- VPS5 evidence: deployed at `fb2268af` without bot restart because this is
  read-only report tooling. A 5-minute smoke reported `ok=true`,
  `hard_failures=0`, `logs.hard_matches=0`, `matched_expected=5`, clean tracked
  repository state, and all hard failure sources at zero.

### Draft Slice: Order Wave Console Dedupe

- Branch: `codex/v8-dedupe-order-wave-console`.
- Scope: observability-only console routing cleanup for order-wave lifecycle
  summaries.
- Result: when the structured live-event console path is active, legacy stdlib
  `[order] wave complete` and `[order] wave settled` lines are suppressed so
  operators see the structured execution/confirmation summaries without
  duplicate legacy lines. If the structured console path is unavailable or
  disabled, legacy order-wave lines remain the fallback.
- Local validation: targeted order-wave console tests and live-event console
  formatter tests passed, plus `py_compile` for touched Python files.

### Draft Slice: Recovered Time-Sync Smoke Classification

- Branch: `codex/v8-smoke-recovered-time-sync`.
- Scope: read-only smoke-report classification for existing
  `cycle.degraded` and `exchange.time_sync` monitor events.
- Triggering evidence: after PR #989 was deployed and Kucoin reached READY,
  VPS5 smoke was temporarily hard-red because the 10-minute window still
  contained a first-cycle `InvalidNonce` `cycle.degraded` event. The following
  `exchange.time_sync` event succeeded, subsequent cycles continued, and the
  same smoke window became green once the recovered event aged out.
- Intended result: keep unrecovered timestamp/nonce cycle errors hard, but
  classify same-cycle successful `exchange.time_sync` recovery as a recovered
  problem event in detailed, summary, and brief smoke output. This changes only
  report classification; it does not add exchange calls, change live recovery,
  or alter trading behavior.
- Result: PR #990 was reviewed, merged to `v8`, and deployed to VPS5. After the
  transient Kucoin maintenance timeout aged out of the requested 10-minute
  window, VPS5 smoke was green on `bed4f285` with all five expected bots
  running and no hard problem/log/process failures.

### Draft Slice: Text Log Match Context

- Branch: `codex/v8-log-match-context`.
- Scope: read-only smoke-report text-log projection for existing log matches.
- Triggering evidence: the post-PR #990 VPS5 smoke briefly stayed hard-red
  because Kucoin logged a transient `maintain_hourly_cycle` exchange timeout
  with traceback fragments. The structured event stream already classified the
  config-refresh failure as a warning, but the hard text-log matches only showed
  `Traceback (most recent call last):` without the timestamped context line in
  the compact match entry.
- Intended result: attach the nearest timestamped log context line to matched
  unparseable traceback/error lines. This should make future smoke reports
  explain what subsystem emitted a hard text-log fragment without suppressing
  or down-classifying the hard match.
- Result: PR #991 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 together with PR #992 at `03f2fc10`.

### Draft Slice: EMA Readiness Reason Smoke Summary

- Branch: `codex/v8-smoke-ema-reason-summary`.
- Scope: read-only smoke-report projection for existing `ema.unavailable`
  events.
- Triggering evidence: current VPS5 smoke was green but still showed non-hard
  EMA-readiness attention. A focused `live-event-query` revealed useful
  structured detail already present in the events, including
  `cache_only_fetch_failed`, `never_fetched_cache_only`, and candidate error
  type groups. Operators should not need a second event-query just to identify
  the dominant EMA-readiness reason in concise smoke output.
- Intended result: aggregate latest EMA-readiness candidate reason counts,
  unavailable reason counts, and candidate error-type counts into the full,
  summary, and brief smoke-report projections. This changes only report output;
  it does not add event producers, exchange calls, readiness gates, EMA
  behavior, order logic, risk logic, or trading behavior.
- Expected validation: focused EMA-readiness smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #992 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `03f2fc10`. The first post-deploy smoke showed a
  transient GateIO positions `RequestTimeout` from an existing running bot; a
  follow-up after the event aged out reported `ok=true`, `hard_failures=0`,
  five expected bots matched, clean tracked repository state, zero
  account-critical failures, no hard log matches, and the new EMA-readiness
  reason maps visible in brief output.

### Draft Slice: Remote Call Failure Cause Smoke Summary

- Branch: `codex/v8-smoke-remote-call-failure-summary`.
- Scope: read-only smoke-report projection for existing `remote_call.*`
  health events.
- Triggering evidence: the post-PR #991/#992 VPS5 smoke briefly went hard-red
  from one GateIO `remote_call.failed` / `cycle.degraded` pair. The brief smoke
  showed remote-call failure counts, but identifying that the failing surface
  was `authoritative_positions` and the error type was `RequestTimeout`
  required a separate `live-event-query`.
- Intended result: aggregate failed remote-call reason codes, surfaces, logical
  call kinds, and error types into full, summary, and brief smoke-report
  projections. This changes only report output; it does not add event
  producers, exchange calls, readiness gates, order logic, risk logic, or
  trading behavior.
- Expected validation: focused remote-call smoke-report tests, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #993 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `32e80518`. The post-deploy 10-minute brief smoke
  reported `ok=true`, `hard_failures=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, `matched_expected=5`, clean tracked
  repository state, and non-hard attention from known EMA/HSL status events.

### Draft Slice: Staged Readiness Surface Smoke Summary

- Branch: `codex/v8-smoke-staged-surface-summary`.
- Scope: read-only smoke-report projection for existing staged-readiness
  `cycle.degraded` events.
- Triggering evidence: the post-PR #995 VPS5 smoke showed
  `staged_readiness.total=1` and `latest_missing_surface_total=1`, but the
  brief output did not name which surface was missing. A focused
  `live-event-query` showed KuCoin deferred Rust order calculation with
  `missing=["completed_candles"]` and
  `defer_reason=staged_planner_inputs_not_fresh`.
- Intended result: include bounded missing/invalid staged-readiness surface
  maps in full, summary, and brief smoke-report output. This changes only
  report output; it does not add event producers, exchange calls, readiness
  gates, staged execution behavior, order logic, risk logic, monitor writes,
  console routing, or trading behavior.
- Expected validation: focused staged-readiness smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #996 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `fae2b0b8`. The post-deploy 10-minute brief smoke
  reported `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked
  repository state, and the five configured live bots still running.

### Draft Slice: EMA-Readiness Symbol Samples In Smoke Summary

- Branch: `codex/v8-smoke-ema-symbol-samples`.
- Scope: read-only smoke-report projection for existing `ema.unavailable`
  monitor events.
- Triggering evidence: the post-PR #997 VPS5 smoke still showed EMA-readiness
  attention by reason, but identifying affected symbols required a separate
  `live-event-query`. The underlying `ema.unavailable` events already included
  bounded `candidate_unavailable_groups` and `unavailable_reasons` symbol
  samples.
- Intended result: include bounded symbol samples by EMA unavailable reason in
  full, summary, and brief smoke-report output. The projection must omit
  `example_error` prose and payload extras from the concise fields. This
  changes only report output; it does not add event producers, exchange calls,
  readiness gates, EMA behavior, order logic, risk logic, monitor writes,
  console routing, or trading behavior.
- Expected validation: focused EMA-readiness smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #998 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `d5c239e9`. The post-deploy smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state, and
  the five configured live bots still running. The new bounded EMA symbol
  fields were visible in VPS5 brief smoke output.

### Draft Slice: Remote-Call Latency Samples In Brief Smoke

- Branch: `codex/v8-smoke-remote-latency-brief`.
- Scope: read-only brief smoke-report projection over existing
  `remote_call_health.groups`.
- Triggering evidence: post-PR #998 VPS5 brief smoke showed healthy remote-call
  counts but hid latency details, while summary output showed slow surfaces
  such as account-critical open-orders calls above 16s and candle remote fetch
  groups above 30s. Operators had to run the larger summary to see which
  bot/surface was slow.
- Intended result: add a bounded `slowest` list to brief `remote_calls` and
  `account_critical_remote_calls`, containing bot, kind/surface, count,
  failed/throttled counts when nonzero, max/p95/latest elapsed milliseconds,
  and latest symbol when present. This changes only report output; it does not
  add event producers, exchange calls, remote-call behavior, readiness gates,
  order logic, risk logic, monitor writes, console routing, or trading
  behavior.
- Expected validation: focused brief smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #999 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `8e6aaf4b`. The post-deploy smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state, and
  the five configured live bots still running. The new brief `slowest` rows
  exposed slow surfaces such as OKX candle fetches near 35s and account-critical
  open-orders/balance calls above 10s.

### Draft Slice: Staged Readiness Reason And Timing Smoke Summary

- Branch: `codex/v8-smoke-staged-degraded-timing`.
- Scope: read-only smoke-report projection over existing `cycle.degraded` and
  `planning.unavailable` staged-readiness events.
- Triggering evidence: post-PR #999 VPS5 brief smoke showed
  `staged_readiness.total=1` and `latest_missing_surface_total=0`, while a
  focused event query showed KuCoin degraded because Rust order calculation was
  deferred with `defer_reason=staged_planner_inputs_not_fresh`,
  `reason_code=staged_execution_precondition`, and long `timings_ms` such as
  `market_state=303339`. The brief smoke did not surface those reason/timing
  fields, so the operator still needed a separate event query to understand why
  planning degraded.
- Intended result: include bounded reason-code counts, latest defer-reason
  counts, latest context counts, and max latest timing fields in full, summary,
  and brief staged-readiness smoke output. This changes only report output; it
  does not add event producers, exchange calls, readiness gates, staged
  execution behavior, order logic, risk logic, monitor writes, console routing,
  or trading behavior.
- Expected validation: focused staged-readiness smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #1000 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `1a73776f`. The post-deploy smoke reported `ok=true`,
  `hard_failures=0`, `matched_expected=5`, clean tracked repository state, and
  the five configured live bots still running. No staged-readiness events
  appeared in the post-deploy smoke window, so the new fields were not
  exercised live yet; the test fixture covers both `cycle.degraded` and
  `planning.unavailable` shapes.

### Draft Slice: Brief Log Match Samples

- Branch: `codex/v8-smoke-brief-log-match-samples`.
- Scope: read-only brief smoke-report projection over existing sanitized text
  log matches.
- Triggering evidence: after PR #1003 deployment, the short post-deploy smoke
  was green, but a wider window reported `log_hard_matches=2` from an older
  KuCoin NEAR HSL RED event. The brief output showed only counts, so the
  operator had to rerun summary sections to identify the log path, line, and
  text.
- Intended result: keep the full `matches` list out of `--brief`, but include
  bounded `hard_samples` and `attention_samples` under `logs`, sourced from the
  same redacted match objects already emitted by the summary report. This
  changes only report output; it does not add log scanning, event producers,
  exchange calls, HSL behavior, order/risk logic, monitor writes, console
  routing, or trading behavior.
- Expected validation: focused log-sample smoke-report tests, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.
- Result: PR #1004 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `291a8711`. A short post-deploy smoke reported
  `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked repository
  state, and the five configured live bots still running. A wider logs-only
  smoke verified the new bounded `hard_samples`/`attention_samples` fields and
  showed basename-only log paths for recent HSL RED log lines. The same smoke
  also showed several recent HSL RED/cooldown/mode events in structured
  `risk_events`, but brief output still required reading specialized HSL
  status fields or a summary rerun to see the latest risk event rows.

### Draft Slice: Brief Risk Event Samples

- Branch: `codex/v8-smoke-brief-risk-event-samples`.
- Scope: read-only brief smoke-report projection over existing summarized
  `risk_events.groups`.
- Triggering evidence: after PR #1004 deployment, VPS5 brief smoke showed
  HSL cooldown and RED context through `risk_events.hsl_status`, but recent
  structured rows such as `hsl.red_finalized_without_order`,
  `risk.mode_changed`, and `unstuck.status` were only visible in the larger
  summary `risk_events.groups` output.
- Intended result: add bounded `risk_events.latest_groups` to brief output with
  safe identifiers only: bot, event type, reason code, status, level,
  symbol/pside, component, count, and latest timestamp. Do not include
  `latest_data` or raw drawdown/balance/order payload fields. This changes only
  report output; it does not add event producers, exchange calls, HSL behavior,
  order/risk logic, monitor writes, console routing, or trading behavior.
- Expected validation: focused risk-event smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.

### Draft Slice: Event Query Rotation Warning

- Branch: `codex/v8-event-query-rotation-warning`.
- Scope: read-only `passivbot tool live-event-query` report output and docs.
- Triggering evidence: a VPS5 Kucoin ASTER HSL RED/cooldown incident was
  visible in text logs and recoverable from rotated monitor event segments with
  `--include-rotated`, but a default filtered `live-event-query` scan over
  `current.ndjson` returned zero matches after event rotation. The same query
  also showed scan-order trace bounds when current segments were read before
  older rotated files.
- Intended result: keep current-only directory scans as the default for speed,
  but emit a warning issue when a filtered query skips rotated event segments.
  Also report trace-summary `first_ts`/`last_ts` as chronological min/max
  timestamps rather than scan-order first/last. No event producers, monitor
  writes, exchange calls, or trading behavior change.
- Expected validation: focused `tests/test_live_event_query.py`, `py_compile`,
  `git diff --check`, added-line silent-handling scan, and a read-only VPS5
  rotated query proving the HSL incident is recoverable with `--include-rotated`.
- Result: PR #1022 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `c090fd5b`. A short post-deploy smoke reported
  `ok=true`, `hard_failures=0`, `matched_expected=5`, clean tracked repository
  state, and five configured live bots still running. A read-only Kucoin ASTER
  current-only query now emits `current_only_rotated_segments_skipped`, while
  `--include-rotated` recovers the HSL RED/cooldown incident events.

### Draft Slice: Dropped Unparsed Log Samples

- Branch: `codex/v8-smoke-dropped-unparsed-samples`.
- Scope: read-only `passivbot tool live-smoke-report` summary/brief projection
  over existing log-window dropped-unparsed counters.
- Triggering evidence: after PR #1022 deployment, VPS5 brief smoke was green
  but reported `dropped_unparsed_attention_matches=1` and
  `dropped_unparsed_hard_matches=1` with `--log-window-unparsed-policy drop`.
  The brief output showed the counts but not the dropped line, so an operator
  could not tell whether the dropped hard-looking signal was a stale traceback
  fragment or a fresh line cut off by the log tail.
- Intended result: keep the existing verdict policy and drop behavior, but
  retain bounded redacted samples for contextless attention/hard-looking lines
  dropped by the unparsed-line policy. Project them into summary and brief
  output with basename-only paths in brief, matching existing log sample
  conventions. This changes only report output; it does not add log scanning,
  event producers, exchange calls, monitor writes, console routing, or trading
  behavior.
- Expected validation: focused dropped-unparsed smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, added-line
  silent-handling scan, and a read-only VPS5 smoke showing dropped samples when
  the condition is present.
- Result: PR #1023 was reviewed by Hermes and Claude, merged to `v8`, and
  deployed to VPS5 at `8aab137c`. A short post-deploy smoke reported
  `ok=true`, `hard_failures=0`, clean tracked repository state, and five
  configured live bots still running. The current VPS5 log window did not
  contain dropped unparsed hard/attention matches after deployment, so the new
  sample fields were absent as expected when the condition is not present.
  The same smoke showed `unstuck.status` in brief `risk_events.latest_groups`,
  but without compact `latest_data`; a follow-up event query showed the
  underlying event had useful allowlisted state such as `changed`,
  `status_counts`, and `over_budget_sides`.

### Draft Slice: Brief Risk Latest Data

- Branch: `codex/v8-smoke-risk-brief-latest-data`.
- Scope: read-only `passivbot tool live-smoke-report --brief` projection over
  existing summarized `risk_events.groups` and `risk_events.attention_groups`.
- Triggering evidence: after PR #1023 deployment, VPS5 brief smoke exposed an
  `unstuck.status` latest group for `hyperliquid/hyperliquid_tradfi`, but the
  brief row did not explain whether the state changed, which status counts were
  present, or whether any side was over budget. A focused `live-event-query`
  showed the existing monitor event already contained safe compact fields:
  `changed=true`, `status_counts`, and `over_budget_sides`, alongside raw
  per-side allowance details that should remain out of brief smoke output.
- Intended result: include only allowlisted `latest_data` keys in brief risk
  groups, covering HSL mode/status/finalization context and unstuck status
  summaries. Continue excluding raw balances, drawdown internals, price-distance
  details, nested per-side allowance maps, and arbitrary event payload keys.
  This changes only report output; it does not add event producers, exchange
  calls, HSL behavior, order/risk logic, monitor writes, console routing, or
  trading behavior.
- Expected validation: focused risk-event smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`,
  added-line silent-handling scan, and a read-only VPS5 smoke after merge
  proving the compact fields appear when the condition is present.
- Result: PR #1024 was reviewed by Claude and Hermes, merged to `v8`, and
  deployed to VPS5 at `59c36ada` without restarting bots. A short post-deploy
  smoke reported `ok=true`, `hard_failures=0`, clean tracked repository state,
  and five configured bots still running. The deployed brief `risk_events`
  output now includes allowlisted `latest_data` for HSL cooldown and green
  status rows, proving the compact state projection is active while keeping raw
  drawdown/balance/allowance details out of brief output.

### Draft Slice: Forager Debug Profile

- Branch: `codex/v8-forager-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment for existing
  `forager.selection` and `forager.feature_unavailable` events.
- Triggering evidence: `forager` is already part of the documented
  `logging.live_event_debug_profiles` / `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES`
  surface, but the existing forager event emitters did not add any
  profile-specific debug shape when that profile was enabled.
- Intended result: when `forager` debug is enabled, add bounded count and
  key-shape metadata to existing forager events: candidate/eligible/selected
  counts, unavailable sample counts, top-score key shape, slot state, and
  related decision counters. Do not add raw score values beyond the existing
  default bounded top-score sample, do not change default forager events, and do
  not change console output, event routing, selection behavior, exchange calls,
  order/risk logic, monitor writes, or trading behavior.
- Expected validation: focused forager debug-profile monitor test,
  live-event debug-profile normalization test, broader live-event/monitor suite
  if review asks for it, `py_compile`, `git diff --check`, and the standard
  added-line silent-handling scan.
- Result: PR #1025 was reviewed by Claude and Hermes, merged to `v8`, and
  deployed to VPS5 at `7f8a1942` without restarting bots. A short post-deploy
  smoke reported `ok=true`, `hard_failures=0`, clean tracked repository state,
  five configured bots still running, no event-pipeline drops/sink errors, and
  only known non-hard HSL cooldown/status attention.

### Draft Slice: State Debug Profile

- Branch: `codex/v8-state-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment for existing
  `state.refresh_timing` and `state.refresh_progress` events.
- Triggering evidence: `state` is part of the documented
  `logging.live_event_debug_profiles` / `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES`
  surface, but only startup events and live performance reports exposed profile
  state. Existing state-refresh events did not add profile-specific debug shape
  when the profile was enabled.
- Intended result: when `state` debug is enabled, add bounded plan/pending
  counts, surface key lists, slowest refreshed surface, and timing scalar
  summaries to existing state refresh timing/progress events. Do not add raw
  account payloads, exchange responses, credentials, event routes, console
  output, refresh behavior, exchange calls, order/risk logic, monitor writes,
  or trading behavior.
- Expected validation: focused state debug-profile monitor test,
  live-event debug-profile normalization test, broader live-event/monitor suite
  if review asks for it, `py_compile`, `git diff --check`, and the standard
  added-line silent-handling scan.
- Result: PR #1026 was reviewed by Claude and Hermes, merged to `v8`, and
  deployed to VPS5 at `2de5a6af` without restarting bots. A short post-deploy
  smoke reported `ok=true`, `hard_failures=0`, clean tracked repository state,
  five configured bots still running, no event-pipeline drops/sink errors, and
  only known non-hard HSL cooldown/status attention.

### Draft Slice: Startup Debug Profile

- Branch: `codex/v8-startup-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment for existing
  `bot.startup_timing` events.
- Triggering evidence: `startup` is part of the documented
  `logging.live_event_debug_profiles` / `PASSIVBOT_LIVE_EVENT_DEBUG_PROFILES`
  surface. Startup lifecycle events already expose enabled profiles and
  performance reports summarize them, but the startup timing events themselves
  did not add profile-specific debug shape when the profile was enabled.
- Intended result: when `startup` debug is enabled, add bounded phase, timing,
  and details-shape metadata to existing startup timing events. Do not duplicate
  raw startup details into the debug block, and do not change default startup
  timing payloads, console output, event routing, startup behavior, exchange
  calls, order/risk logic, monitor writes, or trading behavior.
- Expected validation: focused startup debug-profile monitor test,
  live-event debug-profile normalization test, broader live-event/monitor suite
  if review asks for it, `py_compile`, `git diff --check`, and the standard
  added-line silent-handling scan.
- Result: PR #1027 was reviewed by Claude and Hermes, merged to `v8`, and
  deployed to VPS5 at `9c555384` without restarting bots. A short post-deploy
  smoke reported `ok=true`, `hard_failures=0`, clean tracked repository state,
  five configured bots still running, no event-pipeline drops/sink errors, and
  only known non-hard HSL cooldown/status attention.

### Draft Slice: Cache Debug Profile

- Branch: `codex/v8-cache-debug-profile`.
- Scope: Phase 5/6 opt-in structured debug enrichment for existing
  `cache.load.completed`, `cache.flush.completed`, and `cache.warmup_decision`
  events.
- Triggering evidence: restart/warm-cache startup speed has been a repeated
  live-ops concern, and cache events plus performance-report cache aggregation
  already exist. Unlike `startup`, `state`, `forager`, and other debug-profile
  event families, cache events did not yet support profile-specific bounded
  shape metadata when operators enable a focused profile during restart
  investigations.
- Intended result: when `cache` debug is enabled, add bounded key/count/source
  metadata to existing cache load, flush, and warmup-decision events. Keep raw
  candle rows, file paths, cache contents, event routes, console output, cache
  behavior, startup behavior, exchange calls, monitor writes, order/risk logic,
  and trading behavior unchanged.
- Expected validation: focused cache debug-profile monitor test, live-event
  debug-profile normalization and registry-doc tests, `py_compile`,
  `git diff --check`, and the standard added-line silent-handling scan.
- Result: PR #1042 was reviewed by Claude and Hermes, merged to `v8`, and
  deployed to VPS5 at `06f04070` without restarting bots. A short post-deploy
  smoke reported `ok=true`, `hard_failures=0`, clean tracked repository state,
  five configured bots still running, no event-pipeline drops/sink errors, and
  only known non-hard ZEC HSL cooldown attention.

### Draft Slice: Cache Health Smoke Summary

- Branch: `codex/v8-smoke-cache-health`.
- Scope: read-only smoke-report projection over existing `cache.load.completed`,
  `cache.flush.completed`, and `cache.warmup_decision` events.
- Triggering evidence: `live-performance-report` already summarizes cache
  warmup/load/flush behavior, and PR #1042 made cache events debug-profile
  enriched on demand, but repeated VPS smoke loops still did not expose
  warm-cache reuse, cold-path counts, cache load rows, or cache flush rows
  directly.
- Intended result: add `cache_health` to full/summary smoke output and `cache`
  to brief/section aliases, exposing bounded cache counters and latest compact
  event data. Do not expose raw cache paths, raw cache payloads, candle rows, or
  arbitrary payload keys. Do not add event producers, exchange calls, cache
  behavior, startup behavior, console routing, monitor writes, order/risk logic,
  or trading behavior.
- Expected validation: focused cache smoke-report test, full
  `tests/test_live_smoke_report.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.

### Draft Slice: Event Query Debug Profile Filter

- Branch: `codex/v8-event-query-debug-profile`.
- Scope: read-only operator tooling for the completed live-event debug-profile
  surface.
- Triggering evidence: the documented debug-profile event surface is now
  complete, but querying those events requires remembering the generic
  `--data-eq debug_profile=...` predicate. A first-class filter reduces
  operator friction during incident reconstruction.
- Intended result: add `passivbot tool live-event-query --debug-profile` as a
  shortcut for matching `event.data.debug_profile`, with query metadata
  reporting the selected profile names. Keep existing `--data-eq` behavior
  unchanged. Do not add event producers, exchange calls, monitor writes, console
  routing, startup behavior, order/risk logic, or trading behavior.
- Expected validation: focused API and CLI live-event-query tests, full
  `tests/test_live_event_query.py`, `py_compile`, `git diff --check`, and the
  standard added-line silent-handling scan.

### Draft Slice: Incident Bundle Debug Profile Filter

- Branch: `codex/v8-incident-debug-profile-filter`.
- Scope: read-only incident-bundle tooling.
- Triggering evidence: PR #1028 made `live-event-query --debug-profile` a
  first-class filter, but incident bundles still require the generic
  `--data-eq debug_profile=...` predicate to package the same focused evidence.
- Intended result: add `passivbot tool live-incident-bundle --debug-profile`
  and pass it through event reports, problem-event reports, time-window reports,
  and manifest filter metadata. Keep existing `--data-eq` behavior unchanged.
  Do not add event producers, exchange calls, monitor writes, console routing,
  startup behavior, order/risk logic, or trading behavior.
- Expected validation: focused incident-bundle CLI/API test, full
  `tests/test_live_incident_bundle.py`, `py_compile`, `git diff --check`, and
  the standard added-line silent-handling scan.

### Draft Slice: Performance Report Debug Profile Filter

- Branch: `codex/v8-performance-debug-profile-filter`.
- Scope: read-only performance-report tooling.
- Triggering evidence: `live-event-query` and `live-incident-bundle` can now
  scope reports to one debug profile, but `live-performance-report` still
  aggregates all events for timing/readiness summaries even when an operator is
  investigating one enriched profile.
- Intended result: add `passivbot tool live-performance-report --debug-profile`
  and filter events at the same scan boundary as bot/exchange/user filters,
  recording the selected profiles and skipped-event count in report metadata.
  Do not add event producers, exchange calls, monitor writes, console routing,
  startup behavior, order/risk logic, or trading behavior.
- Expected validation: focused performance-report CLI filter test, full
  `tests/test_live_performance_report.py`, `py_compile`, `git diff --check`,
  and the standard added-line silent-handling scan.

### Draft Slice: Incident Bundle Performance Report Artifact

### Draft Slice: Restart Smoke Performance Bundle Evidence

## Current Next Steps

### Draft Slice: Health Summary CPU Percent

### Draft Slice: Live psutil Requirement

### Draft Slice: Resource Pressure Smoke Projection

- Branch: `codex/v8-smoke-resource-pressure`.
- Scope: read-only smoke-report projection over existing periodic
  `health.summary` resource-pressure fields.
- Triggering evidence: PRs #1148 and #1149 made CPU, memory, RSS, open-FD, and
  load evidence available in deployed `health.summary` events and
  `live-performance-report --section resource_pressure`, but standard repeated
  smoke loops still did not expose those process-pressure signals directly.
- Intended result: add `resource_pressure` to full and summary smoke reports,
  plus a compact `resource_pressure` brief projection and `resources` section
  alias, using only existing `health.summary` events. Keep output bounded to
  per-bot latest values and aggregate latest max/total counters. Do not add
  event producers, exchange calls, monitor writes, console routing, restart
  behavior, order/risk logic, or trading behavior.
- Expected validation: focused smoke-report resource-pressure tests,
  `py_compile`, `git diff --check`, and the standard added-line
  silent-handling scan.
- Result: PR #1150 was reviewed by Hermes, Claude Opus 4.8, and Grok 4.5 on
  current head `c2e68817`, merged to `v8` as `58308f32`, and deployed to VPS5
  without restarting running bots because the slice was read-only report
  projection over existing events. VPS5 was pulled to `58308f32` with
  `--autostash`, preserving a pre-existing local rustfmt-only tracked diff in
  `passivbot-rust/src/equity_hard_stop_loss.rs`. A settled 2-minute smoke with
  dropped-unparsed log policy reported `ok=true`, `hard_failures=0`,
  `matched_expected=5`, `missing_expected_count=0`, `remote_calls.failed=0`,
  `account_critical_remote_calls.failed=0`, and event-pipeline dropped/sink
  errors at zero; non-hard attention remained EMA readiness, one stale dropped
  traceback sample, and `unstuck.status`. The brief smoke exposed the new
  `resource_pressure` projection with one reporting bot, `cpu_percent` max
  `15.1`, RSS total `61177856`, and open FDs total `16`. A focused 30-minute
  `resource_pressure` section showed four reporting bots, `cpu_percent` max
  `21.5`, RSS total `363417600`, and open FDs total `61`; the section command
  exited red only because the report also carried unrelated hard-event and
  dirty-repository markers.

### Draft Slice: Health Summary Scheduling Lag

- Branch: `codex/v8-health-loop-lag`.
- Scope: observability producer plus existing smoke/performance report
  projections for periodic `health.summary` resource-pressure events.
- Triggering evidence: resource-pressure reports now show CPU/load, memory,
  RSS, open FDs, event queue depth, dropped event counters, and sink errors, but
  the performance checklist still lacked loop-lag-style heartbeat evidence.
  Existing `last_loop_duration_ms` measures the previous cycle body and does
  not prove whether periodic health summaries themselves are being delayed.
- Intended result: add non-negative `health_summary_lag_ms` to
  `health.summary` after the first heartbeat, measuring elapsed time beyond the
  configured health-summary interval. Project it through
  `live-performance-report` resource-pressure stats and `live-smoke-report`
  resource-pressure full/summary/brief output. Do not add exchange calls,
  monitor files beyond the existing periodic health event, order/risk logic,
  restart behavior, or trading behavior.
- Expected validation: focused health-summary payload/scheduler tests, focused
  smoke/performance resource-pressure tests, `py_compile`, `git diff --check`,
  and the standard added-line silent-handling scan.

### Draft Slice: Resource Pressure Sample Age

- Branch: `codex/v8-resource-pressure-event-age`.
- Scope: read-only performance-report projection over existing
  `health.summary` resource-pressure events.
- Triggering evidence: PRs #1148 through #1153 made resource-pressure values
  available and visible in performance/smoke reports, but
  `live-performance-report --section resource_pressure` exposed only each bot's
  `latest_ts`, not a directly comparable age. During bounded post-deploy smoke
  windows, operators need to distinguish recent pressure samples from stale or
  absent samples without manually subtracting timestamps.
- Intended result: add `latest_event_age_ms` to each performance-report
  `resource_pressure` group, derived from the report timestamp and the group's
  latest `health.summary` event timestamp. Keep existing field statistics,
  event parsing, monitor writes, console output, smoke-report output, exchange
  calls, restart behavior, order/risk logic, and trading behavior unchanged.
- Expected validation: focused resource-pressure performance-report tests, full
  `tests/test_live_performance_report.py`, `py_compile`, `git diff --check`,
  and the standard added-line silent-handling scan.
- Result: PR #1154 was reviewed by Hermes, Claude Opus 4.8, Grok 4.5, and
  Codex on current head, merged to `v8` as `141e88db`, and deployed to VPS5
  without restarting bots because the slice was read-only report projection.
  A bounded smoke reported process/repository health green, and a focused
  30-minute `live-performance-report --section resource_pressure` check proved
  one Hyperliquid group with `latest_event_age_ms=624447`.

### Draft Slice: Resource Pressure Sample Age Aggregate

- Branch: `codex/v8-resource-pressure-age-aggregate`.
- Scope: read-only performance-report projection over existing
  `health.summary` resource-pressure events.
- Triggering evidence: PR #1154 made per-bot sample age visible in
  `live-performance-report`, but operators still need a compact top-level
  maximum age and reporting-bot count to compare the freshness of the whole
  resource-pressure section without scanning all groups.
- Intended result: add aggregate `latest_event_age_ms_max` and
  `latest_event_age_reporting_bots` fields to `live-performance-report`
  `resource_pressure`, derived from existing per-group ages. Keep existing
  field statistics, event parsing, monitor writes, console output, smoke-report
  output, exchange calls, restart behavior, order/risk logic, and trading
  behavior unchanged.
- Expected validation: focused resource-pressure performance-report tests, full
  `tests/test_live_performance_report.py`, `py_compile`, `git diff --check`,
  and the standard added-line silent-handling scan.

### Active Slice: Current Live-Process Pressure In Smoke Reports

- Branch: `codex/v8-smoke-current-process-pressure`.
- Scope: read-only `live-smoke-report` process aggregation and projection over
  fields already returned by the local `ps` scan.
- Triggering evidence: settled PR #1178 smoke had only one event-derived
  `resource_pressure` bot even though direct process probes showed four
  replay-heavy bots in `D` state with high RSS and sustained swap/I/O pressure.
- Intended result: expose bounded process state counts,
  uninterruptible-sleep count, and CPU, memory, and RSS
  totals/maxima/reporting counts in full, summary, and brief process output.
  Missing metrics remain null; the fields do not change smoke verdicts.
- Non-goals: no live event producer, monitor write, process signal, exchange
  call, restart policy, threshold, trading/risk/order behavior, Rust, or
  backtest change.

### Deployed Slices: Compact Replay, Scorecard, And Fill Index

- PR #1180 merged the compact cold replay payload after exact-head Hermes and
  Grok approval plus green CI. VPS5 was restarted on the merge; all five bots
  reached a settled hard-green smoke, and four coin-HSL bots emitted compact
  protective-ready evidence. Full replay still took `453.980s` to `1728.585s`,
  proving memory pressure improved without removing pair-minute complexity.
- PRs #1181 and #1182 added bounded replay scorecard fields and recovered
  protective elapsed aggregates from completion events. Both were read-only
  report slices and required no bot restart.
- PR #1183 grouped cold replay fills once by `(pside, symbol)` and reused the
  stable index for replay-contract inference and position-size reconstruction.
  Hermes and Grok approved exact head `73a0b935`, CI passed, and it merged as
  `b29d5ca5`. VPS5 pulled the clean merge and restarted all five bots while
  preserving local artifacts and the unrelated `misc` tmux session.
- Immediate and fresh settled smoke windows were hard-green; the recovery
  window recorded `380/380` successful remote and `42/42` account-critical
  calls. The indexed history-loaded stage took only `0.103s` to `0.213s` for
  `1406` to `5359` fills. KuCoin and Binance full replay later completed in
  `601.246s` and `914.691s`; OKX and GateIO remained active without replay
  failures. The remaining cost is the pair-minute metric loop.

### Active Slice: Missing Resource-Pressure Latest Value

### Deployed Slice: Missing Resource-Pressure Latest Value

### Active Slice: Configured-Market Compatibility Events

- Branch: `codex/v8-market-compatibility-events` from current remote `v8`
  `60357a00`; VPS5 remains on deployed `b9748247` until this producer slice is
  approved for restart.
- Triggering evidence: current VPS5 text logs repeatedly report unsupported
  approved coins for Binance (`CRO,MNT`) and OKX (`KAS,MNT,XMR`), but no
  structured event records which configured markets were discarded or why.
- Scope: emit one bounded off-console/off-text
  `config.market_compatibility` event from the existing configured-coin skip
  path, preserving text-log dedupe and all filtering behavior. Include safe
  per-side list/count/sample context and classify stock-perp-looking skipped
  symbols without changing stock-perp policy.
- Non-goals: no exchange calls, eligible-market calculation, stock-perp
  margin/account policy, Hyperliquid fatal startup enforcement, isolated-only
  entry filtering, smoke verdict, HSL/risk/order behavior, Rust, backtest, or
  optimizer change. Existing generic query/problem-event tooling consumes the
  event; this slice does not add a bespoke report aggregator.
- Focused live-event, coin-list, smoke, query, incident, registry-doc,
  compilation, diff, and silent-handling checks pass. Luna's independent
  preflight found and then cleared per-side query provenance, durable symbol
  bounds/redaction, retryable enqueue dedupe, changelog, and equal-side query
  gaps across two delta rounds.

### Deployed Slice: Configured-Market Compatibility Events

- PR #1187 was approved by Hermes, Grok 4.5, and independent Codex on exact
  head `74766c7cb`; CI passed and it merged to `v8` as `b99d1b05`.
- VPS5 fast-forwarded cleanly and restarted only the five supervised bots.
  The first graceful signal stopped OKX and Hyperliquid; a second exact-pane
  Ctrl+C stopped Binance, KuCoin, and GateIO. The unrelated `misc:0.0` pane
  remained PID `434835`.
- A bounded current-segment query returned four exact
  `config.market_compatibility` events: long and short records for Binance
  `CRO,MNT` and OKX `KAS,MNT,XMR`, with bounded payloads, stable generic reason,
  and non-hard degraded status.
- The immediate smoke caught one real KuCoin authoritative-state timeout. The
  settled two-minute smoke was hard-green with all five expected processes
  matched, `370/370` remote calls and `26/26` account-critical calls
  successful, `R=4,S=1`, no uninterruptible sleep, and tracked repository
  status clean.

### Active Slice: HIP-3 Fatal Startup Compatibility Event

- Branch: `codex/v8-stock-perp-compatibility-events` from deployed
  `b99d1b05`.
- Scope: emit and boundedly flush one off-console/off-text
  `config.market_compatibility` event before Hyperliquid's existing non-unified
  HIP-3 startup gate raises. Retain only safe bounded counts/samples for
  approved symbols, positions, open-order symbols, isolated-only capability,
  live isolated margin state, and account abstraction.
- Non-goals: no fatal decision/message, market/margin/account policy, exchange
  call, generic isolated-only entry filtering, configured-coin filtering,
  smoke verdict, HSL/risk/order behavior, Rust, backtest, or optimizer change.
  Emission and flush remain best-effort and must never suppress or replace the
  existing `FatalBotException`.
- Focused validation passes 184 Hyperliquid fatal-state, event-bus, smoke,
  enqueue/flush-failure, and registry-doc tests plus Python compilation and
  `git diff --check`. Independent Luna preflight is green after verifying the
  producer adds no metadata/policy/exchange call before the existing fatal
  path; approved-only symbols do not recompute isolated-margin policy.

### Deployed Slice: HIP-3 Fatal Startup Compatibility Event

- PR #1188 was approved by Hermes and Grok 4.5 on exact head `363eca852`; CI
  passed and it merged to `v8` as `bd169747`.
- VPS5 fast-forwarded cleanly and gracefully restarted only the exact
  Hyperliquid pane. Bot PID `842779` was replaced by `844272`; the other four
  bot PIDs and unrelated `misc:0.0` PID `434835` remained unchanged.
- The unified account reached startup-ready in `48.00s` and full-warmup-ready
  in `74.77s`. A bounded focused query matched zero
  `config_hip3_account_mode_unsupported` events, as expected for healthy
  unified-account startup.
- The immediate smoke was hard-green with `662/662` remote and `71/71`
  account-critical calls successful. The final settled two-minute smoke was
  hard-green with all five expected processes matched, `327/327` remote and
  `45/45` account-critical calls successful, `R=4,S=1`, no uninterruptible
  sleep, no hard/log failures, and tracked repository status clean.

### Active Slice: Isolated-Only Entry-Filter Compatibility Event

- Branch: `codex/v8-isolated-market-compatibility-events` from deployed
  `bd169747`.
- Scope: emit one bounded per-side `config.market_compatibility` event when the
  existing generic CCXT filter blocks isolated-only symbols from new entries
  under cross-margin preference. Use separate retryable event dedupe while
  preserving the current once-per-process warning.
- Non-goals: no margin policy/capability, metadata/exchange call, approved-list
  result, existing-position/order handling, warning, smoke verdict,
  HSL/risk/order behavior, Rust, backtest, or optimizer change.
- Focused margin-filter, CCXT contract-helper, event-bus, smoke, and registry
  validation passes 213 tests plus Python compilation and `git diff --check`.
  Luna's first preflight found that the manually constructed CCXT contract bot
  lacked the new diagnostics dedupe set; an explicit initializer and direct
  filter regression resolve the finding, and delta review found no code
  regression.

### Active Slice: Startup Readiness SLA Semantics

### Deployed Slice: Startup Readiness SLA Semantics

- PR #1192 was approved by Hermes and Grok 4.5 on exact head `9eb8e4a96`; CI
  passed and it merged to `v8` as `3b4b043eb`.
- VPS5 fast-forwarded cleanly and sequentially restarted only the five exact
  supervised panes. Bot PIDs `842617/842655/842687/842721/842757` became
  `850148/850296/850370/850436/850495`; unrelated `misc:0.0` stayed PID
  `434835`.
- Live performance evidence accepted all five account and execution-loop
  scopes, four held-position protective scopes, three first-market-state
  scopes, and a later completed background-candle scope with canonical impact
  labels. The best-effort `active-candle` phase remained timing-only.
- The immediate smoke caught three real KuCoin balance timeouts. After they
  aged out, the settled two-minute smoke was hard-green with `380/380` remote
  and `16/16` account-critical calls successful, all five expected bots
  matched, no event-pipeline drops or sink errors, and a clean tracked repo.
  Two cache/replay-time `D` samples cleared to `R/S` on the quiet follow-up.

### Active Slice: Performance Startup Lifecycle Ordering

- PR #1193, `Keep startup reports on latest lifecycle`.
- Branch: `codex/v8-performance-startup-lifecycle` from deployed
  `3b4b043eb66e8d0b42d792a5b94686b409901220`.
- Triggering evidence: the bounded per-bot file selector reads current first,
  then the selected rotated segment. The performance startup accumulator reset
  inline, allowing older lifecycle records encountered later to overwrite the
  current per-bot snapshot.
- Scope: use event ordering for current per-bot startup state while preserving
  historical aggregate phase/readiness distributions; cover the production
  `include_rotated` plus `max_event_files_per_bot=2` path.
- Non-goals: no producer, event schema, startup/readiness decision, exchange
  call, HSL/risk/order behavior, process control, smoke verdict, Rust,
  backtest, optimizer, or trading change. Expected VPS action is pull plus a
  bounded rotated report and settled smoke, with no bot restart.

### Deployed Slice: Performance Startup Lifecycle Ordering

- PR #1193 was approved by Hermes and Grok 4.5 on exact head `955fe215d`; CI
  passed and it merged to `v8` as `60f1f042d`.
- VPS5 fast-forwarded without bot signals or restarts. The five bot PIDs
  `850148/850296/850370/850436/850495` and unrelated `misc:0.0` PID `434835`
  remained unchanged.
- The exact bounded current-plus-rotated startup-readiness report returned
  `ok=true`, scanned 12 files with zero errors/warnings, retained all five
  current lifecycle snapshots, and preserved historical aggregate timing
  (`account` phase count six versus five current bots).
- The settled smoke was hard-green with `322/322` remote and `57/57`
  account-critical calls successful, all five bots matched, no event-pipeline
  drops/sink errors, and a clean tracked repository. Three transient `D`
  process samples during report I/O cleared; all five bots sampled `R` after a
  quiet interval.

### Active Slice: Startup Action Milestones

- PR #1194, `Report startup action milestones`.
- Branch: `codex/v8-startup-action-milestones` from deployed `60f1f042d`.
- Scope: add a bounded read-only `startup_milestones` performance-report
  section deriving the current lifecycle's first cycle, first Rust call, and
  first exchange-write submission from existing structured events. Absence in
  selected files remains explicitly unknown, and elapsed values require valid
  lifecycle and event timestamps.
- Contract boundary: `execution.create_sent` / `execution.cancel_sent` occur
  before connector invocation, so the report says `submitted`, not actual or
  successful exchange write. A protective-only write does not establish
  fresh-entry eligibility. That eligibility needs a separate producer contract
  after all reconciler/executor filters.
- Non-goals: no event producer, exchange call, startup/readiness decision,
  fresh-entry gate, process control, HSL/risk/order behavior, Rust, backtest,
  optimizer, or smoke-verdict change. Expected VPS action is pull plus an exact
  bounded rotated report and settled smoke, with no bot restart.

### Deployed Slice: Startup Action Milestones

- PR #1194 was approved by Hermes and Grok 4.5 on exact head `a35e44eb2`; CI
  passed and it merged to `v8` as `b15d349be`.
- VPS5 fast-forwarded without bot signals or restarts. Bot PIDs
  `850148/850296/850370/850436/850495` and unrelated `misc:0.0` PID `434835`
  remained unchanged.
- The exact bounded rotated `startup_milestones` report returned `ok=true`,
  scanned 12 files / 46,748 records with zero errors/warnings, retained
  truncated lifecycle evidence as explicit unknowns, and observed KuCoin's
  first cycle at `110.653s` without claiming unseen Rust/write milestones.
- The first smoke caught one recovered Binance `InvalidNonce` and two transient
  `D` samples during report I/O. After settling, all `D` states cleared; the
  final two-minute smoke was hard-green with `384/384` remote and `76/76`
  account-critical calls successful, 5/5 processes matched (`R=4,S=1`), no
  hard/log failures, no pipeline drops/sink errors, and a clean tracked repo.

### Active Slice: Startup Readiness Consumer Correctness

### Deployed Slice: Startup Readiness Consumer Correctness

- PR #1195 was approved by Hermes and Grok 4.5 on exact head `e9aab6b97`; CI
  passed and it merged to `v8` as `739ebd49d`.
- VPS5 fast-forwarded without signals or restarts. Bot PIDs
  `850148/850296/850370/850436/850495` and unrelated `misc:0.0` PID
  `434835` remained unchanged.
- Bounded readiness and milestone reports scanned 12 files with zero issues.
  Incomplete current sources no longer attached stale rotated lifecycle data;
  KuCoin retained its bounded sparse HSL terminal context and exposed its first
  cycle at `220.123s`.
- The first smoke caught a real KuCoin balance timeout. After it aged out, the
  retry was green with `284/284` remote and `57/57` account-critical calls
  successful, all five bots matched, no hard/log/monitor failures, no event
  pipeline errors, and a clean tracked repository. Two report-time `D`
  samples cleared; all five bots were `Rsl+` on the quiet follow-up.

### Active Slice: Fresh-Entry Eligibility Evidence

- Branch: `codex/v8-fresh-entry-eligibility` from deployed `739ebd49d`.
- Scope: emit one bounded, correlated observability contract that distinguishes
  `no_candidate`, `blocked_candidate`, `protective_only`,
  `already_satisfied`, and `eligible` after existing reconciliation and
  local pre-connector filters.
- Non-goals: no duplicate or changed entry gate, no `to_create` mutation, no
  connector/exchange outcome claim, no Rust/order/risk/HSL behavior change, and
  no event-sink failure propagation into execution. Expected VPS action is an
  exact five-bot restart, bounded event query, and immediate plus settled smoke.

### Deployed Slice: Fresh-Entry Eligibility Evidence

- PR #1196 was approved by Hermes and Grok 4.5 on exact head `baf7d6794`; CI
  passed and it merged to `v8` as `2f6f61da6`.
- VPS5 fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes. Python process PIDs became
  `856325/856354/856386/856420/856456`; unrelated `misc:0.0` remained PID
  `434835`.
- A bounded current-segment query observed three
  `entry.initial_eligibility` events. The payloads exposed eligible,
  `initial_entry_distance_gate` blocked, and `rust_no_initial_candidate`
  outcomes with full aggregates, 32-record sampling, cycle IDs, and order-wave
  IDs without price/quantity/raw payload leakage.
- The immediate smoke retained one real pre-restart KuCoin balance timeout;
  KuCoin also had one post-restart authoritative-state timeout. After recovery,
  a two-minute smoke was hard-green with `348/348` remote and `43/43`
  account-critical calls successful, all five expected processes matched, no
  hard/log/pipeline failures, and a clean tracked repository. The quiet
  one-minute smoke remained green at `175/175` and `34/34`; transient sampled
  `D` states cleared to exact `R/S` process states.

### Active Slice: Fresh-Entry Startup Milestone

- Branch: `codex/v8-fresh-entry-startup-milestone` from deployed
  `2f6f61da6`.
- Scope: derive bounded current-lifecycle
  `first_fresh_entry_eligible` performance-report evidence only from an
  `entry.initial_eligibility` event whose eligible outcome count is a positive
  integer. Mark the milestone `entry_blocker` and keep absent evidence
  explicitly unknown.
- Non-goals: no producer, trading/readiness gate, connector/exchange outcome
  claim, process action, Rust/order/risk/HSL behavior, backtest, optimizer, or
  smoke verdict change. Expected VPS action is pull plus a bounded
  current-plus-rotated report and settled smoke, with no bot restart.

### Deployed Slice: Fresh-Entry Startup Milestone

- PR #1197 was approved by Hermes and Grok 4.5 on exact head `cb76fd5b9`; CI
  passed and it merged to `v8` as `fda0e1323`.
- VPS5 fast-forwarded without bot signals or restarts. Bot PIDs
  `856325/856354/856386/856420/856456`, their exact pane PIDs, and unrelated
  `misc:0.0` PID `434835` remained unchanged; the tracked repository was clean.
- A bounded eight-segment Binance lifecycle report returned `ok=true` with no
  issues and observed first cycle at `84.484s`, first Rust call at `239.008s`,
  first submitted write at `240.106s`, and
  `first_fresh_entry_eligible` at `240.110s`.
- The settled two-minute smoke was hard-green with `208/208` remote and `53/53`
  account-critical calls successful, all five expected processes matched, no
  hard/log/monitor failures, and a clean tracked repository. One sampled `D`
  state cleared; all five bots were `R` on the final exact-state check.

### Active Slice: Connector Call Boundary Evidence

- Branch: `codex/v8-connector-invocation-events` from deployed `fda0e1323`.
- Scope: emit one bounded structured/monitor event immediately before each
  concrete `cca.create_order` or `cca.cancel_order` call through the base,
  Hyperliquid, and OKX routes. Preserve normal cycle/order-wave/action
  correlation without adding fields to connector payloads.
- Contract: the events prove only local connector call-site arrival. They do
  not claim bytes sent, exchange receipt, acceptance, or acknowledgement, and
  they do not create another startup milestone. Diagnostic failures remain
  isolated from connector execution.
- Non-goals: no order, risk, HSL, Rust, backtest, optimizer, connector payload,
  or exchange-error behavior change. Expected VPS action is an exact five-bot
  graceful restart, legitimate-activity event query, and immediate plus settled
  smoke; validation must not create synthetic live orders.

### Deployed Slice: Connector Call Boundary Evidence

- PR #1198 was approved by Hermes and Grok 4.5 on exact head `c13d27143`; CI
  passed and it merged to `v8` as `e94da301b`.
- VPS5 fast-forwarded cleanly and gracefully stopped only the five exact bot
  panes. All old Python processes exited naturally. Existing pane PIDs and
  unrelated `misc:0.0` PID `434835` remained preserved; the exact supervisor
  commands then started bot PIDs `861950/861949/861953/861955/861957`.
- The immediate smoke retained one real KuCoin authoritative-state timeout:
  `264/267` remote and `15/18` account-critical calls succeeded. After recovery,
  the settled two-minute smoke was hard-green with `325/325` remote and `53/53`
  account-critical calls successful, all five expected processes matched, no
  hard/log/monitor/pipeline failures, no `D` states, and a clean tracked repo.
- Current post-restart segments contained neither `execution.*_sent` nor
  connector-call events. Validation therefore retained explicit no-observation
  evidence and did not manufacture a live order.

### Active Slice: Startup Fill-Cache Proof Correlation

- Branch: `codex/v8-startup-fill-cache-proof` from deployed `e94da301b`.
- Scope: add one bounded performance-report section that joins the current
  `bot.started` lifecycle with existing startup cache-load and exact fill-history
  proof evidence. Cache presence alone must never claim coverage proof.
- Contract: report `proven`, `unproven`, or explicit `unknown` from valid
  post-start proof evidence only, preserve incomplete-source barriers and
  lifecycle resets, expose cache/proof ordering, and retain only bounded
  allowlisted proof values plus valid elapsed/phase relation evidence.
- Non-goals: no event producer, startup/readiness/risk gate, existing report
  reinterpretation, process action, order/HSL/Rust/backtest/optimizer/exchange
  behavior, or smoke-verdict change. Expected VPS action is pull plus bounded
  report and settled smoke, with no bot restart.

### Deployed Slice: Startup Fill-Cache Proof Correlation

- PR #1199 was approved by Hermes and Grok 4.5 on exact head `3bd521424`; CI
  passed and it merged to `v8` as `7eca90037`.
- VPS5 fast-forwarded without bot signals or restarts. A bounded four-segment
  per-bot `startup_fill_cache_proof` report returned `ok=true` with zero issues,
  reported all five current bots `proven`, observed every cache load before
  proof, and measured proof elapsed from `5.851s` to `38.306s`.
- The first smoke retained a real KuCoin balance timeout; the next retained a
  recovered KuCoin nonce error. After both aged out, the final two-minute smoke
  was green with `283/283` remote and `39/39` account-critical calls successful,
  zero hard/log/monitor/process failures, and a clean tracked repository.
- One report-time `D` sample cleared. The final exact process check showed all
  five bots in `R/S`; bot PIDs `861949/861950/861953/861955/861957`, pane PIDs
  `856294/856332/856364/856398/856434`, and unrelated `misc:0.0` PID `434835`
  remained unchanged.

### Active Slice: Event-Pipeline Service-Time Evidence

- Branch: `codex/v8-event-pipeline-service-time` from deployed `7eca90037`.
- Triggering evidence: current event-pipeline health exposes queue depth,
  unfinished work, drops, sink errors, and worker liveness but not bounded
  queue-wait or worker service-time evidence.
- Architecture selection is in progress. The slice must remain observability
  only, bounded, low-cardinality, and isolated from trading decisions and event
  delivery policy.
- Selected design: private queue envelopes carry monotonic enqueue timestamps.
  The worker aggregates processed count, queue-wait total/max, and sink-service
  total/max over each health-summary window. Periodic structured health emission
  rotates the timing window; accepted enqueue commits it, while failed enqueue
  merges it back into the active window. Ordinary monitor snapshots do not
  consume timing evidence.
- Consumers: project only fixed numeric fields through smoke event-pipeline
  summaries and performance resource-pressure statistics. Do not add timing
  verdicts, alerts, thresholds, histograms, sampled slow events, labels, raw
  payloads, queue/drop-policy changes, or trading/readiness gates.
- Expected VPS action: exact five-bot graceful restart, bounded timing evidence,
  and immediate plus settled smoke while preserving unrelated processes and
  local artifacts.

### Deployed Slice: Event-Pipeline Service-Time Evidence

- PR #1200 was approved by Hermes and Grok 4.5 on exact head `9b7baf0b6`; CI
  passed and it merged to `v8` as `cbf1e5ed5`.
- VPS5 fast-forwarded cleanly and gracefully stopped only the five exact bot
  panes. All old Python processes exited naturally, including KuCoin after a
  bounded uninterruptible wait. Existing pane PIDs and unrelated `misc:0.0` PID
  `434835` remained preserved; the exact supervisor commands then started bot
  PIDs `865892/865897/865901/865903/865905`.
- Fresh smoke and performance reports projected three health windows with
  `1517` processed events, queue-wait total/max `29858.257/1077.844ms`, and
  worker-service total/max `63316.18/1033.432ms`; drops, sink errors, degraded
  counts, and unhealthy pipelines were zero.
- A real GateIO authoritative-balance timeout made one intermediate smoke red.
  After recovery, the final two-minute smoke was hard-green with `432/432`
  remote and `59/59` account-critical calls successful, all five expected
  processes matched, exact states `R/R/R/R/S`, and a clean tracked repository.

### Active Slice: Event-Pipeline Sink Attribution

- Branch: `codex/v8-event-pipeline-sink-attribution` from deployed
  `cbf1e5ed5`.
- Triggering evidence: live worker-service maxima exceeded one second while
  drops, sink errors, and degraded counts remained zero, but the aggregate
  timing cannot distinguish structured from monitor sink cost.
- Scope: add fixed per-health-window structured/monitor sink write counts and
  per-write service total/max. Use the existing monotonic worker clock and
  transactional timing-window token; preserve aggregate worker timing and all
  queue/routing/delivery behavior.
- Consumers: project only fixed numeric fields through smoke event-pipeline
  summaries and performance resource-pressure statistics. Historical events
  without these fields remain absent rather than synthetic zero.
- Non-goals: no console/text timing, labels, samples, thresholds, verdict
  changes, queue policy, routing, event payload, order, risk, HSL, Rust,
  backtest, optimizer, exchange, or trading behavior changes.
- Expected VPS action: exact five-bot graceful restart, bounded per-sink timing
  evidence, and immediate plus settled smoke while preserving unrelated
  processes and local artifacts.

### Deployed Slice: Event-Pipeline Sink Attribution

- PR #1203 was approved by Hermes and Grok 4.5 on exact head `3239c6678`; CI
  passed and it merged to `v8` as `381f6a3f7`.
- VPS5 fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes. Old bot processes exited naturally. Pane PIDs
  `856294/856332/856364/856398/856434` and unrelated `misc:0.0` PID `434835`
  were preserved; new bot PIDs were `867321/867324/867328/867331/867333`.
- Four fresh bots reported 2,525 processed events and monitor writes, zero
  structured writes, monitor service total/max `62842.105/5321.841ms`, worker
  service total/max `62937.772/5321.883ms`, and queue-wait total/max
  `18035.11/2350.05ms`. Drops, sink errors, degraded counts, and unhealthy
  pipelines remained zero, so the runtime bottleneck was localized to the
  monitor sink without changing delivery policy.
- The final settled two-minute smoke was hard-green with `360/360` remote and
  `55/55` account-critical calls successful, all five expected processes
  matched, exact states `R/R/R/R/S`, and a clean tracked repository.

### Active Slice: Monitor-Publisher Phase Attribution

- Branch: `codex/v8-monitor-publisher-phase-timing` from deployed `381f6a3f7`.
- Scope: attribute each real `MonitorEventSink` write to fixed monotonic
  event-conversion, publisher lock-wait, top-level rotation, serialization plus
  NDJSON append, and post-append manifest/retention maintenance phases. Retain
  the existing monitor-sink total and rotate/confirm/restore all new totals and
  maxima with the existing opaque health-window token.
- Consumers: project only fixed numeric fields through smoke event-pipeline
  summaries and performance resource-pressure statistics. Generic monitor sinks
  report zero internal phases; historical records omit absent fields.
- Non-goals: no publisher lock, fsync/durability, rotation, compression,
  retention, routing, queue/backpressure, drop/error policy, payload, label,
  threshold, alert, verdict, order, risk, HSL, Rust, backtest, optimizer,
  exchange, or trading behavior change.
- Expected VPS action: exact five-bot graceful restart, bounded phase timing
  evidence, and immediate plus settled smoke while preserving unrelated
  processes and local artifacts.

### Deployed Slice: Monitor-Publisher Phase Attribution

- PR #1204 was approved by Hermes and Grok 4.5 on exact head `a538b27c2`; CI
  passed and it merged to `v8` as `d1303f55b`.
- VPS5 fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes. All old bot processes exited naturally. Pane PIDs
  `856294/856332/856364/856398/856434` and unrelated `misc:0.0` PID `434835`
  were preserved; new bot PIDs were `869404/869406/869556/869558/869564`.
- Four complete fresh health windows covered 2,643 monitor writes. Monitor
  service total/max was `53903.942/1676.541ms`; maintenance
  `47478.85/900.321ms`; persistence `3619.444/77.067ms`; lock wait
  `2190.037/1661.187ms`; rotation `245.173/20.658ms`; and conversion
  `54.686/1.139ms`. Maintenance represented 88.08% of cumulative monitor
  service and averaged 17.964ms per write. The worst individual write was
  instead almost entirely publisher lock wait.
- The settled two-minute smoke was hard-green with `346/346` remote and
  `51/51` account-critical calls successful, all five expected processes
  matched, exact states `R/R/R/R/S`, and a clean tracked repository.

### Active Slice: Monitor Manifest Checkpoint Coalescing

- Branch: `codex/v8-monitor-manifest-coalescing` from deployed `d1303f55b`.
- Scope: coalesce ordinary best-effort manifest checkpoints to the existing
  snapshot interval, force checkpoints at lifecycle and rotation boundaries,
  and recover the maximum checksummed current-segment event sequence with a
  fixed-memory reverse chunk scan when the manifest is stale after an unclean
  exit.
- Non-goals: no change to NDJSON append/fsync semantics, lock ordering,
  retention/compression, event payloads, routing, queue/backpressure policy,
  monitor configuration, trading behavior, or lock-holder attribution.
- Expected VPS action: exact five-bot graceful restart, immediate and settled
  smoke, and before/after monitor maintenance plus lock-wait evidence.

### 2026-07-15: Console Policy And Raw-Balance Materiality

### 2026-07-15: Activated Console Demotions And Forager Refresh Follow-Up

### 2026-07-15: Forager Refresh Demotion Deployed And Selection Ownership

- PR #1236 merged to canonical `master` at `d4b3055da6`. VPS5 fast-forwarded
  cleanly and all five exact bots exited naturally after one SIGINT round.
  Their pane PIDs and unrelated `misc:0.0` PID `434835` remained unchanged;
  expected untracked artifacts were preserved.
- The immediate smoke caught one KuCoin authoritative-open-orders timeout.
  The settled two-minute report was `ok=true` with `216/216` remote calls and
  `57/57` account-critical calls successful, eight successful fill refreshes,
  all five processes/configs matched, and zero hard, log-attention, monitor, or
  pipeline failures. One sampled `D` process state cleared immediately.
- Fresh logs contain zero `forager refresh complete` INFO lines on all five
  bots while normal candle/cache activity continued. Natural material forager
  selections on Binance, GateIO, and OKX instead exposed two console owners:
  the structured `[forager] succeeded ...` route and the producer's richer,
  materiality-aware `[forager] long selection ...` summary.
- The active `codex/forager-selection-console-ownership` slice keeps every
  `forager.selection` event in structured and monitor sinks but removes the
  Rust-orchestrator events' independent console/text projection. Python-filter
  selection events remain operator-visible; the Rust producer's selected-set,
  slot, and replacement materiality plus periodic heartbeat remain unchanged.

### 2026-07-15: Forager Selection Ownership Deployed And Open-Tail Follow-Up

- PR #1237 merged to canonical `master` at `a6aad93902` after exact-head Hermes
  and Grok approval plus green Python and Rust CI. VPS5 fast-forwarded cleanly.
  Two bots exited immediately after one SIGINT round; GateIO and Binance briefly
  entered uninterruptible I/O sleep while KuCoin drained, then all three exited
  naturally within the bounded wait. Exact pane PIDs and unrelated `misc:0.0`
  PID `434835` remained unchanged.
- The immediate smoke was `ok=true` with `19/19` remote calls and `14/14`
  account-critical calls successful. The settled smoke was also `ok=true` with
  `249/251` remote calls, all `47/47` account-critical calls, six successful
  fill refreshes, five valid processes, and zero hard, log-attention, monitor,
  or event-pipeline failures. Two KuCoin candle-fetch timeouts were non-hard.
- Natural Binance, GateIO, and OKX logs retained only the richer producer-owned
  material selection INFO line. Exact Rust `forager.selection` records remained
  in monitor storage, proving the structured/monitor path stayed durable while
  the duplicate operator projection was removed.
- The first fresh cycle emitted one `open-tail EMA projection contexts` INFO
  record on every bot. The lines measured 1002-1050 characters and expanded up
  to eight symbol timestamp contexts for 19-39 flat projections per bot. The
  active `codex/open-tail-projection-console-detail` slice retains that routine
  aggregate at DEBUG while preserving compact active-tail warnings, structured
  late-projection events, EMA values, and readiness.

### 2026-07-15: Open-Tail Detail Demotion Deployed And Candle Failure Follow-Up

- PR #1238 merged to canonical `master` at `c70a88b147` after exact-head Hermes
  and Grok approval plus green Python and Rust CI. VPS5 fast-forwarded cleanly,
  and all five exact bots exited naturally after one SIGINT round. Exact pane
  PIDs and unrelated `misc:0.0` PID `434835` remained unchanged.
- The immediate window retained one real KuCoin authoritative-balance timeout
  and degraded cycle. The settled two-minute smoke was `ok=true` with `262/264`
  remote calls and all `41/41` account-critical calls successful, six
  successful fill refreshes, five valid processes, and zero hard, log,
  monitor, or event-pipeline failures. The two remaining failures were non-hard
  candle-fetch timeouts.
- Every bot completed a natural market-ready cycle and normal INFO logs
  contained zero `open-tail EMA projection contexts` records, proving the
  demotion without manufacturing candle or trading events.
- Those natural candle timeouts exposed the next operator-sink issue: KuCoin
  and GateIO warnings measured 513 and 487 characters and retained duplicated
  raw exception text, full request URLs, and query parameters. The active
  `codex/bounded-candle-fetch-warning` slice replaces the normal text with
  bounded retry/terminal signatures. Durable remote-call events omit raw
  exception text; explicit URLs retain only a redacted marker and stable hash.

### 2026-07-15: Bounded Candle Failures Deployed And Refresh Timing Follow-Up

- PR #1239 merged to canonical `master` at `a97a815a52` after exact-head Hermes
  and Grok approval plus green Python and Rust CI. Its final review fix ensures
  the first throttled warning is emitted during early process uptime.
- VPS5 fast-forwarded cleanly and all five exact bots exited naturally after
  one SIGINT round. Exact pane PIDs and unrelated `misc:0.0` PID `434835`
  remained unchanged; expected untracked artifacts were preserved.
- The immediate startup window retained one real KuCoin authoritative-state
  timeout and degraded cycle. It aged out before the settled two-minute smoke,
  which was `ok=true` with `219/219` remote calls and `50/50` account-critical
  calls successful, six successful fill refreshes, five matching processes and
  configs, states `R=4,S=1`, and zero hard, log, monitor, process, or
  event-pipeline failures. No natural candle-fetch failure occurred in the
  bounded post-restart window, so the new warning format was not manufactured.
- The prior console-volume sample contained 33 immediate completed staged
  refresh timing INFO lines, about 13.2 records per bot-hour and more than 20%
  of the non-action budget. Natural post-restart startup output again contained
  sub-ten-second lines. The active `codex/staged-refresh-console-threshold`
  slice keeps those completed lines at DEBUG while preserving interesting
  structured INFO events, periodic timing summaries, readiness, and refresh
  behavior.

### 2026-07-15: Staged Refresh Threshold Deployed And Fill Timing Follow-Up

### 2026-07-15: Fill Timing Demotion Deployed And OKX Config Outcome Follow-Up

- PR #1241 merged to canonical `master` at `953ac80e31` after exact-head Hermes
  and Grok approval plus green Python and Rust CI. VPS5 fast-forwarded cleanly,
  and all five exact bots exited naturally after one SIGINT round. Exact pane
  PIDs and unrelated `misc:0.0` PID `434835` remained unchanged.
- The settled smoke was `ok=true` with `225/225` remote calls and `48/48`
  account-critical calls successful, eight successful fill refreshes, all HSL
  replays complete, five matching processes/configs, states `R=4,S=1`, and
  zero hard, log, monitor, process, or event-pipeline failures.
- Fresh natural logs contained zero successful fill-refresh or fetcher-request
  timing INFO lines, while the structured smoke window retained all eight
  successful refresh summaries. No fill or failure was manufactured.
- Code and policy review identified the next narrow sink boundary: OKX's
  explicit `59107` already-configured response prints at INFO even though it
  proves no setting changed. The active `codex/okx-config-outcome-event` slice
  adds bounded per-symbol structured outcomes and demotes only that unchanged
  line, preserving normal success/failure visibility and exchange behavior.
- A read-only four-hour VPS5 sample found nine normal OKX `margin=ok` outcomes
  and no `59107`, so the normal producer path can be validated naturally after
  deploy while the unchanged branch remains a local regression-test claim.
  The same window exposed nine Binance mixed leverage/unchanged lines; that
  separate connector contract is intentionally outside this OKX-scoped PR.

### 2026-07-15: OKX Config Outcomes Deployed And HSL Progress Follow-Up

### 2026-07-15: HSL Cadence Deployed And Execution Incident Follow-Up

- PR #1243 merged to canonical `master` at `c6fba82903` after exact-head
  Hermes and Grok approval plus green Python and Rust CI. VPS5 fast-forwarded
  cleanly, and all five exact bots exited naturally after one SIGINT round;
  KuCoin was last at 35 seconds. Pane PIDs and unrelated `misc:0.0` PID
  `434835` remained unchanged.
- Natural Binance, GateIO, and KuCoin replay progress lines were all at least
  30 seconds apart. A five-minute structured window retained 102 replay
  progress events while the console projected only 15 intermediate lines,
  proving durable detail was preserved. The final two-minute smoke was
  `ok=true` with `204/204` remote and `55/55` account-critical calls
  successful, eight fill refreshes, five matching processes/configs, and no
  hard, log, monitor, process, or event-pipeline failures.
- Two real KuCoin startup timeouts recovered without intervention. The normal
  tmux console exposed a full request URL, an unconditional traceback, and an
  untagged error-budget line. The active
  `codex/execution-incident-projection` slice replaces that family with a
  bounded signature while preserving counters, restart/backoff, timestamp
  recovery, exchange calls, and trading behavior.

### 2026-07-15: Incident And Memory Projections Deployed; Risk Status Follow-Up

- PR #1244 merged as `bff64d3a82` and PR #1245 merged as `de5f21c96d` under
  the temporary maintainer-authorized Hermes-plus-CI gate while Grok was
  halted. PR #1245's current-master integration retained both additive event
  contracts; direct target-relative diff proofs and 428 focused tests showed
  no semantic production/test delta from its approved head.
- VPS5 fast-forwarded cleanly and activated both PRs with one exact five-bot
  restart. All old bots exited naturally after one SIGINT round; KuCoin was
  last at 40 seconds. Exact pane PIDs and unrelated `misc:0.0` PID `434835`
  remained unchanged.
- One natural KuCoin startup timeout made the immediate window red, then
  recovered without intervention. Its normal log retained a bounded operation,
  exception type, endpoint, and action without a raw URL or traceback. All five
  bots naturally emitted complete `resource.memory_snapshot` monitor payloads
  and compact 84-107 character console lines.
- The final two-minute smoke was `ok=true` with `217/217` remote calls and
  `54/54` account-critical calls successful, six successful fill refreshes,
  five config-valid processes in states `R=4,S=1`, and zero hard, log,
  monitor, process, or event-pipeline failures. The checkout was clean at the
  exact merged head.
- The preceding Hyperliquid segment contained 27 trailing and 20 unstuck INFO
  lines over about 135 minutes. Sub-display-precision trailing drift and
  `-1.70` to `-1.76` allowance oscillation repeatedly appeared as changed. The
  active `codex/trailing-unstuck-console-materiality` slice preserves every
  five-minute structured observation while applying explicit producer-owned
  transition and numeric materiality to console/text output.

### 2026-07-15: Risk Status Materiality Deployed; Trailing Format Follow-Up

- PR #1246 merged as `fc5c2cae88` after exact-head Hermes approval and green
  Python/Rust CI under the temporary maintainer-authorized Hermes-plus-CI gate.
  The stale progress-ledger current-work block was explicitly marked historical
  after Hermes identified the contradictory handoff.
- VPS5 fast-forwarded cleanly and activated the PR with one exact five-bot
  restart. All old bots exited naturally after one SIGINT round; KuCoin was
  last at about 30 seconds. Exact pane PIDs and unrelated `misc:0.0` PID
  `434835` remained unchanged.
- The first natural cadence emitted five durable trailing/unstuck events with
  `changed=true` and `operator_visible=true` plus five console lines. The second
  cadence emitted five durable events with `changed=false` and
  `operator_visible=false` and zero console lines. Hyperliquid naturally varied
  below the configured materiality boundaries, proving data retention and
  human suppression without manufactured activity.
- The final two-minute smoke was `ok=true`: `224/224` remote and `56/56`
  account-critical calls succeeded, nine fill refreshes succeeded, five exact
  processes/configs were in state `R`, and hard/log/monitor/process/pipeline
  failures were zero. The repository was clean at the exact merged head.
- The first visible Hyperliquid trailing line measured 311 characters and
  repeated verbose field names already represented in the complete event. The
  active `codex/compact-trailing-status-console` follow-up is formatter-only and
  targets the normal 240-character budget without changing payload or behavior.

### 2026-07-16: Exact Restart Target Preflight Deployed

- PR #1287 merged to canonical `master` at `6cb8bc3c73` after exact-head
  Hermes approval and green Python/Rust CI. VPS5 fast-forwarded cleanly without
  a restart or process signal; all five pane PIDs, bot PIDs, and unrelated
  `misc:0.0` PID `434835` remained unchanged.
- Immediate and settled local-only target reports were `ok=true` with five
  expected/resolved windows, canonical pane IDs `%358/%359/%360/%361/%362`,
  exact PPID-to-pane-PID ownership, and zero missing, duplicate, extra, config,
  or scan failures. The immediate `D=1,R=2,S=2` process snapshot settled to
  `R=5` naturally.
- No exchange request, process action, or event was manufactured. Active
  `codex/restart-target-stability` adds a bounded pre-action identity window
  without making restart execution available.

### 2026-07-16: Stable Restart Targets Deployed

- PR #1288 merged to canonical `master` at `e004ede7dd` after exact-head
  Hermes approval and green Python/Rust CI. VPS5 fast-forwarded cleanly without
  a restart or process signal; all five pane IDs/PIDs, bot PIDs, and unrelated
  `misc:0.0` PID `434835` remained unchanged.
- Immediate and settled three-sample local-only target reports were `ok=true`
  and `sampling.stable=true`, with five stable targets, `3/3` successful
  samples, zero failed samples, zero changed targets, and no issues. Immediate
  process state `R=2,S=3` settled to `R=4,S=1` naturally.
- No exchange request, process action, or event was manufactured. Active
  `codex/restart-plan-target-gate` adds the exact stable report command and
  required verdict to the existing non-executing restart plan.

### 2026-07-16: Restart Target Relaunch Contract Deployed

- PR #1290 merged to canonical `master` at `b490bd75be` after exact-head
  Hermes approval and green Python/Rust CI. VPS5 fast-forwarded from
  `d6cac101` without a restart or process signal; exact pane IDs/PIDs, bot PIDs
  `985592/985594/985596/985598/985600`, and unrelated `misc:0.0` PID `434835`
  remained unchanged.
- Immediate and settled three-sample local-only target reports were `ok=true`
  and stable with all five resolved targets relaunch-ready, zero failed
  samples, zero identity changes, and no issues. Every relaunch proof requires
  verified process exit and an exact post-stop pane recheck. The compact plan
  was `ok=true`, had zero issues, and required the same all-targets-ready
  verdict while execution remained unavailable.
- A bounded four-sample process report observed real temporary I/O waits with
  five stable PIDs and no hard/config/process failure. The quiet exact-PID
  follow-up cleared to `R=4,S=1`. No exchange request, process action, or event
  was manufactured. Active `codex/restart-target-contract-fingerprint` binds
  target stability to the parsed supervisor command contract without exposing
  command content.

### 2026-07-16: Stable Target Gate Bound Into Restart Plan

- PR #1289 merged to canonical `master` at `d6cac1017b` after exact-head
  Hermes approval and green Python/Rust CI. VPS5 fast-forwarded cleanly without
  restart or signal; all five pane IDs/PIDs, bot PIDs, and unrelated
  `misc:0.0` PID `434835` remained unchanged.
- The compact plan was `ok=true` for five bots with zero issues and emitted the
  exact three-sample/five-second local-only target command while keeping
  execution unavailable. A four-sample local-only process smoke was hard-green
  with five stable PIDs and no persistent uninterruptible sleep; the final
  exact sample settled to `R=3,S=2`.
- No exchange request, process action, or event was manufactured. Active
  `codex/restart-target-relaunch-contract` proves the pane-parent candidate path
  and mandatory post-stop recheck without emitting commands or adding process
  control.

### 2026-07-16: Brief Process Report Deployed

- PR #1286 merged to canonical `master` at `7a12430abf` after exact-head
  Hermes approval and green Python/Rust CI. VPS5 fast-forwarded cleanly without
  a restart or process signal; all five configured pane PIDs and unrelated
  `misc:0.0` PID `434835` remained unchanged.
- Immediate and settled four-sample brief reports were `ok=true` with all five
  expected commands/configs matched, five stable PIDs, and zero missing,
  duplicate, extra, config, or scan failures. The first naturally ended with
  one active but non-persistent `D`; the settled repeat observed recovery and
  ended `R=3,S=2` with zero active or persistent uninterruptible processes.
- No exchange request, process action, or event was manufactured. Active
  `codex/restart-target-preflight` adds the read-only exact tmux pane ownership
  prerequisite for future restart execution.

### 2026-07-16: Local-Only Process Report Deployed

- PR #1285 merged to canonical `master` at `175986fd9c` after exact-head
  Hermes approval and green Python/Rust CI. VPS5 fast-forwarded cleanly without
  a restart or process signal; five pane parents, exact bot PIDs
  `985592/985594/985596/985598/985600`, and `misc:0.0` PID `434835` remained
  unchanged.
- Immediate and settled four-sample local-only process reports were `ok=true`
  with all five expected commands/configs matched, five stable PIDs, and zero
  missing, duplicate, extra, config, or scan failures. The first report
  naturally observed recovered Binance uninterruptible sleep and ended with
  GateIO in `D`; the settled report naturally observed GateIO recovery and
  ended `R=5` with zero active or persistent uninterruptible processes.
- No exchange request, process action, or event was manufactured. Active PR
  #1286 adds aggregate-only repeated-check output because the complete
  process/config/group rows are intentionally detailed but too large for
  routine operator loops.

### 2026-07-16: Process Sampling Deployed; Local-Only Validation Split

- PR #1283 merged to canonical `master` at `9bcf6c24fe` after current-head
  Hermes approval and green Python/Rust CI under the temporary
  maintainer-authorized Hermes-plus-CI gate. The final changelog-only
  integration delta preserved the reviewed production/test/config/contract
  diff, and 132 focused local smoke, incident, and docs tests passed.
- VPS5 fast-forwarded from `82bfa98e` to `9bcf6c24` with a tracked-clean
  checkout. Configured pane process IDs
  `856294/856332/856364/856398/856434` and unrelated `misc:0.0` PID `434835`
  remained unchanged; no restart or process signal was required.
- The requested full `live-smoke-report` validation was not executed after the
  production-action approval layer rejected it as a possible authenticated
  exchange probe. The rejection was not retried or bypassed, and no exchange
  request, process action, or trading/risk/state event was manufactured.
- The active `codex/smoke-process-only-sampling` follow-up adds a dedicated
  local-only process-report command. It reuses the bounded #1283 sampler while
  remaining outside monitor-event, text-log, credential-store, network,
  exchange, process-control, and file-write paths.

### 2026-07-16: Cycle Recovery Health Deployed; Process Sampling Follow-Up

- PR #1282 merged to canonical `master` at `82bfa98e20` after exact-head
  Hermes approval and green Python/Rust CI under the temporary
  maintainer-authorized Hermes-plus-CI gate. VPS5 fast-forwarded cleanly
  without a restart or process signal; the five exact bot PIDs, pane parents,
  and unrelated `misc:0.0` PID `434835` remained unchanged.
- The immediate focused report naturally retained a latest KuCoin
  `InvalidNonce` degradation. The fresh two-minute smoke was `ok=true` with
  zero hard failures, `193/194` remote calls, `56/57` account-critical calls,
  `9/9` fill refreshes, and five exact/config-valid processes.
- The settled focused report was `ok=true`: all five latest cycle outcomes
  were successful, and KuCoin was counted as completed after the retained
  degradation. A final exact process check then observed that unchanged
  KuCoin PID in `D` for about 20 seconds before three consecutive `R` samples.
  No event, failure, process state, or trading activity was manufactured.
- The active `codex/smoke-process-state-sampling` follow-up adds opt-in bounded
  process-state sampling so repeated smoke checks can distinguish observed,
  persistent, and recovered uninterruptible sleep while preserving the
  one-snapshot default and existing smoke verdict.
