# Live Logging Overhaul Current Status

Updated: 2026-07-20.

This is the compact operational source for the active logging-overhaul loop.
Read it before the historical progress ledger. Update it whenever the active
PR, head SHA, review gate, deployed SHA, VPS state, or next action changes.

## Goal

Deliver bounded, correlated, operator-useful live observability through small
reviewable PRs, current-head review gates, and evidence-based VPS5 validation.
Rust remains the trading-logic authority; observability must not become a
trading control plane.

Estimated completion:

- core event/observability architecture: about 85%
- original logging migration: about 70%
- expanded logging, performance-readiness, restart, and ops scope: about 65%
  overall, with substantial uncertainty from the intentionally growing backlog

## Active Review Slice

- Pending PR, `Report live smoke log scan cost`; branch:
  `codex/live-log-scan-cost`, based on canonical
  `f9ceb9678448201af0c0cc5f40f889b661d3021c`.
- Scope: expose bounded elapsed-time, byte, successful-file, selected-line,
  and read-method cost metadata for the text-log scan used by
  `live-smoke-report`, including full, summary, and brief projections.
- Behavior boundary: read-only local tooling only. The slice changes no event
  producer, log discovery, sequential read strategy, tail selection, line
  numbering, matching, redaction, time-window filtering, verdict, monitor
  persistence, exchange access, process inspection/control, planning,
  strategy, order, or risk behavior.
- Baseline: PR #1338's bounded VPS5 incident bundle scanned eight selected log
  files and considered 321 in-window lines after skipping 2,943 older lines,
  but the log report could not identify successful reads, physical/decoded
  bytes, read methods, or elapsed scan time. The active slice closes only that
  diagnostic gap using the established `scan_cost` schema.
- Review gate: exact-current-head Hermes approval plus green Python/Rust CI.
  Built-in Codex automatic review is additional and every finding must be
  verified and resolved.
- Expected VPS action: tracked-clean pull plus bounded read-only smoke and
  incident-bundle reports; no bot restart or process signal.

## Deployed Baseline (PR #1338)

- PR #1338 merged exact reviewed head
  `8924c4ff30f96b8ea3dbf58fc6a97d1faa54514d` as canonical
  `f9ceb9678448201af0c0cc5f40f889b661d3021c` after exact-head Hermes approval
  and green Python/Rust CI. Built-in Codex was additionally requested and had
  posted no finding when the proportional temporary gate merged the PR.
- VPS5 guarded-prepared tracked-clean from
  `848eb60c502b7af21dd7aa52f50f86aece552bd9` without a Rust build, bot
  restart, or process signal. The source fingerprint and compiled stamp stayed
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`.
- The bounded five-minute incident bundle was `ok=true`, found all five exact
  processes, and projected identical time-window `scan_cost` through the full
  report, manifest, and command result: six `seek_tail` files, 8,848 records,
  15,501,809 known physical/decoded bytes, and 3,515.111 ms. Its repository
  evidence was tracked-clean at `f9ceb967`.
- An immediate post-bundle identity snapshot observed four bot processes in
  transient `D` state. After 30 seconds the unchanged exact PIDs were
  `R/R/S/R/R`; pane parents `%358`-`%362` and protected `misc:0.0`
  `%8`/PID `434835` were unchanged. No direct exchange call, manufactured
  event, build, restart, or process signal occurred.

## Previous Deployed Baseline (PR #1329 After PR #1337)

- PR #1337 merged exact reviewed head
  `51a82c23230daf4aea5ec68784ef7168293b408e` as canonical
  `7d463dca56e1b29840954876a25a3f4d0e5df961` after exact-head Hermes and
  built-in Codex approval plus green Python/Rust CI. VPS5 fast-forwarded
  tracked-clean from `e0927ed86f0b10ed8187d8e6a523017baefb98b3`
  without a Rust build. A settled bounded smoke was `ok=true`, found all five
  exact processes, and reported six `seek_tail` files, 10,388 records,
  18,777,444 known physical/decoded bytes, and 5,751.099 ms through the new
  smoke `scan_cost` projection.
- External PR #1329 then merged exact mechanical-integration head
  `6bd2eb0a0a49d21107a2c45ff9784e737d5c4d1c` as canonical
  `848eb60c502b7af21dd7aa52f50f86aece552bd9`. The maintainer explicitly
  approved its contributor CI; exact-head Hermes review and Python/Rust CI
  were green. The target-relative result remained the previously reviewed
  one-line aware-UTC timestamp fix. VPS5 fast-forwarded tracked-clean again
  without a Rust build or restart because the changed Gate.io quarantine
  timestamp path is startup-only and did not warrant disrupting running bots.
- Immediate smokes retained natural KuCoin authoritative-state timeouts and
  were not mislabeled green. The final settled five-minute smoke was `ok=true`
  with zero hard failures, zero failed remote calls, all five exact processes,
  and a tracked-clean repository at `848eb60c`; its monitor scan reported six
  `seek_tail` files, 9,471 records, 17,038,544 known physical/decoded bytes,
  and 4,021.646 ms. Throughout both pulls, bot PIDs
  `1066081/1066091/1066084/1066093/1066087`, pane parents `%358`-`%362`, and
  protected `misc:0.0` `%8`/PID `434835` remained unchanged. No direct
  exchange call, event manufacture, restart, or process signal occurred.

## Previous Deployed Baseline (PR #1333)

- PR #1333 merged exact reviewed head
  `ce5a24d72ea811c6b04a376bbd9fcd228ab4c9af` as canonical
  `e0927ed86f0b10ed8187d8e6a523017baefb98b3` after exact-head Hermes
  approval, a green built-in Codex review, and green Python/Rust CI. VPS5
  fast-forwarded tracked-clean from
  `02fb43f6398fc9edba64849cf2ed0bf0f7a6af09` without a Rust build; the
  source fingerprint/stamp remained
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`.
- The bounded five-minute event query was `ok=true` and reported five files,
  8,133 records, 13,926,420 known physical/decoded bytes, and 4,793.586 ms.
  The matching performance report was `ok=true` with zero errors/warnings and
  reported six files, 8,661 records, 14,740,828 known physical/decoded bytes,
  and 4,781.454 ms. Both used only `seek_tail` reads.
- No bot was restarted or signalled. Bot PIDs
  `1066081/1066091/1066084/1066093/1066087`, pane parents `%358`-`%362`, and
  protected `misc:0.0` `%8`/PID `434835` remained unchanged. Final process
  states were all `Rl+`; the checkout remained tracked-clean. No direct
  exchange call or event was manufactured.

## Previous Deployed Baseline (PR #1335)

- PR #1335 merged exact reviewed head
  `c4d5b8e55f6fd453fd707999ae74ec2dd127e55d` as canonical
  `02fb43f6398fc9edba64849cf2ed0bf0f7a6af09`. VPS5 fast-forwarded
  tracked-clean from `97aa36da4c9a9885c840ea48590837e28e5b8069` without a
  Rust build; the source fingerprint/stamp remained
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`.
- The guarded restart replaced bot PIDs
  `1063302/1063311/1064329/1063314/1063308` with
  `1066081/1066091/1066084/1066093/1066087` under unchanged pane parents.
  A post-action three-sample target report confirmed all five configured
  processes stable with no missing, duplicate, or extra target; protected
  `misc:0.0` remained `%8`/PID `434835`.
- The immediate smoke retained one natural KuCoin authoritative-refresh
  `RequestTimeout` and was not mislabeled green. The fresh settled two-minute
  smoke was `ok=true` with zero hard failures, `47/47` account-critical calls,
  `184/185` remote calls, successful latest cycles, a clean event pipeline,
  and five stable processes. The sole retained remote failure was a non-hard
  OHLCV `RequestTimeout`. No direct exchange call or event was manufactured.

## Previous Deployed Baseline (PR #1334)

- Emergency PR #1334 merged exact reviewed head
  `1fb218a61999ed1cf06ba1974407dbfe350ed0ea` as canonical
  `97aa36da4c9a9885c840ea48590837e28e5b8069`. VPS5 fast-forwarded
  tracked-clean from `77b97ab8c7b7c4e6ba70025c3ca53e828aa0a321` without a
  Rust build; the source fingerprint/stamp remained
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`
  and the artifact SHA-256 remained
  `7611f3eff1d8702ff29d90490a1aba490db5c816e7e3f09a2c33e5c4085da023`.
- The recovery relaunched only the stopped Gate.io pane `%360`, producing bot
  PID `1064329` under unchanged pane parent PID `856364`. The other four bot
  PIDs remained `1063302/1063311/1063314/1063308`, and protected
  `misc:0.0` remained `%8`/PID `434835`.
- Stable three-sample exact-target validation retained all five configured
  processes with no missing, duplicate, or extra target. The final fresh
  two-minute smoke was `ok=true` with zero hard failures, five stable
  processes, and a tracked-clean checkout. An earlier wider smoke retained an
  unrelated natural KuCoin `RequestTimeout` and was not mislabeled green; the
  final window observed one non-hard recovered KuCoin `InvalidNonce`. No direct
  exchange call or event was manufactured.

## Previous Deployed Baseline (PR #1331)

- PR #1331 merged as canonical
  `0a6be3ed00261c6d6a2cdbe0f2588e5709e670f1` after exact-head Hermes
  approval, a green built-in Codex review, and green Python/Rust CI. VPS5
  fast-forwarded tracked-clean from
  `c0386ff5673d93b732786cf12c4cd48f6a381767` without a Rust build; the source
  fingerprint/stamp remained
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`.
- The guarded runner gracefully restarted only exact panes `%358`-`%362`.
  Bot PIDs `1057982/1057991/1057985/1057994/1057988` became
  `1059196/1059205/1059199/1059208/1059202`; pane parents were unchanged and
  protected `misc:0.0` remained `%8`/PID `434835`.
- The immediate bounded smoke was hard-green with complete five-bot lifecycle
  evidence, zero hard/monitor/text-log failures, `81/81` account-critical
  calls, `436/436` remote calls, and five stable processes. The settled smoke
  remained hard-green with `65/65` account-critical and `345/345` remote calls,
  zero latest degraded cycles, both remaining HSL replays complete, and five
  stable processes.
- Natural post-restart snapshots persisted numeric `source_row_count` values
  `36` and `1`. The exact-window performance report produced two summary and
  metric observations with zero malformed or missing proofs. No direct exchange
  call or event was manufactured.

## Previous Deployed Baseline (PR #1328)

- PR #1328 merged as canonical
  `c0386ff5673d93b732786cf12c4cd48f6a381767` after exact-head Hermes
  approval, a green built-in Codex review, and green Python/Rust CI. VPS5
  fast-forwarded tracked-clean from `0a57187ff9f0def7eb4976721f5b04d17f03fb74`.
- Rust source fingerprint and compiled stamp remained
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`;
  no Rust build was required. The guarded runner gracefully restarted only exact
  panes `%358`-`%362` without force. Bot PIDs
  `1056607/1056616/1056610/1056619/1056613` became
  `1057982/1057991/1057985/1057994/1057988`, while protected `misc:0.0`
  remained `%8`/PID `434835`.
- The bounded two-minute restart smoke was hard-green with complete five-bot
  shutdown/startup evidence and zero hard, monitor, or text-log failures. A
  natural post-restart performance report then classified all seven observed
  completed-candle summaries as malformed because the sanitizer had replaced
  `signature_row_count` with `[redacted]`. No direct exchange call or event was
  manufactured; merged PR #1331 corrected only this public diagnostic contract.

## Deployed Baseline (PR #1326)

- PR #1326 merged as canonical
  `0a57187ff9f0def7eb4976721f5b04d17f03fb74` after exact-head Hermes
  approval, a green built-in Codex review, and green Python/Rust CI. VPS5
  fast-forwarded tracked-clean from `ac9ff15029cb728eeb33563c134f196df3e3a49e`.
- Rust source fingerprint and compiled stamp remained
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`;
  no Rust build, bot restart, or process signal was required.
- The bounded offline 240-minute/eight-symbol benchmark matched compact and
  dense-reference final state and all 240 replay samples. It emitted both
  exclusive stage taxonomies and reported zero network, cache, latch, and
  monitor side effects.
- Exact bot PIDs `1056607/1056616/1056610/1056619/1056613` remained running,
  the checkout stayed tracked-clean, and protected `misc:0.0` remained
  `%8`/PID `434835`. No direct exchange call or event was manufactured.

## Previous Deployed Baseline (PR #1325)

- PR #1325 merged as canonical
  `ac9ff15029cb728eeb33563c134f196df3e3a49e`. Its successful fill-fetch
  generation confirms trailing readiness independently of unrelated pending
  PnL while the risk-authoritative generation remains fail-closed.
- VPS5 fast-forwarded tracked-clean from `881978dec502296c4ab990c34bf06fdf74f024b2`
  without a Rust build. The guarded runner gracefully restarted only exact panes
  `%358`-`%362` without force; the five bot PIDs became
  `1056607/1056616/1056610/1056619/1056613` and `misc:0.0` remained unchanged.
- The immediate smoke was hard-green with `58/58` account-critical and
  `224/224` remote calls successful. The settled smoke was also hard-green with
  `17/17` account-critical and `328/328` remote calls successful, zero hard,
  monitor, or text-log failures, three completed HSL replays, and no latest
  degraded cycle. No direct exchange call or event was manufactured.

## Previous Deployed Baseline (PR #1322)

- PR #1322 merged as canonical
  `881978dec502296c4ab990c34bf06fdf74f024b2`. VPS5 fast-forwarded
  tracked-clean from `8b433cc22b087b0efab51ba2bcf003f1e2b31806` without a Rust
  build, bot restart, or process signal.
- The exact Hyperliquid attribution smoke scanned 26 selected artifacts and
  `15763500` bytes, read 51 fill records, returned 37 trailing fills and zero
  warnings, and reduced the previously duplicated `1784414326269` runtime to
  one cohort carrying `runtime_manifest`, `monitor_event`,
  `legacy_startup_log`, and `runtime_log` sources.
- Smaller explicit scan caps failed closed before the successful exact-file
  smoke. Temporary over-broad scanner PIDs created by interrupted local-only
  commands were identity-checked and terminated exactly; no bot or tmux pane
  was signalled.
- Final local-only target sampling was hard-green with five expected and stable
  bot PIDs, no missing, duplicate, or extra process, and natural recovery from
  one transient `D` observation to `R=5`. The checkout remained tracked-clean,
  no scanner remained, and protected `misc:0.0` stayed `%8`/PID `434835`. No
  direct exchange call or event was manufactured.

## Previous Deployed Baseline (PR #1312)

- PR #1312 merged as canonical
  `8b433cc22b087b0efab51ba2bcf003f1e2b31806`. The guarded tracked-clean
  fast-forward from `30870252` required no Rust build; source fingerprint and
  compiled stamp remained
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`,
  with artifact SHA-256
  `7611f3eff1d8702ff29d90490a1aba490db5c816e7e3f09a2c33e5c4085da023`.
- The guarded runner gracefully restarted only exact panes `%358`-`%362`
  without force. Old bot PIDs
  `1044483/1044492/1044486/1044495/1044489` became
  `1048663/1048672/1048666/1048675/1048669`; pane parents and protected
  `misc:0.0` `%8`/PID `434835` were unchanged.
- The exact `1784426343279..1784426527215` lifecycle window retained complete
  five-bot shutdown/startup proof and one naturally emitted hard event rather
  than hiding it. A fresh `1784426762780..1784426882865` window had zero smoke
  hard failures, log errors, or monitor errors. The subsequent ordinary
  two-minute report was hard-green with `46/46` account-critical calls
  successful, successful latest observed cycles, and no latest degraded cycle.
- Final target sampling was 3/3 stable with five exact processes, no missing,
  duplicate, or extra targets, and a tracked-clean repository. No direct
  exchange request or event was manufactured.

## Previous Deployed Baseline (PR #1315)

- PR #1315 merged as canonical
  `308702523760ae7a0b309419ae1616b0a4938721`. The guarded tracked-clean
  fast-forward was `fc9dad83` to the exact merge; the exact target-derived Rust
  fingerprint/stamp
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449` matched
  without a Rust build, bot restart, or signal.
- The bounded Hyperliquid attribution smoke selected 13 files totaling
  `3695111` bytes, scanned 50 fills with 37 trailing, and reported zero warnings.
  All 37 legacy first-ingestion and 37 producer-attribution records correctly
  remained unattributed.
- The same smoke observed the full manifest start `1784414326269` ms and its
  matching startup-log prefix at `1784414328000` ms, exposing the `1731` ms
  second-resolution duplicate-runtime condition addressed by the active slice.
- Bot PIDs `1044483/1044492/1044486/1044495/1044489` remained unchanged;
  protected `misc:0.0` remained pane `%8`/PID `434835`. Final exact target
  sampling retained those PIDs in normal `R/S` states with a tracked-clean
  checkout. No direct exchange call or event was manufactured.

## Previous Deployed Baseline (PR #1321)

- PR #1321 deployed as canonical
  `fc9dad83cd3ecf51cae15e8dda66afb7cfb895a1`. It completes distinct latest
  shutdown lifecycle diagnostics without changing event producers, trading, or
  process behavior.
- VPS5 fast-forwarded tracked-clean at `fc9dad83`; bot PIDs
  `1044483/1044492/1044486/1044495/1044489` were unchanged, and protected
  `misc:0.0` remained pane `%8`/PID `434835`.
- The initial smoke retained four natural KuCoin degraded cycles/timeouts rather
  than hiding them. The settled two-minute smoke was `ok=true` with
  `hard_failures=0`, `44/44` account-critical calls and `243/243` remote calls
  successful, and five configured processes stable.
- This was diagnostics-only: no Rust build, bot restart, signal, direct exchange
  call, or event was manufactured.

## Previous Deployed Baseline (PR #1314)

- PR #1314 merged as canonical `0dbfbca74b029353a0e11888e71077fa711835ff`.
  It records immutable runtime/Rust/config provenance and fill-to-runtime
  attribution without changing strategy, order, risk, or exchange behavior.
  PR #1319 had already merged as `84c8e040334820ccc049787c82048358e18179c6`
  with its offline fake-live shutdown-clock repair and required no VPS action.
- VPS5 fast-forwarded tracked-clean from `eb82e256c2` to the exact PR #1314
  merge. A deliberately wrong Rust fingerprint failed closed before build or
  signal. The exact target-derived fingerprint
  `691bff9683deec9382a4e96ab6a107c14145f88edd6ae2f8e2380b8ba6824449`
  then rebuilt and verified the loaded extension after restoring the explicit
  `/root/.cargo/bin` non-login PATH.
- The guarded runner stopped, exited, relaunched, and verified all five exact
  panes without force. Bot PIDs
  `1042130/1042139/1042133/1042142/1042136` became
  `1044483/1044492/1044486/1044495/1044489`. The exact
  `1784414260464..1784414921350` window selected 10/1,011 event segments and
  `73414648` bytes with complete lifecycle, repository, target, monitor, and
  text-log evidence, but correctly remained red on one natural KuCoin
  authoritative-refresh `RequestTimeout` at `1784414448511`.
- KuCoin subsequently emitted readiness evidence and successful authoritative
  calls without intervention. The exact settled
  `1784414681768..1784415199226` report was hard-green with zero hard problem
  events, log matches, monitor errors, or process failures. Final target
  sampling was 3/3 stable with states `R=3,S=2`, no extras or issues, and
  protected `misc:0.0` remained `%8`/PID `434835`. No direct exchange request
  or event was manufactured.

## Previous Deployed Baseline (PRs #1318 And #1317)

- PR #1318 merged as canonical `8ca7f034bce7ffaaa99800590c88651de4d267c5`.
  It adds bounded redacted structured-event artifacts and post-panic planning
  availability proof to the offline fake-live harness. No VPS action was
  warranted because it changes no live producer, runtime, exchange, order,
  risk, or configuration behavior.
- PR #1317 merged as canonical `eb82e256c2dfeac29af158f389f93a7ddba8eae2`.
  It adds bounded Hyperliquid unified-account composition diagnostics without
  changing scalar balance, exchange calls, planning, orders, risk, or console
  admission.
- VPS5 fast-forwarded tracked-clean from `e7fe7f79` without a Rust rebuild.
  The exact panes `%358`-`%362` gracefully restarted without force; old PIDs
  `1040903/1040911/1040905/1040914/1040908` became
  `1042130/1042139/1042133/1042142/1042136`.
- The exact `1784407239491..1784407888217` window selected 10/1,011 event
  segments totaling `72553076` bytes, retained all five
  stopping/stopped/startup cohorts, and had zero hard, monitor, text-log,
  repository, or target failures. Delayed 3/3 sampling recovered one transient
  GateIO `D` observation to final `R=4,S=1`, with five stable PIDs and no extras
  or issues. Protected `misc:0.0` stayed `%8`/PID `434835`. No direct exchange
  call or event was manufactured.

## Previous Deployed Baseline (PR #1316)

- PR #1316 merged as canonical `e7fe7f796fb76a829003933dc7e5d937c6df8c64`.
  It adds bounded Binance CCXT unified composition diagnostics while preserving
  scalar balance, exchange-call, planning, order, risk, and console-admission
  behavior.
- VPS5 fast-forwarded tracked-clean from `a0db60f9` without a Rust rebuild.
  The exact panes `%358`-`%362` gracefully restarted without force; old PIDs
  `1038760/1038769/1038763/1038772/1038766` became
  `1040903/1040911/1040905/1040914/1040908`.
- The exact `1784404124757..1784404777859` window selected 7/1,012 event files
  totaling `19420705` bytes, retained five stopping/stopped/startup cohorts, and
  had zero hard, monitor, or text-log issues. Delayed target sampling was 3/3
  with `R=4,S=1` and no extras; `misc:0.0` stayed `%8`/PID `434835`. No direct
  exchange call or event was manufactured.

## Previous Deployed Baseline (PR #1313)

- PR #1313 merged as canonical `a0db60f9ca97dbc5b9b37aa3230fce97eb0917ce`.
  It adds the bounded OKX-first balance-composition diagnostic and preserves
  scalar balance, exchange-call, planning, order, risk, and console-admission
  behavior.
- VPS5 fast-forwarded cleanly from `32156cbc`. The first guarded preparation
  failed closed after the repository move because the caller supplied a Rust
  fingerprint from a different local untracked `Cargo.lock`; no bot was
  signalled. A same-head target-derived proof then passed with no Rust rebuild
  and exact source/stamp/runtime-artifact agreement.
- The authorized exact-pane executor gracefully stopped, exited, relaunched,
  and verified only `%358/%359/%360/%361/%362`; old bot PIDs
  `1036076/1036085/1036080/1036088/1036082` became
  `1038760/1038769/1038763/1038772/1038766` with no force signal. The exact
  lifecycle window `1784401342000..1784402056447` selected 6/1,011 managed
  segments totaling `39263703` bytes, retained all five stopping/stopped/startup
  cohorts, and returned zero hard, monitor, text-log, repository, or target
  failures. A wider pre-action window correctly retained one earlier natural
  KuCoin `RequestTimeout` instead of hiding it.
- A natural post-restart OKX `balance.changed` event carried two bounded
  connector-proven asset rows with no raw payload. Final targets remained 3/3
  stable and exact, the checkout remained tracked-clean, and protected
  `misc:0.0` stayed `%8`/PID `434835`. No direct exchange request or event was
  manufactured.
- The merge is the exact base for this slice.

## Previous Deployed Baseline

- PR #1311 merged as canonical `32156cbc251d666902f20b8b000a9a1dfe05a0a2`.
  It carries bounded hard-only problem evidence into smoke summary, brief, and
  incident-bundle metadata without changing runtime event production or
  trading behavior. VPS5 prepared the exact clean merge without a Rust rebuild
  or bot restart; one immediate natural KuCoin positions timeout was retained,
  the settled smoke was hard-green, and all five bot panes plus `misc:0.0`
  remained unchanged.
- PR #1310 merged as canonical `5d06887b78c2790efd15e1bd67bae6b3f5d96636`.
  It added full-report `hard_problem_events` with authoritative `count`,
  bounded chronological `sample`, and explicit `retained`/`truncated` counts;
  the concise/brief and incident-bundle projections were unchanged. VPS5
  prepared the exact clean merge without a Rust rebuild or bot restart, and a
  bounded read-only smoke was hard-green with
  `hard_problem_events={count:0,retained:0,truncated:0,sample:[]}`; all five
  pane parents and `misc:0.0` remained unchanged.
- Canonical `master` and VPS5 are
  `f1ae7970393e8299d1b0a98c8ff68d42adddd2d0` after PR #1299. The clean
  checkout fast-forwarded from PR #1309 without a Rust rebuild after five
  targets passed 3/3 stable preflight samples with no extras or issues. The
  authorized exact-pane restart stopped, exited, relaunched, and verified all
  five targets without force. Its bounded
  `1784392681320..1784393372572` window selected 10/1,011 managed segments
  totaling `66399577` bytes, recovered all five stopping, stopped, and startup
  cohorts, and returned zero hard, monitor, text-log, repository, or target
  failures. The checkout remained exact and tracked-clean; all five pane
  parents and protected `misc:0.0` `%8`/PID `434835` remained stable. No direct
  exchange probe or event was manufactured.
- PR #1309 merged as
  `50c37db6049206634b62f45798a8b240a035e3b5` after exact-current-head Hermes
  approval and green Python/Rust CI. It removed raw exception text from
  `sink.degraded` while preserving stable sink and exception classifications.
  VPS5 prepared the exact merge without a Rust rebuild and gracefully restarted
  all five exact panes. The bounded `1784339097380..1784339763543` window
  recovered complete shutdown/startup cohorts and left every bot running, but
  correctly remained red on two hard structured problem events including a
  real KuCoin positions-fetch `RequestTimeout`. Only one hard classification
  remained visible in the bounded mixed sample, exposing the active
  hard-evidence-retention follow-up. Repository, target, monitor, and text-log
  gates stayed green, tracked state stayed clean, and `misc:0.0` remained
  unchanged.
- Canonical `master` and VPS5 are
  `9a5e3585d0c5641a14c2c359acd01e7f3e74bf7d` after PR #1308. Exact-head Hermes
  and Python/Rust CI were green. Exact repository preparation fast-forwarded
  cleanly without a Rust rebuild, and independent preflight resolved all five
  targets with 3/3 stable samples and no extras or issues. The authorized
  restart stopped, exited, relaunched, and verified all five exact panes without
  force. Its bounded `1784335954292..1784336611652` window selected 6/1,012
  managed segments totaling `21328854` bytes, recovered all five shutdown and
  startup cohorts, and returned zero hard, monitor, text-log, or target
  failures. All five bots remained stable, the checkout stayed exact and clean,
  and `misc:0.0` retained its pre-restart pane/PID identity. No direct exchange
  probe or event was manufactured.
- Canonical `master` and VPS5 are
  `8aefdbc82339b756ff642e726ae0924d5ca8774d` after PR #1307. Exact-head Hermes
  and Python/Rust CI were green. Exact repository preparation fast-forwarded
  cleanly without a Rust rebuild; independent preflight resolved all five
  targets with 3/3 stable samples and no extras or issues.
- The merged orchestrator gracefully stopped, observed exit, relaunched, and
  verified all five exact targets with no force signal. The bounded window
  `1784332307933..1784332994590` selected 6/1,012 managed event segments
  totaling `19335163` projected bytes and recovered five stopping, five stopped,
  and five startup cohorts. Repository, target, monitor, and text-log gates were
  green, but smoke correctly remained red on one real KuCoin positions-fetch
  `RequestTimeout`; a second later timeout showed recovery was not yet proven.
  All bots were left running. Pane parents stayed unchanged, tracked state
  stayed clean, and `misc:0.0` remained `%8`, PID `434835`. No direct exchange
  probe or event was manufactured. The retained raw exception string exposed
  the active redaction follow-up.
- Canonical `master` and VPS5 are
  `0d1b06b82f3bab011e29a350b4a5276c2ebd5356` after PR #1306. Exact-head Hermes
  and Python/Rust CI were green. VPS5 first fast-forwarded to make the
  repository-preparation tool available, without a bot restart or signal. Its
  same-head execution returned green with no repository move or Rust build and
  exact source/stamp/final fingerprints; a valid but wrong target commit failed
  with `fetched_target_head_mismatch` before any build. All five configured pane
  parents and unrelated `misc:0.0` remained unchanged, and tracked state stayed
  clean.
- PR #1305 previously deployed at
  `300fdd703fee9e1ce0e9c54df43bb7b1dcb858d8`. Exact-head Hermes
  and Python/Rust CI were green. VPS5 fast-forwarded without a bot restart or
  signal. The merged collector selected 10/1,008 managed event segments for the
  exact retained bounds `1784316350000..1784317500000`, projected
  `131834602` scan bytes under the 128 MiB cap, and recovered five stopping,
  five stopped, and five startup cohorts with zero hard failures. A malformed
  expected head exited 2 before collection. All five bot pane parents stayed
  `856294/856332/856364/856398/856434`, tracked state stayed clean, and
  `misc:0.0` remained `%8`, PID `434835`.
- PR #1303 previously deployed at
  `9e8d1343e0f1f43fc3207d611a8b06d88af8b6c0`. Its first complete-archive
  collector run was interrupted after more than ten CPU-active minutes by
  exact collector PID only. A read-only prototype selected 10/1,012 segments
  and recovered the complete lifecycle in 37.3 seconds, triggering PR #1305.
- PR #1302 previously deployed at
  `0b5503b2a9ee4817618b7aca25dab417af4292dd`. The retained PR #1296 window
  evaluated green with exact bounds `1784316350000..1784317500000`; a
  one-millisecond mismatch and one dropped hard-looking line each failed with
  `log_scan_invalid` while panes and `misc:0.0` remained unchanged.
- PR #1301 previously deployed at
  `46a28795dec40acbee0dbaa3602be955bbecf23e`. Exact-head Hermes
  and Python/Rust CI were green. VPS5 fast-forwarded without a restart or
  signal. The exact PR #1296 shutdown-through-startup window evaluated green
  with 3/3 stable targets, five stopping/stopped lifecycles, startup timing for
  five bots, 24 bounded monitor segments, eight text logs, and zero hard
  failures. A wider three-hour report correctly evaluated red on 30 real hard
  events. Final pane parents and bot PIDs were unchanged; the checkout remained
  tracked-clean and `misc:0.0` remained `%8`, PID `434835`. Validation exposed
  the lossy timestamp projection addressed by the active follow-up.
- PR #1300 previously deployed at
  `e1a4837914c1e4768cd7963bba47212499d32937`. Exact-head Hermes
  and Python/Rust CI were green. VPS5 fast-forwarded without a restart or
  signal; a deliberately wrong expected Rust fingerprint failed before target
  sampling with `action_started=false`, and the final 3/3 target report retained
  the same five PIDs, zero extras or issues, and exact states `R=3,S=2`.
  Repository state stayed tracked-clean and unrelated `misc:0.0` remained `%8`,
  PID `434835`.
- PR #1296 previously fast-forwarded VPS5 cleanly from `4a7a6753` and the
  exact local executor gracefully restarted only panes
  `%358/%359/%360/%361/%362`. Old PIDs
  `1015403/1015406/1015410/1015412/1015414` exited; replacement PIDs
  `1019670/1019679/1019673/1019681/1019676` retained the same pane parents and
  private supervisor fingerprint. The executor and independent settled target
  reports were hard-green with 3/3 stable samples, all five targets
  relaunch-ready, zero extras or duplicates, and zero issues. VPS5 remained
  tracked-clean; unrelated `misc:0.0` remained `%8`, PID `434835`. No direct
  exchange probe or event was manufactured.
- PR #1298 merged as `4a7a6753bff00f9b8749d9707f9bdccc4b3a5ffc`. Exact-head Hermes and
  both CI jobs were green. VPS5 fast-forwarded cleanly from `7b833471` without a
  restart or process signal. A deliberately wrong expected head returned exit
  1 with `action_started=false`, no target preflight, zero tracked changes, and
  a source-matched Rust extension. The post-deploy 3/3 target report was
  hard-green with all five unchanged bot PIDs relaunch-ready, exact states
  `R=3,S=2`, zero issues, and the unchanged private command fingerprint.
  Unrelated `misc:0.0` remained `%8`, PID `434835`.
- PR #1294 first
  fast-forwarded cleanly as `3de024c76d5c07bda2b4e64400c1a204d6be38a8`
  and produced a hard-green 3/3 stable target report with all five targets,
  zero issues, fingerprint
  `01d200d4a38c5c85a2123b5210224a18cdef08d0a8be3efc48edb4a159fc5db4`,
  and no command exposure. PR #1293 then fast-forwarded cleanly without a Rust
  rebuild.
- The five configured bot panes were restarted to load PR #1293. Exact panes
  `%358/%359/%360/%361/%362` and pane parents
  `856294/856332/856364/856398/856434` remained stable. Old bot PIDs
  `1013205/1013207/1013209/1013210/1013211` exited after one exact-pane Ctrl-C
  round. Replacement PIDs `1015403/1015406/1015410/1015412/1015414` matched the
  same exact ownership contract and retained the full-command fingerprint.
  Unrelated `misc:0.0` remained `%8`, PID `434835`.
- The immediate target report and smoke were `ok=true` with zero hard failures,
  five stable exact processes, and successful natural account/fill activity. A
  later bounded window correctly remained red after two real KuCoin
  account-state `RequestTimeout` failures and one degraded cycle; process
  sampling observed all temporary `D` states recover without PID churn. The
  fresh recovery smoke was `ok=true` with zero hard failures, `326/326` remote
  calls, `61/61` account-critical calls, and nine successful fill refreshes.
  A quiet exact-PID sample reached `R=4,S=1`; the final target report retained
  3/3 stable samples, five relaunch-ready targets, zero issues, and the same
  fingerprint. No direct exchange probe or event was manufactured.

- Canonical `master` and VPS5 are
  `b490bd75bebd0228628c1b628b46d3f3ac52cee4`, PR #1290. VPS5 fast-forwarded
  cleanly from `d6cac101` without restart or signal; exact pane IDs/PIDs, bot
  PIDs `985592/985594/985596/985598/985600`, and unrelated `misc:0.0` PID
  `434835` remained unchanged.
- Immediate and settled three-sample local-only target reports were `ok=true`
  and `sampling.stable=true`, with all five resolved targets relaunch-ready,
  zero failed samples, zero changed targets, and no issues. The compact plan was
  `ok=true`, had zero issues, and required the same all-targets-ready verdict.
  A bounded process report observed real temporary I/O waits without PID churn
  or hard failures; the quiet exact-PID check cleared to `R=4,S=1`. No exchange
  request, process action, or event was manufactured.
- Canonical `master` and VPS5 are
  `d6cac1017b27c56c2dca57c9ca0f637cd300dac2`, PR #1289. VPS5 fast-forwarded
  cleanly from `e004ede7` without restart or signal; exact pane IDs/PIDs, bot
  PIDs, and unrelated `misc:0.0` PID `434835` remained unchanged.
- The compact plan was `ok=true` for five bots with zero issues, an explicitly
  configured three-sample/five-second exact target command, and execution still
  unavailable. A corrected four-sample local-only process smoke was `ok=true`
  with five stable PIDs, zero hard/config/process failures, one non-persistent
  final `D`, and no PID churn; the settled exact sample was `R=3,S=2`. No
  exchange request, process action, or event was manufactured.
- Canonical `master` and VPS5 are
  `e004ede7ddaa4935bc8bf69739285724656198d3`, PR #1288. VPS5 fast-forwarded
  cleanly from `6cb8bc3c` without a restart or process signal; all five pane
  IDs/PIDs, bot PIDs, and unrelated `misc:0.0` PID `434835` remained unchanged.
- Immediate and settled three-sample local-only target reports were `ok=true`
  and `sampling.stable=true`, with `3/3` successful samples, five stable
  targets, zero failed samples, zero changed targets, and no issues. Immediate
  process state `R=2,S=3` settled to `R=4,S=1`; no exchange request, process
  action, or event was manufactured.
- Canonical `master` and VPS5 are
  `6cb8bc3c73c1e5b8c7ed1e70fa953e07a8e6b85e`, PR #1287. VPS5 fast-forwarded
  from `7a12430a` with a tracked-clean checkout. Configured pane process IDs
  `856294/856332/856364/856398/856434`, exact bot PIDs
  `985592/985594/985596/985598/985600`, and unrelated `misc:0.0` PID `434835`
  remained unchanged; no restart or process signal occurred.
- Immediate and settled local-only target reports were `ok=true` with all five
  expected windows resolved to canonical pane IDs `%358/%359/%360/%361/%362`,
  exact PPID-to-pane-PID ownership, and zero missing, duplicate, extra, config,
  or scan failures. The immediate `D=1,R=2,S=2` process snapshot naturally
  settled to `R=5`; all pane and bot PIDs remained unchanged. No exchange
  request, process action, or event was manufactured.
- Immediate and settled four-sample `--brief` process reports were `ok=true`
  with all five expected commands/configs matched, five stable PIDs, and zero
  missing, duplicate, extra, config, or scan failures. The first naturally
  ended with one active but non-persistent `D`; the settled repeat observed
  recovery and ended `R=3,S=2` with zero active or persistent uninterruptible
  processes. No exchange request, process action, or event was manufactured.
- The immediate four-sample local-only process report was `ok=true`: all five
  expected commands/configs matched, with zero missing, duplicate, extra,
  config, or scan failures and five stable PIDs. It naturally observed one
  recovered Binance `D` sample and a GateIO `D` final sample. The settled
  four-sample report was also `ok=true`, retained five stable PIDs and zero
  hard failures, naturally observed GateIO recovery, and ended in `R=5` with
  zero active or persistent uninterruptible processes. No exchange request,
  process action, or event was manufactured.
- PR #1283's merged process sampler passed current-head Python/Rust CI and 132
  focused local smoke/incident/docs tests. The requested VPS5 full smoke command
  was not executed after the production-action approval layer rejected it as a
  possible authenticated exchange probe. The rejection was preserved: no retry,
  bypass, authenticated exchange request, process action, or manufactured event
  occurred. PR #1285's local-only process-report slice separated the remaining
  validation from monitor/log and exchange-adjacent command paths.
- PR #1282 required no restart or process signal. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged. The immediate focused report
  projected 37 completions and two degradations across all five bots and
  correctly retained a latest KuCoin `InvalidNonce` degradation. The fresh
  two-minute smoke was `ok=true` with zero hard failures, `193/194` remote
  calls, `56/57` account-critical calls, `9/9` fill refreshes, and five exact
  processes/configs matched.
- The settled focused report was `ok=true` with zero hard failures and 20
  completions plus one retained degradation. All five latest outcomes were
  successful, and KuCoin was explicitly counted as completed after its latest
  degradation. A final exact process check caught that same KuCoin PID in `D`
  for about 20 seconds before three consecutive `R` samples, exposing the
  active process-sampling follow-up. No event or trading activity was
  manufactured.
- PR #1280 required no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged. The immediate two-minute smoke
  was `ok=true` with `197/197` remote calls, `54/54` account-critical calls,
  `9/9` fill refreshes, and five exact/config-valid processes.
- A focused five-minute report naturally projected 34 snapshots across all five
  bots, with zero missing or stale markets. An intermediate fresh smoke caught
  a real KuCoin balance/positions/open-orders `RequestTimeout` and correctly
  remained hard-red; all processes were already in `R`, later calls succeeded,
  and the event was not hidden. The final fresh two-minute smoke was `ok=true`
  with zero hard/log/monitor/process failures, `242/242` remote calls, `55/55`
  account-critical calls, `12/12` fill refreshes, and 16 snapshots across all
  five bots with zero missing or stale markets. No event or trading activity
  was manufactured.
- PR #1278 required no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged. Immediate and settled bounded
  two-minute smokes were `ok=true` with zero hard/log/monitor/process failures,
  `170/170` and `173/173` remote calls successful, `45/45` and `47/47`
  account-critical calls successful, and `5/5` and `9/9` fill refreshes
  successful. A transient immediate `D=1,R=1,S=3` sample settled to all five
  exact/config-valid processes in state `R` with no uninterruptible sleep.
- The focused five-minute data-packet report naturally projected 110 events
  across all five bots as 15 latest packet observations, three packet kinds per
  bot. All were fresh and quality `ok`, with zero latest warning/error packets.
  The same natural inventory contained 35 `snapshot.built` events, exposing the
  active follow-up. No event or trading activity was manufactured.
- PR #1274 required no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged. The fresh two-minute smoke was
  `ok=true` with zero hard/log/monitor/process failures, `248/248` remote and
  `57/57` account-critical calls successful, `8/8` fill refreshes successful,
  and all five processes/configs matched in state `R` with no uninterruptible
  sleep.
- One natural KuCoin `forager.eligibility_changed` event appeared in a bounded
  30-minute window and the scoped monitor inventory counted exactly one. A
  fresh two-minute focused report contained no eligibility section and omitted
  that event type from inventory, directly validating the requested-window
  boundary. The historical 30-minute report retained four hard failures and
  was not represented as deploy-green. No event or trading activity was
  manufactured.
- PR #1273 required no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged. The settled two-minute smoke was
  `ok=true` with zero hard/log/monitor/process failures, `210/210` remote and
  `55/55` account-critical calls successful, `9/9` fill refreshes successful,
  and all five processes/configs matched. One report-time OKX `D` sample
  cleared after ten seconds; the final exact states were `R=4,S=1`.
- The focused current-plus-rotated 180-minute eligibility report found six
  natural `forager.eligibility_changed` events across all five bots, including
  two KuCoin events, and returned bounded redacted symbol samples. Its 17 hard
  failures belonged to the historical validation window and were not used as
  the deploy-health verdict. The fresh two-minute smoke was hard-green. No
  event or trading activity was manufactured.
- PR #1272 required no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged. The settled two-minute smoke was
  `ok=true` with zero hard/log/monitor/process failures, `279/279` remote and
  `44/44` account-critical calls successful, `7/7` fill refreshes successful,
  and all five processes/configs matched in states `R=3,S=2` with no
  uninterruptible sleep. Twelve non-hard EMA readiness events remained durable.
- The previously failing absent `forager_features` selector returned a valid
  base-only report. The focused staged-readiness section returned a valid
  zero-event projection. No natural `planning.defer_summary` occurred in the
  bounded window, and no event or trading activity was manufactured.
- PR #1271 required no restart. Exact bot PIDs
  `985592/985594/985596/985598/985600`, all five pane parents, and unrelated
  `misc:0.0` PID `434835` remained unchanged. The settled two-minute smoke was
  `ok=true` with zero hard/log/monitor/process/pipeline failures, `186/186`
  remote and `55/55` account-critical calls successful, `9/9` fill refreshes
  successful, and all five processes/configs matched in states `R=4,S=1`
  with no uninterruptible sleep. Fourteen non-hard EMA readiness events
  remained durable.
- No `forager.feature_unavailable` event occurred naturally in the bounded
  window. The zero-event section probe exposed the registered-selector defect
  addressed by the active slice; no event or trading activity was manufactured.
- PR #1270 required no restart. VPS5 fast-forwarded cleanly while exact bot
  PIDs `985592/985594/985596/985598/985600`, all five pane parents, and
  unrelated `misc:0.0` PID `434835` remained unchanged.
- A bounded performance report was `ok=true` with zero errors or warnings and
  preserved the legacy no-budget shape for naturally unconfigured startup
  events. The two-minute smoke was `ok=true` with zero hard, log, monitor,
  process, or event-pipeline failures, `147/147` remote and `37/37`
  account-critical calls successful, `9/9` fill refreshes successful, all five
  processes/configs matched in states `R=4,S=1`, and no uninterruptible sleep.
  Fifteen non-hard attention events remained durable: eleven EMA readiness,
  two staged-cycle degradation, and two websocket reconnect events.
- PR #1269 was activated with one SIGINT at `2026-07-16T11:59:17Z` to exact
  old PIDs `979190/979193/979196/979199/979202`; all exited naturally within
  36 seconds. New PIDs are `985592/985594/985596/985598/985600`; all five pane
  parents and unrelated `misc:0.0` PID `434835` remained unchanged.
- Immediate smoke was hard-green. One later window retained a real recovered
  KuCoin fill-refresh timeout and was not treated as deployment green. A fresh
  two-minute settled window was `ok=true` with zero hard, log, monitor,
  process, or event-pipeline failures, `46/46` account-critical calls and
  `7/7` fill refreshes successful, all five expected processes/configs
  matched, states `R=4,S=1`, no uninterruptible sleep, and no active HSL
  replay. Two non-hard candle timeouts remained durable.
- VPS5 intentionally has no configured `startup_phase_budgets`. Reports showed
  `no_baseline`, with zero invalid configured assessments; no budget event,
  exchange state, or trading activity was manufactured for validation.
- PR #1268 required no restart. Its bounded aggregate-only inventory scanned
  40 of 1,182 discovered files and 8,153,519 bytes, reported ten positive and
  25 truncated files, skipped six symlinks, and had zero unreadable files or
  discovery errors. It retained the #1267 class counts of 144 secret-query and
  143 private-websocket-query matches without per-file paths, ages, or hashes.
  All five configured bot panes and unrelated `misc:0.0` remained present; no
  process signal or remediation occurred.
- PR #1267 required no restart. Its 40-file, 250,000-byte bounded inventory
  found 144 secret-query matches versus 143 private-websocket-query matches,
  proving one additional non-websocket query fragment was classified. The scan
  discovered 1,182 files, skipped six symlinks, and had zero discovery or read
  errors. No values or source lines were emitted and no remediation occurred.
  All five configured bot panes and unrelated `misc:0.0` remained present; no
  process signal was sent.
- PR #1266 required no restart. A four-hour, current-segment-only brief report
  scanned six monitor files and 19,163 records with zero monitor parse errors,
  and emitted the new startup budget coverage fields. Startup timing rows had
  already rotated out, so all coverage counts were naturally zero. The
  section-only report retained two hard and 319 attention problem events and
  was not represented as a green deploy-health verdict. Exact bot and
  `misc:0.0` PIDs remained unchanged; a transient GateIO `D` sample cleared to
  `R` after 20 seconds.
- PR #1265 required no restart. Its bounded read-only VPS5 dry run scanned 40
  of 1,182 discovered files with a 250,000-byte decompressed cap per file,
  skipped six symlinks, and reported zero discovery or read errors. Ten
  retained May logs contained private websocket/query credential classes; no
  values or source lines were emitted. No remediation was attempted. The five
  bot PIDs and unrelated `misc:0.0` PID remained unchanged.
- PR #1264 was activated with one SIGINT at `2026-07-16T08:34:29Z` to old
  PIDs `977722/977725/977728/977731/977734`; all exited naturally within 23
  seconds. The settled two-minute smoke was `ok=true` with zero hard, log,
  monitor, process, or event-pipeline failures, `281/281` remote and `28/28`
  account-critical calls successful, six successful fill refreshes, and five
  exact processes in state `R`. All five fresh logs contained zero INFO
  matches for successful maintainer-stop, OHLCV-watcher-stop, or hourly-jitter
  detail. No cancellation failure occurred naturally, and none was
  manufactured.
- PR #1263 was activated with one SIGINT at `2026-07-16T07:54:15Z` to old
  PIDs `975507/975510/975511/975513/975515`; all exited naturally by
  `07:54:37Z`. The settled smoke was `ok=true` with zero hard, log, monitor,
  process, or event-pipeline failures, `251/252` remote and all `37/37`
  account-critical calls successful, seven successful fill refreshes, and five
  exact processes in states `R=4,S=1`. One non-hard KuCoin candle timeout was
  retained. Every bot emitted exactly one `[bot] started`, one
  `phase=startup-ready`, zero removed lifecycle INFO lines, and a durable
  `bot.ready` event; Hyperliquid also completed full warmup in the bounded
  window.
- PR #1262 merged as `82a56bb15445a1effff8501aa9f66540009b8f3f` and was
  activated with one SIGINT at `2026-07-16T06:54:33Z` to old PIDs
  `974157/974160/974161/974163/974165`; all exited naturally by `06:54:56Z`.
  An immediate KuCoin authoritative timeout recovered, and settled smoke was
  `ok=true` with zero hard failures, `220/222` remote and `43/43`
  account-critical calls successful, six fill refreshes successful, all three
  observed HSL replays complete, and five exact processes in state `R`.
- Eleven natural durable `entry.initial_distance_gate_blocked` events produced
  three console representatives (one each on Binance, GateIO, and OKX), with
  eight structured events suppressed; `active_max=4` and `suppressed_max=3`.
- PR #1261 was activated after one exact five-bot SIGINT at `06:18:37Z`; old
  PIDs `973301/973303/973305/973307/973309` all exited naturally by `06:19:12Z`
  without escalation. Natural HSL warning lengths were 224-228 characters, with
  `0/287` above 240. The final two-minute smoke was `ok=true` with `202/202`
  remote and `49/49` account-critical calls successful, nine fill refreshes,
  HSL replay complete, states `R=4,S=1`, zero hard/log/monitor/process/pipeline
  failures, and a clean head. Transient startup KuCoin `RequestTimeout` and
  Binance `InvalidNonce` recovered naturally.
- PR #1260 was activated after one exact five-bot SIGINT round; old PIDs
  `972601/972603/972605/972607/972609` all exited naturally within 24 seconds
  without escalation. The final settled smoke was `ok=true` with zero
  hard/log/monitor/process failures, `371/371` remote and `80/80`
  account-critical calls successful, 11 successful fill refreshes, no active
  HSL replay, all five exact processes in normal `R/S` states, and a clean
  tracked checkout. All five bots naturally emitted the new structured
  approved-coin membership projection at 130-150 visible characters, versus
  245-250 before, with zero legacy membership lines. The same exact new logs
  exposed one 241-character Binance startup HSL safety warning.
- PR #1259 was activated after one exact five-bot SIGINT round; old PIDs
  `971695/971697/971699/971701/971703` all exited naturally within 28 seconds
  without escalation. The settled three-minute smoke was `ok=true` with zero
  hard/log/monitor/process failures, `483/486` remote and `49/50`
  account-critical calls successful, seven successful fill refreshes, all
  three observed HSL replays completed, five matching/config-valid processes,
  and a clean tracked checkout. The three recoverable remote failures were one
  Binance `InvalidNonce` and two KuCoin candle timeouts. The natural Binance
  clock-offset recovery line measured 203 visible characters, retained stable
  recovery facts, and contained no raw exception text. A transient report-time
  `D` sample cleared on direct process inspection. The same exact new logs
  exposed natural approved-coin membership lines at 245-250 characters.
- PR #1258 was activated after one exact five-bot SIGINT round; old PIDs
  `970778/970780/970782/970784/970786` all exited naturally within 24 seconds
  without escalation. The settled two-minute smoke was `ok=true` with zero
  hard/log/monitor/process failures, `317/317` remote and `31/31`
  account-critical calls successful, six successful fill refreshes, no active
  HSL replay, all five exact processes in normal `R/S` states, and a clean
  tracked checkout. One natural post-ready KuCoin
  `ccxt_fetch_ohlcv_failed` warning measured 201 visible characters with zero
  legacy caller-bearing duplicates. The same exact logs exposed one natural
  post-ready clock-offset recovery warning at 266 visible characters that
  retained raw exception text.
- PR #1257 was activated after one exact five-bot SIGINT round; old PIDs
  `968739/968741/968743/968745/968747` all exited naturally within 16 seconds
  without escalation. The settled two-minute smoke was `ok=true` with zero
  hard/log/monitor/process failures, `241/241` remote and `58/58`
  account-critical calls successful, nine successful fill refreshes, no active
  HSL replay, all five exact processes in state `R`, and a clean tracked
  checkout. No candidate-required-EMA condition occurred naturally in the new
  segments, so no compact target line was manufactured; zero legacy summary
  duplicates appeared. The exact new logs then exposed two natural post-ready
  KuCoin candle-fetch retry warnings at 256-257 visible characters.
- PR #1256 was activated after one exact five-bot SIGINT round; old PIDs
  `967753/967755/967757/967759/967760` all exited naturally within eight
  seconds without escalation. Four natural periodic health lines on Binance,
  GateIO, OKX, and Hyperliquid measured 202-209 visible characters and retained
  uptime, loop, position, balance, order/fill/error, RSS, and slow-phase
  context. A settled one-minute report was `ok=true` with zero hard failures,
  `138/138` remote and `29/29` account-critical calls successful, three
  successful fill refreshes, no active HSL replay, and a clean tracked
  checkout. A bounded ten-sample process series showed brief rotating
  filesystem journal/page waits, four all-normal samples, and no persistently
  blocked PID. The exact new logs then exposed required-EMA-unavailable
  warnings at 374-447 visible characters on four bots.
- PR #1255 was activated after one exact five-bot SIGINT round; all old bots
  exited naturally without escalation. Three natural forager selection lines
  on Binance, GateIO, and OKX measured 182-185 visible characters, retained
  all three selected symbols, hysteresis, reason, the top three scores, and
  the omitted-score count, and none exceeded 240. The final settled smoke was
  `ok=true` with zero hard failures, `248/248` remote and `49/49`
  account-critical calls successful, eight successful fill refreshes, no
  active HSL replay, all five exact processes in state `R`, and a clean tracked
  checkout. The exact new logs then exposed natural periodic health summaries
  at 244-248 visible characters on four bots.
- PR #1254 was activated after one exact five-bot SIGINT round; all old bots
  exited naturally within eight seconds and no escalation was used. Four
  natural candle-health summaries on Binance, GateIO, OKX, and Hyperliquid
  measured 156-211 visible characters, retained aggregate counts, whole
  samples, and `+N more`, and none exceeded 240. The final two-minute smoke was
  `ok=true` with zero hard/log/monitor/process failures, `301/301` remote and
  `33/33` account-critical calls successful, seven successful fill refreshes,
  no active HSL replay, five matching/config-valid processes in normal `R/S`
  states, and a clean tracked checkout. The exact new logs then exposed
  natural forager selection lines at 247-250 characters on Binance, GateIO,
  and OKX.
- PR #1253 was activated after one exact five-bot SIGINT round; all old bots
  exited naturally within four seconds and no escalation was used. Natural
  initial-entry distance-gate records appeared 16 times across Binance,
  GateIO, and OKX at 181-190 visible characters, with zero records above 240
  and zero legacy duplicates. The initial window included active expected HSL
  replay and one recoverable KuCoin authoritative-state timeout cycle. The
  final two-minute smoke was `ok=true` with zero hard/log/monitor/process
  failures, `52/52` account-critical calls successful, eight successful fill
  refreshes, no active HSL replay, five matching/config-valid processes in
  normal `R/S` states, and a clean tracked checkout. One non-account-critical
  KuCoin OHLCV fetch timeout remained non-hard evidence. The exact new logs
  then exposed natural candle-health summaries at 257-263 characters on
  Binance, GateIO, and OKX.
- PR #1252 was activated after one exact five-bot SIGINT round; all old bots
  exited naturally within ten seconds and no escalation was used. GateIO's
  first new process then failed closed during HSL replay because one required
  held-pair unrealized-PnL minute was unavailable. One bounded exact-pane retry
  succeeded, reached `bot.ready`, and completed replay. Natural compact
  close-EMA warnings measured 158, 166, and 225 visible characters on OKX,
  Binance, and GateIO, with zero legacy duplicates in the post-deploy files.
  The final two-minute smoke was `ok=true` with zero hard/log/monitor/process/
  pipeline failures, `347/347` remote and `53/53` account-critical calls
  successful, seven successful fill refreshes, no active HSL replay, five
  config-valid processes in state `R`, and a clean tracked checkout.
- PR #1251 was activated with one exact five-bot graceful restart. Every old
  bot exited naturally within ten seconds after one SIGINT round; no escalation
  was used. Three natural admitted slow-refresh lines measured 155-171 visible
  characters, versus 252-305 before, while retaining plans, wall and surface
  timings, parallel execution, and material residuals. Zero legacy duplicate
  lines appeared. The final two-minute smoke was `ok=true` with zero hard/log/
  monitor/process/pipeline failures, `202/203` remote and `46/46`
  account-critical calls successful, six successful fill refreshes, no active
  HSL replay, five config-valid processes, and a clean tracked checkout. The
  single remote failure was non-account-critical; one report-time `D` sample
  cleared to exact `R` on all five processes.
- PR #1250 was activated with one exact five-bot graceful restart. Every old
  bot exited naturally within 24 seconds after one SIGINT round; no escalation
  was used. Natural forager HSL settings lines measured 163-167 visible
  characters, versus 310-314 before, while retaining every configured setting.
  One KuCoin timeout cycle and active GateIO replay appeared in the first
  startup window and recovered naturally. The final two-minute smoke was
  `ok=true` with zero hard/log/monitor/process/pipeline failures, `43/43`
  account-critical calls successful, eight successful fill refreshes, no
  active HSL replay, five config-valid processes in states `R=4,S=1`, and a
  clean tracked checkout. One non-account-critical remote failure remained
  non-hard evidence.
- PR #1249 was activated with one exact five-bot graceful restart. Every old
  bot exited naturally within 24 seconds after one SIGINT round; no escalation
  was used. Eleven logical lock-hold warnings then occurred naturally at
  205-218 visible characters, versus 346-349 before the change, while retaining
  per-symbol scope and compact holder identity/timing. Immediate and settled
  smoke were hard-green. The final fresh window had `216/216` remote and
  `52/52` account-critical calls successful, seven successful fill refreshes,
  five config-valid processes, zero hard/log/monitor/pipeline failures, no
  active HSL replay, and a clean tracked checkout.
- PR #1248 was activated with one exact five-bot graceful restart. Every old
  bot exited naturally within ten seconds after one SIGINT round. Natural
  KuCoin output reduced the approved-coin override line from 678 to 154 visible
  characters. Real pre- and post-restart KuCoin timeouts recovered without
  intervention; the final fresh smoke was hard-green with `56/56`
  account-critical calls successful, nine successful fill refreshes, five
  config-valid processes in exact `R/S` states, and a clean tracked checkout.
- PR #1247 was activated with one exact five-bot graceful restart. Every old
  bot exited naturally after one SIGINT round, with no escalation. The
  immediate five-minute smoke was hard-green with `338/338` remote and `93/93`
  account-critical calls successful. The settled smoke remained `ok=true`
  with zero hard failures and `55/55` account-critical calls successful; one
  non-account-critical candle timeout remained non-hard evidence, and two
  report-time `D` samples cleared to exact `R/S` states. Natural Hyperliquid
  output proved the compact trailing projection at 211 visible characters.
- PR #1246 was activated with one exact five-bot graceful restart. Every old
  bot exited naturally after one SIGINT round; KuCoin was last at about 30
  seconds, with no escalation. The first natural cadence emitted five durable
  trailing/unstuck events with `changed=true` and `operator_visible=true` plus
  five human lines. The second cadence emitted five durable events with
  `changed=false` and `operator_visible=false` and zero human lines, including
  natural sub-threshold Hyperliquid drift.
- The final two-minute smoke was `ok=true` with `224/224` remote calls and
  `56/56` account-critical calls successful, nine successful fill refreshes,
  five matching/config-valid processes in state `R`, and zero hard, log,
  monitor, process, or event-pipeline failures. The 311-character first
  trailing line exposed the active formatter-only follow-up.
- PRs #1244 and #1245 were activated together with one exact five-bot graceful
  restart. Every old bot exited naturally after one SIGINT round; KuCoin was
  last at 40 seconds, with no escalation. A real immediate KuCoin timeout
  recovered without intervention and naturally proved the bounded #1244
  incident line: operation, error type, endpoint, and action remained visible,
  while raw URL and traceback text were absent from the new log.
- All five bots naturally emitted `resource.memory_snapshot`; the complete
  bounded payloads reached monitor storage and compact console lines measured
  84-107 characters. The final two-minute smoke was `ok=true` with `217/217`
  remote calls and `54/54` account-critical calls successful, six successful
  fill refreshes, five matching processes/configs in states `R=4,S=1`, and
  zero hard, log, monitor, process, or event-pipeline failures.
- PR #1243 was activated with one exact five-bot graceful restart. Every old
  bot exited naturally after one SIGINT round; KuCoin was last at 35 seconds,
  and no escalation was required. Natural Binance, GateIO, and KuCoin replay
  progress lines were all at least 30 seconds apart while complete structured
  progress remained durable. The final two-minute smoke was `ok=true` with
  `204/204` remote calls and `55/55` account-critical calls successful, eight
  successful fill refreshes, five matching processes/configs, and zero hard,
  log, monitor, process, or event-pipeline failures. Two real KuCoin startup
  timeouts in the first five-minute window recovered without intervention and
  exposed the active incident-projection follow-up.
- PR #1242 was activated with one exact five-bot graceful restart. Every old
  bot exited naturally after one SIGINT round; KuCoin was last at 40 seconds,
  and no escalation was required. One real KuCoin authoritative-state timeout
  made the first settled window red, then recovered without intervention. The
  final two-minute smoke was `ok=true` with `198/198` remote calls and `46/46`
  account-critical calls successful, eight successful fill refreshes, five
  matching processes/configs in state `R`, and zero hard, log, monitor,
  process, or event-pipeline failures. No natural OKX config-refresh outcome
  occurred in the bounded window and none was manufactured. Fresh HSL replay
  progress counts of `27/26/14/16` on Binance/GateIO/KuCoin/OKX exposed the
  active console-cadence follow-up.
- PR #1241 was activated with one exact five-bot graceful restart. Every old
  bot exited naturally after one SIGINT round; no escalation was required.
  The settled smoke was `ok=true` with `225/225` remote calls and `48/48`
  account-critical calls successful, eight successful fill refreshes, all HSL
  replays complete, five matching processes/configs, states `R=4,S=1`, and
  zero hard, log, monitor, process, or event-pipeline failures. Fresh natural
  logs contained zero successful fill-refresh or fetcher-request timing INFO
  lines, while structured smoke evidence retained eight successful refresh
  summaries. No fill or failure was manufactured.
- PR #1240 was activated with one exact five-bot graceful restart. Every old
  bot exited naturally after one SIGINT round; no escalation was required.
  The immediate window caught one real KuCoin authoritative-state timeout and
  active startup HSL replay. Both aged out before the settled two-minute smoke,
  which was `ok=true` with `176/176` remote calls and `32/32`
  account-critical calls successful, six successful fill refreshes, all five
  expected processes/configs in state `R`, and zero hard, log, monitor,
  process, or event-pipeline failures. Natural INFO completed staged-refresh
  lines were all at or above ten seconds (`10061-13600ms`), while structured
  INFO retained interesting sub-threshold samples at `2190ms`, `5948ms`, and
  `7431ms`, proving the sink boundary without manufactured events.
- PR #1239 was activated with one exact five-bot graceful restart. Every old
  bot exited naturally after one SIGINT round; no escalation was required.
  The immediate startup window retained one real KuCoin authoritative-state
  timeout and degraded cycle. It aged out before the settled two-minute smoke,
  which was `ok=true` with `219/219` remote calls and `50/50` account-critical
  calls successful, six successful fill refreshes, five matching processes and
  configs, states `R=4,S=1`, and zero hard, log, monitor, process, or
  event-pipeline failures. No natural candle-fetch failure occurred in the
  bounded post-restart window, so the new warning format was not manufactured.
- PR #1238 was activated with one exact five-bot graceful restart. Every old
  bot exited naturally after one SIGINT round; no escalation was required.
  The immediate window retained one real KuCoin authoritative-balance timeout
  and its degraded cycle. The settled two-minute smoke was `ok=true` with
  `262/264` remote calls and all `41/41` account-critical calls successful, six
  successful fill refreshes, five valid processes, and zero hard, log,
  monitor, or event-pipeline failures. The two remote failures were non-hard
  candle-fetch timeouts. All five bots completed a natural market-ready cycle,
  normal logs contained zero `open-tail EMA projection contexts` INFO records,
  and the complete diagnostic remains available at DEBUG.
- PR #1237 was activated with one exact five-bot graceful restart. Two bots
  exited immediately after one SIGINT round; GateIO and Binance briefly entered
  uninterruptible I/O sleep while KuCoin drained, and all three then exited
  naturally within the bounded wait without escalation. The immediate smoke
  was `ok=true` with `19/19` remote and `14/14` account-critical calls
  successful. The settled smoke remained `ok=true`: `249/251` remote calls and
  all `47/47` account-critical calls succeeded, six fill refreshes succeeded,
  all five processes/configs matched, and hard, log-attention, monitor, and
  pipeline failures were zero. The two remote failures were non-hard KuCoin
  candle-fetch timeouts.
- Natural Binance, GateIO, and OKX logs now contain only the producer-owned
  materiality-aware forager selection INFO lines; no structured `[forager]
  succeeded` duplicate appeared. Exact Rust `forager.selection` records remain
  present in monitor storage, proving durable delivery was preserved.
- PR #1236 was activated with one exact five-bot graceful restart. All old bots
  exited naturally after one SIGINT round; no escalation was required. A
  startup KuCoin authoritative-open-orders timeout aged out. The settled
  two-minute smoke was `ok=true`: `216/216` remote calls and all `57/57`
  account-critical calls succeeded, eight fill refreshes succeeded, all five
  processes/configs matched, and hard, log-attention, monitor, and pipeline
  failures were zero. One sampled `D` state cleared immediately, with all five
  processes then observed running in their original panes.
- Natural post-restart logs contain zero `forager refresh complete` INFO lines
  on all five bots while normal candle/cache activity continued. The same logs
  exposed paired structured and producer-owned INFO summaries for material
  forager selections, which triggered PR #1237's ownership fix.
- PRs #1233, #1234, and #1235 were activated together with one exact five-bot
  graceful restart. Four old bots exited within ten seconds; KuCoin exited
  naturally after a bounded uninterruptible wait, with no escalation. The
  settled two-minute smoke was `ok=true`: `259/260` remote calls succeeded,
  all `46/46` account-critical calls succeeded, seven fill refreshes succeeded,
  all five expected processes/configs matched, HSL replay completed, and hard,
  log-attention, monitor, and event-pipeline failures were zero. The sole remote
  failure was a non-hard KuCoin candle-fetch timeout.
- Fresh natural output contained zero routine `[warmup]` INFO lines and zero
  successful candle-index maintenance INFO lines on all five bots. Startup
  readiness milestones remained visible, proving the PRs #1234-#1235 level
  boundary without manufactured events. Raw-only balance jitter did not occur
  naturally in the bounded window, so PR #1233 retained explicit
  no-observation evidence.
- PR #1231 merged at `991dadb69124e838a4a3b63fff65036a223b4195`.
  It defines evidence-based console admission, incident projection, and volume
  budgets. It is documentation-only, so no VPS5 restart was required.
- PR #1233 merged at `22ca1a78fa16c2dad827fcf39a6b1fb245302c2b`
  and the VPS5 checkout fast-forwarded cleanly. It suppresses raw-only balance
  jitter from the console while preserving structured, monitor, and durable
  text delivery. It is now active on VPS5; the bounded post-restart window did
  not contain a natural raw-only balance transition to validate the projection.
- PR #1234 merged at `e1dcf3164a88b77762e3cd3ed3e40b834513469a`.
  It moves routine successful warmup detail to DEBUG while preserving startup
  milestones, failures, structured cache decisions, and all warmup behavior.
  It is now active on VPS5, where fresh natural logs retained readiness
  milestones without routine `[warmup]` INFO detail.
- PR #1221 merged and deployed at
  `dacd66adebfd230999aebf7f9fbd34a5b2990490`. It made the structured
  realized-loss gate warning the sole normal console/text owner while preserving
  the legacy fallback and all gate behavior. Only the five exact bot panes were
  gracefully restarted; unrelated processes and local artifacts were preserved.
  After transient GateIO/KuCoin timeouts and HSL replay completed naturally, the
  final two-minute smoke was `ok=true`: `198/198` remote calls, `49/49`
  account-critical calls, six successful fill refreshes, five expected
  processes, and zero hard, log-attention, monitor, or event-pipeline failures.
- PR #1220 merged and deployed at
  `9773889ecb8a396bec31e1e11c326aed9fa2cbe7`. It made structured
  min-effective-cost block events own normal per-block console/text output
  while preserving the aggregate summary, DEBUG details, fallback, throttles,
  and trading behavior. All five old bot processes exited naturally within ten
  seconds after one signal round; no escalation was needed. Pane PIDs and
  unrelated `misc:0.0` PID `434835` remained unchanged. Fresh two-minute smoke
  windows were `ok=true`; the final recorded `279/279` remote calls, `44/44`
  account-critical calls, seven successful fill refreshes, five config-valid
  processes, and zero hard/log/pipeline failures. Every active HSL replay had
  completed required protective work with no failed or stale status. A sampled
  OKX `D` state cleared and remained `R` across five consecutive checks. Natural
  GateIO output contained only the structured min-effective-cost line and no
  adjacent legacy warning, proving runtime single ownership without
  manufacturing a decision.
- PR #1219 merged and deployed at
  `23d9e72af180e8636de7f80cdff8178a60e61937`. It made structured
  initial-entry distance-gate blocked/cleared events own normal console/text
  output while preserving throttles, state transitions, fallback, and trading
  behavior. All five old bot processes exited naturally after one signal round;
  KuCoin was last at 45 seconds, with no force action. Pane PIDs and unrelated
  `misc:0.0` PID `434835` remained unchanged. Immediate and settled smoke
  reports were `ok=true`. The final bounded report recorded `299/299` remote
  calls, `32/32` account-critical calls, six successful fill refreshes, five
  config-valid processes, zero hard/log/monitor/pipeline failures, and complete
  required work for every active HSL replay. A transient `D` state cleared; the
  final exact process sample showed all five bots `R` and a clean repository.
  Natural blocked events on Binance, KuCoin, GateIO, and OKX appeared only as
  the structured line, proving runtime single ownership without manufacturing
  a transition.
- PR #1218 merged and deployed at
  `4bf7706d79f2e2404f785195973d13ea49c31efb`. It made the structured
  ambiguous-cancel terminal warning the sole normal console/text owner and
  added the compact full-account-confirmation cue while preserving cancellation
  and authoritative-confirmation behavior. All five old bot processes exited
  naturally after one signal round; pane PIDs and unrelated `misc:0.0` PID
  `434835` remained unchanged. The settled two-minute smoke returned `ok=true`,
  `472/472` remote calls, `25/25` account-critical calls, `7/7` fill refreshes,
  five matching processes/configs, zero hard/log/monitor/pipeline failures, four
  complete active HSL replays, and an exact clean repository. A transient `D`
  state cleared within 20 seconds. No natural ambiguous cancel occurred after
  restart, so runtime format evidence remains absent rather than manufactured.
- PR #1217 merged and deployed at
  `6599fba08cadffac99ce6a1ce2bfd3f58ca3fa15`. It made the structured
  execution-loop error-burst health event the sole normal console/text owner
  while preserving legacy fallback and every threshold, payload, restart, and
  trading boundary. All five old bot processes exited naturally after one
  signal round; exact pane PIDs and unrelated `misc:0.0` PID `434835` remained
  unchanged. A real KuCoin timeout made the first settled window red. After
  recovery, the final two-minute smoke returned `ok=true`, `284/284` remote
  calls, `62/62` account-critical calls, five running processes, `8/8` fill
  refreshes, zero pipeline failures, no active HSL replay, and an exact clean
  repository. No natural post-restart error burst occurred, so runtime format
  evidence remains absent rather than manufactured.
- PR #1216 merged and deployed at
  `13e6e484cf20b1265f2b4874b14ff7ab32d10bfd`. The settled smoke returned
  `ok=true`, `414/414` remote calls, `70/70` account-critical calls, five
  green process/config checks, `12/12` fill refreshes, and exact repository
  head with zero tracked changes. Four natural periodic health lines proved
  compact single ownership and sane RSS. A natural KuCoin error burst
  separately exposed the remaining adjacent legacy/structured error-burst
  duplicate, which is the active follow-up above.
- PR #1215 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  It made structured fill events own normal console/text output, added one
  bounded bulk summary while preserving every durable per-fill event, and kept
  the stdlib fallback for configurations without a structured console. VPS5
  fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes; every old bot exited naturally without force kill. Pane PIDs and the
  unrelated `misc:0.0` PID `434835` were preserved. Initial windows honestly
  reported real KuCoin timeouts; after recovery, the final two-minute smoke was
  hard-green with `384/384` remote and `71/71` account-critical calls, all five
  expected processes, zero hard/log/monitor failures, no active HSL replay, and
  a clean tracked repository. No natural post-restart fill occurred, so runtime
  fill-format evidence remains absent rather than manufactured.
- PR #1214 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes; every old bot exited naturally without force kill. Pane PIDs and
  unrelated `misc:0.0` PID `434835` were preserved. A real KuCoin timeout made
  the first settled window red and remained visible; after it aged out, the
  final two-minute smoke was hard-green with `251/251` remote and `38/38`
  account-critical calls, all five expected processes, zero hard/log/monitor
  failures, no active HSL replay, and a clean tracked repository.
- Seven distinct post-deploy health windows contained 52 natural retention
  runs. Paired wall/thread-CPU/non-CPU totals were
  `54076.226/7448.346/46627.879ms`: thread CPU was `13.774%` and direct non-CPU
  time was `86.226%`, with the accounting identity matching within `0.001ms`.
  Drops, sink errors, degraded counts, final queue depth, and unfinished work
  remained zero. The residual wall-time tail is host descheduling/contention
  evidence and does not justify another retention CPU or phase optimization.
- PR #1213 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes; every old bot exited naturally, including KuCoin after about 132
  seconds, with no force kill. Pane PIDs and unrelated `misc:0.0` PID `434835`
  were preserved. A real KuCoin timeout correctly made one settled smoke red;
  after it aged out, the final two-minute smoke was hard-green with `403/403`
  remote and `67/67` account-critical calls, all five expected processes, zero
  hard/log/monitor failures, no active HSL replay, and a clean tracked
  repository. A later quiet check found all five bots `R` at the deployed head.
- Four fresh health windows contained 54 natural retention runs with
  `60803.819ms` total and an `11066.952ms` maximum. Inventory accounted for
  `59485.603ms` total and an `11039.628ms` maximum, while age-filter total/max
  was `528.574/44.451ms`. Combined with the earlier isolated age-filter outlier,
  this proves the wall-time tail moves between phases under VPS5 contention;
  it does not justify another phase optimization. Drops, sink errors, degraded
  counts, final queue depth, and unfinished work remained zero.
- PR #1212 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes; every old bot exited naturally, and pane PIDs plus unrelated
  `misc:0.0` PID `434835` were preserved. The final two-minute smoke was
  hard-green with `396/396` remote and `72/72` account-critical calls, all five
  expected processes, zero hard/log/monitor failures, no active HSL replay, and
  a clean tracked repository. Across 59 due runs, inventory maximum improved
  about 78% while drops, sink errors, degraded counts, and final queue depth
  remained zero. The residual overall retention maximum is diagnostic evidence,
  not justification for another behavior optimization.
- PR #1211 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly through PR #1209 and #1211 and gracefully
  restarted only the five exact bot panes; all old bots exited naturally, pane
  PIDs and unrelated `misc:0.0` PID `434835` were preserved. Immediate smoke
  was hard-green with `130/130` remote and `42/42` account-critical calls.
  Settled smoke was hard-green with `385/385` remote and `45/45`
  account-critical calls, no active HSL replay, zero hard/log/monitor failures,
  all five expected processes, and a clean tracked repository. Three processes
  were sampled briefly in `D` during inventory I/O and all returned to `R`.
- Four fresh health windows covered 2,238 monitor writes. Twelve retention runs
  consumed `15591.553ms` total with a `10253.648ms` maximum; inventory explained
  `15369.148ms` total and a `10241.787ms` maximum. The runs visited 20,158
  entries and found 20,032 direct candidates, with zero age/cap deletions,
  proving inventory traversal/stat is the active long-tail source.
- PR #1210 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes; every old bot exited naturally and pane PIDs remained unchanged.
  The final settled smoke was hard-green with `428/428` remote and `76/76`
  account-critical calls successful, all five processes present, zero hard,
  log, or monitor failures, and a clean tracked repository. Natural balance
  changes on all five bots showed the compact raw/snap transition contract;
  no trading state was created or altered for validation.
- A later 10-run window recorded `14255.296ms` retention total and an
  `8654.591ms` maximum. PR #1208 improved ordinary and cumulative work but did
  not eliminate the retention long tail; phase attribution is required before
  another behavior change.
- PR #1208 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes; every old bot exited naturally and pane PIDs remained unchanged.
- The immediate smoke was hard-green. One settled window correctly retained a
  real KuCoin timeout; after it aged out, the final two-minute smoke was
  hard-green with `315/315` remote and `62/62` account-critical calls
  successful, all five processes `R`, zero hard/log/monitor failures, and a
  clean tracked repository.
- Four fresh health windows covered 2,345 monitor writes. Twelve retention runs
  consumed `5612.290ms` with a `690.434ms` maximum, down from PR #1206's matched
  12-run `16210.383ms` total and `8953.523ms` maximum. Drops, sink errors,
  degraded counts, unhealthy pipelines, and final queue depth were zero.
- PR #1207 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes; every old bot exited naturally and pane PIDs remained unchanged.
- Immediate and settled two-minute smokes were hard-green. The settled window
  reported `359/359` remote and `51/51` account-critical calls successful,
  all five expected processes matched, states `R=4/S=1`, zero hard/log/monitor
  failures, and a clean tracked repository. Hyperliquid naturally emitted the
  aligned position line with `WE`, `WEL`, `eWEL`, `TWEL`, and `uPnL`; no live
  position was created or altered for validation.
- PR #1206 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes. All old bot processes exited naturally; pane PIDs
  `856294/856332/856364/856398/856434` and unrelated `misc:0.0` PID `434835`
  were preserved.
- The settled two-minute smoke was hard-green with `575/575` remote and
  `17/17` account-critical calls successful, all five expected processes
  matched, states `R=3/S=2`, zero monitor or pipeline errors, and a clean
  tracked repository. One HSL replay remained active but non-stale.
- Four fresh health windows covered 2,328 monitor writes. Inclusive maintenance
  was `23651.665ms`: 12 retention runs consumed `16210.383ms` with an
  `8953.523ms` maximum, while 352 periodic manifest checkpoints consumed
  `7377.475ms` with a `347.039ms` maximum. Retention therefore explains the
  recurring long tail and is the next persistence optimization target. Drops,
  sink errors, degraded counts, unhealthy pipelines, and final queue depth
  were zero.
- PR #1205 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes. All old bot processes exited naturally; pane PIDs
  `856294/856332/856364/856398/856434` and unrelated `misc:0.0` PID `434835`
  were preserved.
- The settled two-minute smoke was hard-green with `471/471` remote and
  `61/61` account-critical calls successful, all five expected processes
  matched, zero monitor parse/errors, and a clean tracked repository. One
  report-time `D` sample caused no process hard failure.
- Four initial fresh health windows covered 2,490 monitor writes. Monitor
  service total/max was `20064.057/5543.611ms`; maintenance
  `16914.708/5543.163ms`; persistence `1733.352/119.369ms`; lock wait
  `544.284/487.369ms`; rotation `286.78/22.825ms`; and conversion
  `48.654/0.325ms`. Drops, sink errors, degraded counts, unhealthy pipelines,
  and final queue depth were zero.
- Four later settled health windows covered 8,959 writes. Monitor service
  total/max was `83557.728/7148.911ms`; maintenance
  `77452.154/7148.422ms`; persistence `3084.764/111.555ms`; lock wait
  `1507.36/508.744ms`; rotation `586.952/32.995ms`; and conversion
  `206.946/32.574ms`. Maintenance fell from the PR #1204 baseline's
  `17.964ms` to `8.645ms` per write, but remained the recurring long-tail
  source. Drops, sink errors, degraded counts, unhealthy pipelines, and final
  queue depth were zero. One unrelated hard exchange event remained in that
  five-minute report window.
- PR #1204 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes. All old bot processes exited naturally; pane PIDs
  `856294/856332/856364/856398/856434` and unrelated `misc:0.0` PID `434835`
  were preserved.
- Four complete fresh health windows covered 2,643 monitor writes. Monitor
  service total/max was `53903.942/1676.541ms`; maintenance
  `47478.85/900.321ms`; persistence `3619.444/77.067ms`; lock wait
  `2190.037/1661.187ms`; rotation `245.173/20.658ms`; and conversion
  `54.686/1.139ms`. Drops, sink errors, degraded counts, unhealthy pipelines,
  and final queue depth were zero.
- The settled two-minute smoke was green with `346/346` remote and `51/51`
  account-critical calls successful, all five expected processes matched,
  exact states `R/R/R/R/S`, and a clean tracked repository.
- PR #1203 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully restarted only the five exact bot
  panes. All old bot processes exited naturally; pane PIDs
  `856294/856332/856364/856398/856434` and unrelated `misc:0.0` PID `434835`
  were preserved.
- Four fresh bots reported 2,525 processed events and monitor writes, zero
  structured writes, monitor service total/max `62842.105/5321.841ms`, worker
  service total/max `62937.772/5321.883ms`, and queue-wait total/max
  `18035.11/2350.05ms`. Queue depth and unfinished work returned to zero; drops,
  sink errors, degraded counts, and unhealthy pipelines remained zero. The
  performance report independently projected the same fields and omitted them
  from historical rows that predated the producer.
- The final settled two-minute smoke was green with `360/360` remote and
  `55/55` account-critical calls successful, all five expected processes
  matched, exact states `R/R/R/R/S`, pane and unrelated process IDs preserved,
  and a clean tracked repository.
- PR #1200 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully stopped only the five exact bot
  panes; all old Python PIDs exited naturally, including KuCoin after a bounded
  uninterruptible wait. Pane PIDs and unrelated `misc:0.0` PID `434835` were
  preserved before the exact supervisor commands started the five new bots.
- Fresh smoke and performance reports projected three health windows with
  `1517` processed events, queue-wait total/max `29858.257/1077.844ms`, and
  worker-service total/max `63316.18/1033.432ms`; drops, sink errors, degraded
  counts, and unhealthy pipelines were zero. A real GateIO authoritative
  balance timeout made one intermediate smoke red. After recovery, the final
  two-minute smoke was green with `432/432` remote and `59/59`
  account-critical calls successful, all five expected processes matched,
  exact states `R/R/R/R/S`, and a clean tracked repository.
- PR #1199 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded without bot signals or restarts. A bounded four-segment
  per-bot proof report returned `ok=true` with zero issues and reported all five
  current bots `proven`; cache loads preceded proof, and proof elapsed ranged
  from `5.851s` to `38.306s`.
- The first smoke retained a real KuCoin balance timeout; the next retained a
  recovered KuCoin nonce error. After both aged out, the final two-minute smoke
  was green with `283/283` remote and `39/39` account-critical calls successful,
  zero hard/log/monitor/process failures, and a clean tracked repository. A
  report-time `D` sample cleared; all five bots were `R/S` on the final exact
  state check. Bot PIDs, pane PIDs, and unrelated `misc:0.0` PID `434835`
  remained unchanged.
- PR #1198 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded cleanly and gracefully stopped only the five exact bot
  panes; all old Python PIDs exited naturally. Existing pane PIDs and unrelated
  `misc:0.0` PID `434835` were preserved, then the exact supervisor commands
  started the five new bots.
- The immediate smoke retained a real KuCoin authoritative-state timeout:
  `264/267` remote and `15/18` account-critical calls succeeded. After recovery,
  the settled two-minute smoke was hard-green with `325/325` remote and `53/53`
  account-critical calls successful, all five expected processes matched, no
  hard/log/monitor/pipeline failures, no `D` states, and a clean tracked repo.
  Current post-restart segments contained neither `execution.*_sent` nor
  connector-call events, so no live order was fabricated for validation.
- PR #1197 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded without bot signals or restarts. A bounded eight-segment
  Binance lifecycle report returned `ok=true` with zero issues and observed
  `first_fresh_entry_eligible` at `240.110s`, alongside first cycle, Rust call,
  and submitted write evidence. The settled two-minute smoke was hard-green
  with `208/208` remote and `53/53` account-critical calls successful, all five
  expected processes matched, no hard/log/monitor failures, and a clean tracked
  repository. One report-time `D` sample cleared; all five bots were `R` on the
  final exact-state check. Bot, pane, and unrelated `misc:0.0` PIDs remained
  unchanged.
- PR #1196 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded and restarted only the five supervised panes. A focused
  query observed three bounded eligibility events with eligible,
  distance-blocked, and no-candidate outcomes. The first smoke retained a real
  KuCoin timeout; after recovery, the settled two-minute smoke was hard-green
  with `348/348` remote and `43/43` account-critical calls successful. The
  quiet one-minute smoke remained green at `175/175` and `34/34`; transient
  report-time `D` samples cleared to exact `R/S` process states.
- PR #1195 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded without bot signals or restarts, preserving bot PIDs
  `850148/850296/850370/850436/850495` and unrelated `misc:0.0` PID
  `434835`. Bounded readiness and milestone reports scanned 12 files with
  zero issues; incomplete sources no longer attached stale rotated lifecycle
  data, while KuCoin retained its bounded sparse HSL context and first-cycle
  milestone at `220.123s`. The first smoke caught a real KuCoin balance
  timeout. After it aged out, the retry was green with `284/284` remote and
  `57/57` account-critical calls successful, 5/5 bots matched, no hard/log/
  monitor failures, no event-pipeline errors, and a clean tracked repository.
  Two report-time `D` samples cleared; the quiet follow-up showed all five bots
  `Rsl+`.
- PR #1194 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded without process signals, preserving bot PIDs
  `850148/850296/850370/850436/850495` and unrelated `misc:0.0` PID `434835`.
  The exact bounded rotated `startup_milestones` report returned `ok=true`,
  scanned 12 files / 46,748 records with zero errors/warnings, kept truncated
  lifecycle evidence explicitly unknown, and observed KuCoin's first cycle at
  `110.653s` without claiming unseen Rust/write milestones. The first smoke
  caught one recovered Binance `InvalidNonce` and two transient `D` samples.
  After settling, all `D` states cleared and the final two-minute smoke was
  hard-green: `384/384` remote and `76/76` account-critical calls successful,
  5/5 processes matched (`R=4,S=1`), no hard/log failures, no pipeline
  drops/sink errors, and a clean tracked repository.
- PR #1193 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 fast-forwarded without bot signals or restarts, preserving bot PIDs
  `850148/850296/850370/850436/850495` and unrelated `misc:0.0` PID `434835`.
  The exact capped current-plus-rotated report returned `ok=true`, scanned 12
  files with zero errors/warnings, retained all five current per-bot lifecycle
  snapshots, and preserved historical aggregates (`account` phase count six
  versus five current bots). The settled smoke was hard-green with `322/322`
  remote and `57/57` account-critical calls successful, all five bots matched,
  no event-pipeline drops/sink errors, and a clean tracked repository. Report
  I/O briefly produced three `D` samples; all five bots returned to `R` after
  the quiet follow-up.
- PR #1192 merged after exact-head Hermes and Grok 4.5 approval plus green CI.
  VPS5 replaced bot PIDs `842617/842655/842687/842721/842757` with
  `850148/850296/850370/850436/850495`, preserving `misc:0.0` PID `434835`.
  Live reports accepted five account and execution-loop scopes, four
  held-position protective scopes, three first-market-state scopes, and a
  completed background-candle scope using canonical impact labels;
  `active-candle` remained timing-only. The immediate smoke caught three real
  KuCoin balance timeouts, then the settled two-minute smoke was hard-green
  with `380/380` remote and `16/16` account-critical calls successful, all five
  configured bots matched, no pipeline drops/sink errors, and a clean tracked
  repository. Two transient `D` states during cache/replay work cleared to
  `R/S` on the quiet follow-up sample.
- PR #1191 merged as `359929007dce0b47c023a36fdef90a7106ae46da` after
  exact-head Hermes and Grok 4.5 approval plus green CI, then was pulled to
  VPS5 without restarting bots. Real optional-replay progress with
  `required_pairs=0` retained positive generic dense remaining work while
  required remaining stayed zero; exact candidate terminal records stayed
  zero. A retained legacy KuCoin terminal record without candidate totals
  reported `legacy_terminal_no_candidate_rows`, generic remaining zero, and
  retained its diagnostic dense remainder. The settled two-minute smoke was
  green with `278/278` remote and `58/58` account-critical calls successful,
  all five expected bots matched, no hard/log/monitor failures, and a clean
  repository. Large rotated-history validation briefly produced host I/O waits;
  after a quiet interval all five bot processes were `Rsl+` with their
  original PIDs.
- PR #1190 merged as `6b2da757f2fbc590c12365870475176632269021` and was deployed
  to VPS5. All five exact supervised panes were restarted, unrelated
  `misc:0.0` PID `434835` was preserved, and immediate live evidence showed the
  new scanned counters, scan rate, and candidate-row source. All four HSL
  replays completed successfully in `108.782s` to `246.091s`, reaching
  protective-ready in `9.920s` to `71.744s`; `total_scanned_rows` equaled
  `candidate_rows`, `observed_candidate_work_pct=100`, and the source was
  `candidate_rows_terminal`. The final fresh smoke was `ok=true` with `333/333`
  remote and `45/45` account-critical calls successful, all five expected bots
  matched, no hard/log/monitor failures, and a clean tracked repository. One
  process was sampled in `D`, but no process hard failure occurred. A prior
  monitor-row parse boundary was immediately revalidated clean.
  A later direct process check showed `D` cleared: Binance, KuCoin, GateIO, and
  Hyperliquid were `Ssl+/ep_poll`, OKX was `Rsl+`; `vmstat` showed `b=0` before
  a transient `b=1` with low or zero sampled iowait.
- PR #1188 was approved on exact head `363eca852` by Hermes and Grok 4.5;
  CI passed and it merged as `bd169747`. VPS5 fast-forwarded cleanly and
  gracefully restarted only `passivbot:4.0`; Hyperliquid bot PID `842779` was
  replaced by `844272`, while the other four bot PIDs and unrelated
  `misc:0.0` PID `434835` remained unchanged. The unified account reached
  startup-ready in `48.00s` and full-warmup-ready in `74.77s`. A focused query
  found zero fatal HIP-3 compatibility events, as expected. The final settled
  smoke was hard-green with all five processes matched, `327/327` remote and
  `45/45` account-critical calls successful, states `R=4,S=1`, no
  uninterruptible sleep, no hard/log failures, and tracked repository status
  clean.
- PR #1187 was approved on exact head `74766c7cb` by Hermes, Grok 4.5, and the
  independent Codex reviewer; CI passed and it merged as `b99d1b05`. VPS5
  restarted only the five supervised bots, preserving `misc:0.0` PID `434835`.
  A bounded event query found the exact four expected per-side
  `config.market_compatibility` records: Binance `CRO,MNT` and OKX
  `KAS,MNT,XMR`, all non-hard degraded approved-list evidence. An immediate
  smoke caught a real KuCoin timeout; after it aged out, the settled smoke was
  hard-green with `370/370` remote and `26/26` account-critical calls
  successful, all five processes matched, states `R=4,S=1`, and no
  uninterruptible sleep.
- PR #1186 was approved on exact head `65d61702f` by Hermes, Grok 4.5, and the
  independent Codex reviewer that found both ordering regressions; CI passed
  and it merged as `b9748247`. VPS5 fast-forwarded without restarting bots.
  The exact bounded rotated `hsl_replay_profile` report returned `ok=true`,
  zero errors/warnings, and no resource-pressure crash. A settled two-minute
  smoke was hard-green with `216/216` remote and `56/56` account-critical calls
  successful, all five expected bots matched, and no text-log hard matches.
  The original five bot PIDs remained unchanged and returned to `Rl+`; unrelated
  `misc:0.0` remained PID `434835`. An earlier smoke window caught one real
  KuCoin timeout cycle, which aged out before the settled green window.
- PR #1184 was approved by Hermes and Grok 4.5 on exact head `9177dfed9`, CI
  passed, and it merged as `5526d5de`. VPS5 fast-forwarded cleanly while
  preserving untracked artifacts. The five old bots stopped after the second
  exact-pane Ctrl-C; only the `passivbot` session was reloaded, and unrelated
  `misc:0.0` remained PID `434835`.
- Immediate and fresh settled smoke reports were hard-green with all five
  expected bots matched and zero hard failures. The settled window recorded
  `249/249` successful remote and `34/34` account-critical calls, process states
  `R=4, S=1`, no uninterruptible sleep, and tracked repository status clean.
- All four coin-HSL replays completed with strategy `mixed` and candidate
  reduction from `86.865%` to `93.449%`: KuCoin `140.746s`, Binance `208.098s`,
  GateIO `229.248s`, and OKX `269.711s`. The post-PR #1183 baseline was
  `601.246s` to `2279.519s`, so the deployed replay is about `77%` to `89%`
  faster while preserving dense held/ambiguous pairs.
- PR #1181 required no restart; all five bot pane PIDs remained unchanged. Its
  bounded current-segment scorecard reported four compact full-replay
  completions from `453.98s` to `1728.585s`, but exposed the active slice's
  missing completion fallback for the rotated protective milestones.
- PR #1182 required no restart and recovered protective elapsed aggregates
  from explicit completion evidence without synthesizing milestone records.
  A fresh recovery smoke was green with `279/279` remote and `66/66`
  account-critical calls successful; all five expected bots matched.
- Preserve local/VPS configs, logs, monitor data, reports, and temporary files

## Review Gate

- Normal gate: all reviewers currently designated by the maintainer plus green
  CI on the exact head SHA.
- Temporary gate while Grok 4.5 is halted: exact-head Hermes + green CI.
- Findings from any additional reviewer must still be verified and resolved.
- Any pushed delta requires current-head re-review.

## Agent Routing

- Sol: architecture, high-risk implementation, finding adjudication, merge,
  VPS signals/restart, and incident judgment.
- Terra: isolated low/medium-risk docs, tests, report/query tooling, and bounded
  observability implementation with explicit file scope.
- Luna or deterministic automation: metadata polling, state-change detection,
  CI/reviewer summaries, and read-only output parsing.
- Parallel PRs must be orthogonal. Dependent work waits for merge to `master`.

## Next Slice

The coin-HSL protective-readiness split, cooperative background cadence,
current process-pressure query, compact cold replay payload, bounded replay
scorecard, stable per-pair fill index, exact sparse flat-pair replay, and the
rotated resource-pressure report fix, configured-market skip events, and fatal
HIP-3 startup compatibility are merged and deployed. PR #1189's isolated-only
initial-entry filter visibility and PR #1190's HSL replay scanned-row
throughput, PR #1191's corrected active/legacy-terminal replay estimates,
PR #1192's machine-readable startup readiness SLA semantics, PR #1193's
latest-lifecycle report ordering, PR #1194's bounded startup action milestones,
PR #1195's startup consumer correctness fixes, PR #1196's true fresh-entry
eligibility producer, PR #1197's fresh-entry startup milestone, PR #1198's
local connector-call boundary evidence, and PR #1199's startup fill-cache proof
correlation, PR #1200's event-pipeline service timing, and PR #1203's fixed
sink-class attribution, PR #1205's coalesced monitor manifest checkpoints, and
PR #1206's monitor-maintenance phase attribution are also merged and deployed.
PR #1207's position console projection and PR #1208's one-pass monitor
retention inventory are also merged and deployed with their behavior boundaries
preserved. PR #1210's balance console transition is merged, deployed, and
validated from natural balance changes on all five bots. PR #1211's retention
phase attribution, PR #1212's direct `os.scandir` inventory, and PR #1213's
whole-loop age/cap attribution are merged and deployed. PR #1214's paired
thread-CPU/non-CPU attribution is also merged and deployed; 52 natural runs
showed only `13.774%` thread CPU and `86.226%` direct non-CPU time. The long
retention wall-time tail is host descheduling/contention evidence and does not
justify another retention optimization.

PR #1215's fill console/text migration, PR #1216's periodic health console
migration, PR #1217's execution-loop error-burst console migration, PR #1218's
ambiguous-cancel console migration, PR #1219's entry-distance-gate console
migration, and PR #1220's min-effective-cost console migration are merged and
deployed. Natural post-PR #1220 GateIO output proved structured single
ownership. PR #1221's realized-loss gate console migration is also merged and
deployed with a settled hard-green smoke. PR #1231's console-verbosity policy
and PR #1232's canonical-master review contract are merged. PR #1233's
raw-balance materiality change, PR #1234's warmup-detail demotion, and PR
#1235's candle-index maintenance demotion are merged, deployed, and active on
VPS5. The settled smoke is green, and natural output proves the warmup and
candle-index level boundaries. PR #1236's forager-refresh completion demotion is
also merged, deployed, and naturally absent from the normal INFO logs. Fresh
output then exposed dual console ownership for Rust-orchestrated material
forager selections. PR #1237's source-scoped ownership fix is also merged,
deployed, and naturally validated: normal logs retain only the producer-owned
material summary while monitor storage retains the complete Rust event.
Python-filter selection visibility remains unchanged.

PR #1238's open-tail EMA projection-context demotion is merged, deployed, and
naturally validated: all five bots completed market-ready cycles without the
1002-1050 character aggregate appearing in normal INFO logs. PR #1239's bounded
candle-fetch warning and durable remote-call redaction are also merged and
deployed. Its settled smoke is green; no natural candle-fetch failure occurred
in the bounded post-restart window. PR #1240's staged-refresh console threshold
is merged, deployed, and naturally validated: completed INFO lines were all at
or above ten seconds while interesting sub-threshold structured INFO samples
remained durable. PR #1241's successful fill-refresh timing demotion is also
merged, deployed, and naturally validated: successful timing detail is absent
from normal INFO while structured refresh summaries remain available. PR
#1242's bounded OKX configuration outcomes are merged and deployed with a
hard-green settled smoke. No natural config outcome occurred in the bounded
window. PR #1243's HSL replay console cadence is merged, deployed, and
naturally validated: intermediate console progress stayed at least 30 seconds
apart while complete structured progress remained durable. Its restart exposed
the execution-loop incident projection addressed by merged PR #1244. PR #1245
is also merged and deployed: bounded execution incidents and compact memory
snapshots are naturally validated with a settled hard-green smoke. PR #1246 is
merged, deployed, and naturally validated: producer-owned materiality suppresses
unchanged trailing/unstuck human repeats while preserving five-minute durable
observations. PR #1247 is also merged, deployed, and naturally validated: its
compact trailing projection reduced the observed Hyperliquid line from 311 to
211 visible characters while retaining the operator-relevant state and leaving
payload, admission, cadence, and behavior unchanged. The restart and settled
smoke were hard-green; report-time `D` samples cleared to exact `R/S` states.

PR #1248 is merged, deployed, and naturally validated: its bounded
`live.approved_coins` startup projection reduced the exact KuCoin line from 678
to 154 visible characters while retaining counts and bounded samples. PR #1249
is also merged, deployed, and naturally validated: eleven logical lock-hold
warnings measured 205-218 visible characters while retaining per-symbol scope
and compact holder identity/timing. PR #1250 is merged, deployed, and naturally
validated: the four forager HSL settings lines measured 163-167 visible
characters while retaining all configured settings. PR #1251 is also merged,
deployed, and naturally validated: admitted staged-refresh lines measured
155-171 visible characters with zero legacy duplicates and a hard-green
settled smoke. PR #1252 is merged, deployed, and naturally validated: compact
close-EMA warnings measured 158-225 visible characters with zero post-deploy
legacy duplicates and a hard-green settled smoke after one bounded GateIO
retry. PR #1253 is also merged, deployed, and naturally validated: 16
initial-entry distance-gate lines measured 181-190 visible characters with
zero legacy duplicates and a hard-green settled smoke. PR #1254 is also
merged, deployed, and naturally validated: four candle-health lines measured
156-211 visible characters with a hard-green settled smoke. PR #1255 is also
merged, deployed, and naturally validated: three forager selection lines
measured 182-185 visible characters with a hard-green settled smoke. PR #1256
is also merged, deployed, and naturally validated: four periodic health lines
measured 202-209 visible characters while preserving all operator-relevant
health facts. PR #1257 is also merged and deployed with a hard-green settled
smoke. Its candidate-required-EMA target did not occur naturally in the new
segments, and zero legacy summary duplicates appeared. PR #1258 is merged,
deployed, and naturally validated: the observed candle-fetch retry warning
measured 201 visible characters with zero legacy caller-bearing duplicates and
a hard-green settled smoke. PR #1259 is also merged, deployed, and naturally
validated: a real Binance `InvalidNonce` recovery produced a 203-character
clock-offset line with no raw exception text and the settled smoke was
hard-green. PR #1260 is also merged, deployed, and naturally validated: all
five approved-coin membership lines measured 130-150 visible characters with
zero legacy duplicates and a hard-green settled smoke. PR #1261's compact
startup HSL warning and PR #1262's initial-entry distance-gate admission are
also merged, deployed, and naturally validated. PR #1263's startup lifecycle
console consolidation is also merged, deployed, and naturally validated:
every bot emitted exactly one human start and ready signal, durable
`bot.ready` remained present, and removed lifecycle INFO lines stayed absent.
PR #1264's maintainer-detail demotion is also merged, deployed, and naturally
validated: successful stop summaries and hourly scheduler jitter remain at
DEBUG while cancellation errors remain at ERROR. PR #1265's bounded,
value-free historical secret-log inventory is also merged and deployed; its
dry run confirmed retained credential-bearing logs without exposing values or
changing artifacts. PR #1266's brief startup-budget coverage, PR #1267's
scheme-less credential-query detection, and PR #1268's aggregate-only inventory
projection are also merged and deployed without restart. PR #1269's configured
startup-budget events are merged and deployed with the same non-enforcement
boundary. PR #1270's matching performance projection is also merged and
deployed without restart. PR #1271's forager feature-unavailability projection
and PR #1272's planning-defer/absent-selector projection are merged and deployed
without restart. PR #1273's forager eligibility projection, PR #1274's
requested-window event inventory, and PR #1275's planning symbol-state health
are also merged, deployed, and naturally validated. PR #1276's initial-entry
eligibility health, PR #1277's correlated planning-output health, and PR
#1278's latest-per-bot-and-kind data-packet health, PR #1280's planning
snapshot health, and PR #1282's cycle terminal-outcome/recovery health are
merged, deployed, and naturally validated. PR #1283's bounded multi-sample
process-state recovery is also merged and deployed without a restart or process
signal; its full VPS smoke remained intentionally unrun after the
production-action rejection. PR #1285's dedicated local-only
`live-process-report` path is also merged, deployed, and naturally validated
with five stable exact processes and recovered uninterruptible observations.
PR #1286's aggregate-only follow-up, PR #1287's exact tmux target preflight,
PR #1288's bounded target-identity stability, and PR #1289's plan binding are
merged, deployed, and naturally validated without process control. PR #1290's
pane-parent relaunch classification and the restart preparation/orchestration
slices through PR #1309 are also merged and deployed. Later logging slices
through PR #1338, including adjacent PR #1329, are deployed at canonical
`f9ceb9678448201af0c0cc5f40f889b661d3021c`; their current evidence is recorded
above. The active log scan-cost slice measures the remaining read-only text-log
artifact work without changing discovery, matching, or verdict behavior.

Do not create progress-only PRs or resume unrelated logging work from stale
worktrees.

## References

- Operating workflow: `live_logging_overhaul_pr_loop_workflow.md`
- Architecture: `live_logging_overhaul_plan.md`
- Historical evidence: `live_logging_overhaul_progress.md`
- Performance goals: `live_performance_readiness_goals.md`
- Operational backlog: `live_ops_improvement_backlog.md`
- Reviewer contract: `../ai/runbooks/pr_review.md`
