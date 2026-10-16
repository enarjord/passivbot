# Organillo historical entry eligibility

Organillo is an optional backtest dataset input, consumed by Rust CPU backtests and native
shared-account GPU optimization (EMA Anchor and Trailing Martingale). It adds no live behavior
or selection-rule calculations. See [user guide](../../organillo.md).

## Input and preparation

`backtest.organillo_csv: null` disables the feature. A nonempty path enables a strict UTF-8 CSV:
`date` followed by unique exact canonical coin identities; ordered unique `YYYY-MM-DD` UTC dates;
every cell exactly `0` or `1`. Reject unreadable files, malformed rows, missing operational dates,
missing prepared coin columns, unresolved columns and malformed transport arrays with actionable
errors. No fuzzy matching, carry-forward or default eligibility.

A daily row applies at UTC midnight until the following midnight. The parent freezes the source
SHA-256 into `backtest.organillo_sha256` and aligns a read-only daily `uint8` matrix plus one `int32`
row index per execution candle. Warmup-only rows and the unused final execution-lookahead candle
map to an inert zero row; that final candle needs no additional CSV date. Every consumed candle
still requires coverage, including any extra consumed rows supplied beyond a configured end.
Candle format stays unchanged. CPU workers borrow shared arrays; native GPU datasets borrow the
same descriptors and upload the schedule when resident, shared across candidates and compatible
directional runners.

The data universe is the period union intersected with existing enabled-side approved coins.
CSV headers must be exact identities in the resolved approved universe (including disabled sides);
side restrictions still apply. Never-selected coins need no candles. For an entirely zero period,
CSV coin identities supply the candle universe for an empty-trading replay. Future-only admissions
do not increase earlier slot counts or impose their warmup on earlier coins. Producer validity and
activation metadata remain authoritative; eligibility never fabricates candle history.

Suites may preload a broader shared candle master. Apply the period union, source hash and
alignment separately to each scenario, so a base schedule cannot prune another scenario's coins
or impose its date coverage on that scenario. Validate CSV identities against the suite's resolved
master universe before selecting each scenario's coin and time slices.

Enabled prepared-data, GPU execution and saved-fitness identities include source content and the
aligned schedule. A changed schedule requires new preparation and invalidates saved fitness.
Explicit expected hashes survive scenario overrides; deliberately replacing a scenario input
requires clearing its expected hash. Shared segment lifetime extends through all CPU workers and
GPU service shutdown. Eviction releases device eligibility tensors along with market tensors.
Reject optimizer runtime pins of these dataset fields and mismatched prepared-buffer hashes;
changing a schedule requires dataset/scenario preparation.

## Simulation

A zero imposes graceful stop on ordinary entry modes. Held positions continue configured DCA,
closes, unstuck and protective management. Once flat, no new position opens. A one restores
ordinary Forager eligibility without overriding another restriction. Recheck at the execution
candle, after closes and before each entry: stale initial orders cannot open on a zero, including
a close followed by an entry within the same candle. Both sides follow the same ordering.

With `dynamic_wel_by_tradability=true`, count currently ready/tradable, side-entry-eligible,
nonzero-WEL coins whose mask is one. Preserve each side's maximum simultaneous count seen so far.
Effective slots are the minimum of configured slots and that peak; automatic WEL is TWEL divided
by effective slots, with existing zero-count handling. Rotation never counts distinct historical
coins cumulatively. Retired held coins retain current budgets; explicit WEL overrides remain
explicit. With the option false, the configured denominator remains fixed.

The disabled Rust engine uses a const-generic specialization. Disabled GPU preprocessing removes
all eligibility arguments, reads and instructions and preserves replay state layout. No schedule
parsing, hashing, shared segments or device transfers occur without a CSV. Native replay has the
existing 64-coin ceiling and existing candle/HSL limitations; never introduce a CPU fallback.
