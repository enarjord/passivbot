"""Version-neutral economics from full ordinary-account observations.

Inputs are account equity, never HSL signals or model-dependent strategy equity.
The replay uses fixed initial capital and has no external deposits/withdrawals.
The caller supplies the pre-first-fill initial equity and complete observations on
an explicit grid. Values are held from each timestamp until the next observation;
there is no invented tail after an incomplete run. Fee cashflow is negative for a
cost and positive for a rebate. Net realized PnL already includes that cashflow.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real


DAY_MS = 86_400_000
METRIC_DEFINITION = "ordinary_account_v1"
# These account economics/activity fields may be used by a frozen discovery
# protocol. Clocks, series metadata and model-dependent HSL metrics may not.
SCORABLE_METRICS = frozenset({
    "net_return", "log_growth_per_day", "geometric_growth_per_day",
    "account_max_drawdown", "underwater_fraction", "recovery_duration_max_ms_lower_bound",
    "gross_exposure_ratio_mean", "gross_exposure_ratio_max", "gross_exposure_ratio_terminal", "exposed_fraction",
    "fees_paid", "rebates_received", "net_fees", "net_realized_pnl", "traded_notional",
    "fill_count", "close_fill_count", "active_days",
})


def _number(value, path):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{path} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{path} must be a finite number")
    return value


def _clock(value, path):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a nonnegative integer millisecond timestamp")
    if isinstance(value, Integral):
        value = int(value)
    else:
        value = _number(value, path)
        if not value.is_integer():
            raise ValueError(f"{path} must contain exact integer milliseconds")
        value = int(value)
    if value < 0:
        raise ValueError(f"{path} must be a nonnegative integer millisecond timestamp")
    return value


def extract_account_metrics(*, start_ms, expected_end_ms, initial_equity,
                            timestamps_ms, equity, gross_exposure_ratio, fills,
                            completed, liquidated, sample_interval_ms):
    """Validate full inputs and return ordinary-account metrics and censor facts.

    Every fill requires identity, timestamp_ms, net_realized_pnl, fee_cashflow,
    notional and is_close. Empty fills are an explicit known-empty history.
    The first equity sample may include start-time economics; initial_equity is
    the separate account baseline before those events. Growth is unavailable for
    nonpositive terminal equity, with an explicit reason rather than a score.
    """
    start = _clock(start_ms, "start_ms")
    expected_end = _clock(expected_end_ms, "expected_end_ms")
    if expected_end <= start:
        raise ValueError("expected_end_ms must be later than start_ms")
    if isinstance(sample_interval_ms, bool) or not isinstance(sample_interval_ms, Integral) or sample_interval_ms <= 0:
        raise ValueError("sample_interval_ms must be a positive integer")
    sample_interval_ms = int(sample_interval_ms)
    if type(completed) is not bool or type(liquidated) is not bool:
        raise ValueError("completed and liquidated must be explicit booleans")
    initial = _number(initial_equity, "initial_equity")
    if initial <= 0:
        raise ValueError("initial_equity must be positive")
    times = [_clock(value, f"timestamps_ms[{i}]") for i, value in enumerate(timestamps_ms)]
    values = [_number(value, f"equity[{i}]") for i, value in enumerate(equity)]
    exposures = [_number(value, f"gross_exposure_ratio[{i}]")
                 for i, value in enumerate(gross_exposure_ratio)]
    if len(times) < 2 or len(times) != len(values) or len(times) != len(exposures):
        raise ValueError("full timestamp/equity/exposure arrays must have equal length >= 2")
    if times[0] != start or times[-1] > expected_end:
        raise ValueError("observations must start at start_ms and not exceed expected_end_ms")
    for i, (left, right) in enumerate(zip(times, times[1:])):
        delta = right - left
        terminal_partial = i == len(times) - 2 and 0 < delta < sample_interval_ms
        if delta != sample_interval_ms and not terminal_partial:
            raise ValueError("observations must cover the full declared sampling grid")
    if completed and times[-1] != expected_end:
        raise ValueError("completed run does not cover expected_end_ms")
    if any(value < 0 for value in exposures):
        raise ValueError("gross exposure ratios must be nonnegative; hedges must not net")
    duration = times[-1] - start
    if duration <= 0:
        raise ValueError("observed duration must be positive")

    peak, peak_time, maximum_dd = initial, start, 0.0
    underwater_ms = exposed_ms = 0
    weighted_exposure = 0.0
    episode = None
    recoveries = []
    for i, (timestamp, value) in enumerate(zip(times, values)):
        if value >= peak:
            if episode is not None:
                recoveries.append({**episode, "end_ms": timestamp,
                                   "duration_ms": timestamp - episode["peak_timestamp_ms"],
                                   "censored": False})
                episode = None
            peak, peak_time = value, timestamp
        elif episode is None:
            episode = dict(peak_timestamp_ms=peak_time, peak_equity=peak,
                           underwater_start_ms=timestamp)
        maximum_dd = max(maximum_dd, (peak - value) / peak)
        if i + 1 < len(times):
            interval = times[i + 1] - timestamp
            if value < peak:
                underwater_ms += interval
            if exposures[i] > 0:
                exposed_ms += interval
            weighted_exposure += exposures[i] * interval
    if episode is not None:
        recoveries.append({**episode, "end_ms": times[-1],
                           "duration_ms": times[-1] - episode["peak_timestamp_ms"],
                           "censored": True})

    fees, rebates, pnl, notional = [], [], [], []
    identities, active_days = set(), set()
    close_count = 0
    previous_fill = start
    fill_count = 0
    for index, fill in enumerate(fills):
        if not isinstance(fill, Mapping):
            raise ValueError(f"fills[{index}] must be a mapping")
        required = {"identity", "timestamp_ms", "net_realized_pnl", "fee_cashflow", "notional", "is_close"}
        if required - fill.keys():
            raise ValueError(f"fills[{index}] missing {', '.join(sorted(required - fill.keys()))}")
        identity = fill["identity"]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("fill identities must be nonempty and unique")
        identities.add(identity)
        timestamp = _clock(fill["timestamp_ms"], f"fills[{index}].timestamp_ms")
        if not previous_fill <= timestamp <= times[-1]:
            raise ValueError("fills must be ordered and inside observed coverage")
        previous_fill = timestamp
        if type(fill["is_close"]) is not bool:
            raise ValueError("fill is_close must be an explicit boolean")
        fee = _number(fill["fee_cashflow"], f"fills[{index}].fee_cashflow")
        trade_pnl = _number(fill["net_realized_pnl"], f"fills[{index}].net_realized_pnl")
        traded = _number(fill["notional"], f"fills[{index}].notional")
        if traded < 0:
            raise ValueError("fill notional must be nonnegative")
        fees.append(max(0.0, -fee))
        rebates.append(max(0.0, fee))
        pnl.append(trade_pnl)
        notional.append(traded)
        close_count += int(fill["is_close"])
        fill_count += 1
        active_days.add(timestamp // DAY_MS)

    final = values[-1]
    log_growth = (math.log(final) - math.log(initial)) / (duration / DAY_MS) if final > 0 else None
    try:
        growth = math.expm1(log_growth) if log_growth is not None else None
    except OverflowError as error:
        raise ValueError("dailyized account growth overflows; input cannot supply a finite objective") from error
    result = dict(
        metric_definition=METRIC_DEFINITION, start_ms=start, observed_end_ms=times[-1],
        expected_end_ms=expected_end, duration_days=duration / DAY_MS,
        initial_equity=initial, final_equity=final, net_return=final / initial - 1.0,
        log_growth_per_day=log_growth, geometric_growth_per_day=growth,
        growth_unavailable_reason=None if final > 0 else "nonpositive_terminal_equity",
        account_max_drawdown=maximum_dd, underwater_fraction=underwater_ms / duration,
        recovery_episodes=recoveries,
        recovery_duration_max_ms_lower_bound=max((row["duration_ms"] for row in recoveries), default=0),
        terminal_underwater_censored=bool(recoveries and recoveries[-1]["censored"]),
        gross_exposure_ratio_mean=weighted_exposure / duration,
        gross_exposure_ratio_max=max(exposures), exposed_fraction=exposed_ms / duration,
        gross_exposure_ratio_terminal=exposures[-1],
        fees_paid=math.fsum(fees), rebates_received=math.fsum(rebates),
        net_fees=math.fsum(fees) - math.fsum(rebates), net_realized_pnl=math.fsum(pnl),
        traded_notional=math.fsum(notional), fill_count=fill_count,
        close_fill_count=close_count, active_days=len(active_days),
        completed=completed, liquidated=liquidated,
        nonpositive_equity_observed=any(value <= 0 for value in values),
        completion_ratio=duration / (expected_end - start), sample_count=len(times),
        sample_interval_ms=sample_interval_ms,
    )
    for key, value in result.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError(f"derived account metric {key} is nonfinite")
    return result
