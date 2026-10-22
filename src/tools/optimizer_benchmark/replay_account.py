"""Ordinary-account confirmation from full outputs of either version's CPU runner.

No native module or runner is imported here. Native equity timestamp k names the
close[k] valuation, observable at end(k+1). Fill index, rather than timestamp alone,
keeps a next-bar opening fill out of the preceding close at their shared clock.
Gross close-sampled notional uses initial capital as its fixed denominator, so a
depleted or clamped account balance cannot erase or invent an exposure ratio.
"""

from collections.abc import Mapping
import hashlib
import json
import math

from .account_metrics import _clock, _number, extract_account_metrics
from .discovery import _sha, canonical_configuration_identity


def account_valuation_input_identity(*, hlcvs, candle_timestamps_ms, coins, market_settings,
                                     initial_capital, first_bar_index, last_bar_index,
                                     candle_interval_ms, bootstrap_bar_indices):
    """Bind immutable valuation inputs separately from the full replay input hash.

    The scenario hash must additionally bind all runner inputs (including BTC
    collateral prices, activation/eligibility and execution policy). Freeze this
    valuation hash during preparation, before producing the native results.
    """
    import numpy as np

    metadata = dict(version=1, coins=list(coins), market_settings=market_settings,
                    initial_capital=initial_capital, first_bar_index=first_bar_index,
                    last_bar_index=last_bar_index, candle_interval_ms=candle_interval_ms,
                    bootstrap_bar_indices=list(bootstrap_bar_indices))
    try:
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("valuation metadata must be finite JSON data") from error
    digest = hashlib.sha256(encoded)
    for values in (hlcvs, candle_timestamps_ms):
        array = np.asarray(values)
        if array.dtype.kind not in "iuf":
            raise ValueError("prepared arrays must have a real numeric non-object dtype")
        digest.update(json.dumps([str(array.dtype), list(array.shape)]).encode())
        digest.update(memoryview(array).cast("B") if array.flags.c_contiguous else array.tobytes(order="C"))
    return digest.hexdigest()


def metrics_from_native_replay(*, native_fills, native_equities, liquidated, execution_completed,
                               hlcvs, candle_timestamps_ms, coins, market_settings,
                               initial_capital, initial_positions, first_bar_index,
                               last_bar_index, candle_interval_ms, bootstrap_bar_indices,
                               valuation_input_sha256):
    """Require full native artifacts and a known-flat pre-first-fill baseline.

    Expected first/last bars come from the prepared replay contract, never from
    the observed output. Incomplete executions may have a contiguous prefix;
    successful non-liquidated runs must cover the entire expected grid. Declared
    pre-evaluation bootstrap rows must equal flat initial capital; their gap to
    activation is excluded from evaluation coverage, never filled synthetically. Missing
    marks of held inventory, unknown order-side attribution or omitted fills are
    errors. Neither native HSL signals nor strategy equity is a quality ruler.
    """
    import numpy as np

    _sha(valuation_input_sha256, "valuation_input_sha256")
    prepared = dict(hlcvs=hlcvs, candle_timestamps_ms=candle_timestamps_ms, coins=coins,
                    market_settings=market_settings, initial_capital=initial_capital,
                    first_bar_index=first_bar_index, last_bar_index=last_bar_index,
                    candle_interval_ms=candle_interval_ms, bootstrap_bar_indices=bootstrap_bar_indices)
    if account_valuation_input_identity(**prepared) != valuation_input_sha256:
        raise ValueError("prepared valuation inputs differ from their frozen identity")
    if type(liquidated) is not bool or type(execution_completed) is not bool:
        raise ValueError("liquidated and execution_completed must be authoritative booleans")
    if not isinstance(initial_positions, Mapping) or initial_positions:
        raise ValueError("this native adapter requires explicitly known-flat initial_positions={}")
    capital = _number(initial_capital, "initial_capital")
    if capital <= 0:
        raise ValueError("initial capital must be positive")
    interval = _clock(candle_interval_ms, "candle_interval_ms")
    if interval <= 0:
        raise ValueError("candle_interval_ms must be positive")
    timestamps = [_clock(value, "candle timestamp") for value in candle_timestamps_ms]
    if len(timestamps) < 2 or any(right - left != interval for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("prepared candle timestamps must cover the full declared grid")
    if (type(first_bar_index) is not int or type(last_bar_index) is not int or
            not 0 <= first_bar_index <= last_bar_index < len(timestamps)):
        raise ValueError("expected first/last bar indices must explicitly select prepared coverage")
    if (not isinstance(bootstrap_bar_indices, (list, tuple)) or
            any(type(index) is not int or not 0 <= index < first_bar_index for index in bootstrap_bar_indices)
            or list(bootstrap_bar_indices) != sorted(set(bootstrap_bar_indices))):
        raise ValueError("bootstrap bar indices must explicitly identify unique ordered pre-evaluation rows")
    if (not isinstance(coins, (list, tuple)) or not coins or
            any(not isinstance(coin, str) or not coin for coin in coins) or len(set(coins)) != len(coins)):
        raise ValueError("ordered coins must be explicit, nonempty and unique")
    candles = np.asarray(hlcvs)
    if candles.ndim != 3 or candles.shape[:2] != (len(timestamps), len(coins)) or candles.shape[2] < 3:
        raise ValueError("prepared HLCV shape must match the complete timestamp/coin grid")
    if not isinstance(market_settings, Mapping):
        raise ValueError("market settings must be an explicit per-coin mapping")
    multipliers = {}
    coin_indices = {coin: index for index, coin in enumerate(coins)}
    for coin in coins:
        if (coin not in market_settings or not isinstance(market_settings[coin], Mapping)
                or "c_mult" not in market_settings[coin]):
            raise ValueError(f"missing contract multiplier for {coin}")
        multiplier = _number(market_settings[coin]["c_mult"], f"{coin}.c_mult")
        if multiplier <= 0:
            raise ValueError("contract multipliers must be positive")
        multipliers[coin] = multiplier
    if native_fills is None or native_equities is None:
        raise ValueError("full native fills/equities are required; metrics-only output is unsupported")
    equities = np.asarray(native_equities)
    if equities.ndim != 2 or equities.shape[0] < 1 or equities.shape[1] < 2:
        raise ValueError("full native ordinary-account equity rows are required")
    observed_indices = []
    account_values = []
    for row in equities:
        timestamp = _clock(row[0], "native equity timestamp")
        offset = timestamp - timestamps[0]
        if offset < 0 or offset % interval:
            raise ValueError("native equity timestamp is outside the prepared candle grid")
        observed_indices.append(offset // interval)
        account_values.append(_number(row[1], "ordinary account equity"))
    bootstrap_count = len(bootstrap_bar_indices)
    if (observed_indices[:bootstrap_count] != list(bootstrap_bar_indices) or
            any(value != capital for value in account_values[:bootstrap_count])):
        raise ValueError("declared native bootstrap rows must retain known-flat initial capital")
    observed_indices = observed_indices[bootstrap_count:]
    account_values = account_values[bootstrap_count:]
    if not observed_indices:
        raise ValueError("native bootstrap rows do not supply evaluated account coverage")
    final_bar = observed_indices[-1]
    if (final_bar > last_bar_index or
            observed_indices != list(range(first_bar_index, final_bar + 1))):
        raise ValueError("native equity rows must be a contiguous prefix of the expected full grid")
    if execution_completed and not liquidated and final_bar != last_bar_index:
        raise ValueError("successful non-liquidated native replay omitted required account observations")

    parsed_fills = []
    previous_index = first_bar_index
    for sequence, row in enumerate(native_fills):
        if len(row) < 14:
            raise ValueError("native fill rows require index/timestamp/coin/PnL/fee/qty/price/position/type columns")
        index = _clock(row[0], "fill index")
        timestamp = _clock(row[1], "fill timestamp")
        coin = row[2]
        order_type = row[13]
        if not first_bar_index <= index <= final_bar or index < previous_index:
            raise ValueError("fills must retain native bar order and lie inside observed account coverage")
        if not timestamps[index] <= timestamp <= timestamps[index] + interval:
            raise ValueError("fill timestamp is inconsistent with its native bar index")
        previous_index = index
        if not isinstance(coin, str) or coin not in multipliers:
            raise ValueError("fill coin is outside the frozen market universe")
        if (not isinstance(order_type, str) or order_type.rsplit("_", 1)[-1] not in {"long", "short"}
                or not order_type.startswith(("entry_", "close_"))):
            raise ValueError("native fill requires explicit entry/close and position-side attribution")
        pside = order_type.rsplit("_", 1)[-1]
        qty = _number(row[9], "fill signed quantity")
        price = _number(row[10], "fill price")
        post_size = _number(row[11], "post-fill signed position")
        pnl = _number(row[3], "fill realized PnL")
        fee = _number(row[4], "fill signed fee cashflow")
        close = order_type.startswith("close_")
        if qty == 0 or price <= 0 or (post_size < 0 if pside == "long" else post_size > 0):
            raise ValueError("native signed quantities/position-side/positive price are inconsistent")
        if (qty > 0) != ((pside == "long") != close):
            raise ValueError("native fill quantity sign contradicts entry/close position side")
        parsed_fills.append(dict(index=index, timestamp_ms=timestamp, coin=coin, pside=pside,
                                 qty=qty, post_size=post_size, sequence=sequence,
                                 identity=f"native-fill:{sequence}", net_realized_pnl=pnl + fee,
                                 fee_cashflow=fee, notional=abs(qty) * price * multipliers[coin],
                                 is_close=close))

    positions = {}
    exposure = [0.0]
    cursor = 0
    for index in observed_indices:
        while cursor < len(parsed_fills) and parsed_fills[cursor]["index"] <= index:
            fill = parsed_fills[cursor]
            key = (fill["coin"], fill["pside"])
            before = positions.get(key, 0.0)  # Proven flat initialization above.
            projected = before + fill["qty"]
            if not math.isfinite(projected):
                raise ValueError("native signed position transition overflows")
            tolerance = 16 * max(math.ulp(value) for value in (before, fill["qty"], projected, fill["post_size"]))
            if abs(projected - fill["post_size"]) > tolerance:
                raise ValueError("native fill tape omits or contradicts signed position transitions")
            positions[key] = fill["post_size"]
            cursor += 1
        notionals = []
        for (coin, _), size in positions.items():
            if size == 0:
                continue
            mark = _number(candles[index, coin_indices[coin], 2], f"held {coin} close mark")
            if mark <= 0:
                raise ValueError("held inventory requires a positive causal close mark")
            notionals.append(abs(size) * mark * multipliers[coin])
        exposure.append(math.fsum(notionals) / capital)

    start = timestamps[first_bar_index]
    # Sort only commutative reporting cashflows/activity, never reconstruct
    # positions in timestamp order across a next-bar boundary.
    reporting_fills = sorted(parsed_fills, key=lambda fill: (fill["timestamp_ms"], fill["sequence"]))
    result = extract_account_metrics(
        start_ms=start, expected_end_ms=timestamps[last_bar_index] + interval,
        initial_equity=capital, timestamps_ms=[start, *(timestamps[i] + interval for i in observed_indices)],
        equity=[capital, *account_values], gross_exposure_ratio=exposure, fills=reporting_fills,
        completed=execution_completed and final_bar == last_bar_index, liquidated=liquidated,
        sample_interval_ms=interval)
    if account_valuation_input_identity(**prepared) != valuation_input_sha256:
        raise ValueError("prepared valuation inputs changed while extracting account metrics")
    result.update(valuation_input_sha256=valuation_input_sha256,
                  exposure_denominator="initial_capital", exposure_denominator_value=capital,
                  exposure_sampling="left-held causal close samples; known-flat initial baseline",
                  native_equity_clock="bar start k stores close[k]; observed at end(k+1)",
                  first_native_bar_index=first_bar_index, last_native_bar_index=final_bar,
                  excluded_bootstrap_bar_indices=list(bootstrap_bar_indices))
    return result


def cpu_confirmation_from_native(native_result, *, model, effective_config, input_sha256,
                                 runtime_sha256, elapsed_seconds, **replay_inputs):
    """Build DiscoveryCounter evidence, sampling the cold clock after extraction.

    The caller runs the matching version's CPU extension against the frozen
    scenario and verifies its runtime identity. This bridge consumes its outputs;
    it neither substitutes a current runner nor launches any work.
    """
    _sha(input_sha256, "input_sha256")
    _sha(runtime_sha256, "runtime_sha256")
    if not isinstance(native_result, (list, tuple)) or len(native_result) < 3:
        raise ValueError("full native result tuple is required")
    analysis = native_result[2]
    if not isinstance(analysis, Mapping) or type(analysis.get("liquidated")) is not bool:
        raise ValueError("native analysis must supply authoritative liquidation status")
    metrics = metrics_from_native_replay(native_fills=native_result[0], native_equities=native_result[1],
                                         liquidated=analysis["liquidated"], **replay_inputs)
    if not callable(elapsed_seconds):
        raise ValueError("elapsed_seconds must sample the common-origin clock after confirmation work")
    finished = _number(elapsed_seconds(), "confirmation elapsed_seconds")
    if finished < 0:
        raise ValueError("confirmation elapsed clock must be nonnegative")
    return dict(candidate_sha256=canonical_configuration_identity(effective_config, model=model),
                input_sha256=input_sha256, runtime_sha256=runtime_sha256, engine="cpu",
                completed_elapsed_seconds=finished, metrics=metrics)
