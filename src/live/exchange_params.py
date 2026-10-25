"""Exchange settings shared by ordinary and protective order construction."""
import math


def _get_exchange_fee_fractions(self, symbol: str) -> tuple[float, float]:
    """Read unchanged CCXT/cache keys and expose signed scale-one fee fractions."""
    market = self.markets_dict[symbol]
    maker_fee_fraction = market.get("maker_fee")
    if maker_fee_fraction is None:
        maker_fee_fraction = market.get("maker")
    taker_fee_fraction = market.get("taker_fee")
    if taker_fee_fraction is None:
        taker_fee_fraction = market.get("taker")
    if maker_fee_fraction is None:
        raise ValueError(f"missing maker_fee_fraction for {symbol}")
    if taker_fee_fraction is None:
        raise ValueError(f"missing taker_fee_fraction for {symbol}")
    maker_fee_fraction = float(maker_fee_fraction)
    taker_fee_fraction = float(taker_fee_fraction)
    if not math.isfinite(maker_fee_fraction):
        raise ValueError(f"maker_fee_fraction must be finite for {symbol}, got {maker_fee_fraction}")
    if not math.isfinite(taker_fee_fraction):
        raise ValueError(f"taker_fee_fraction must be finite for {symbol}, got {taker_fee_fraction}")
    return maker_fee_fraction, taker_fee_fraction


def _orchestrator_exchange_params(self, symbol: str) -> dict:
    maker_fee_fraction, taker_fee_fraction = self._get_exchange_fee_fractions(symbol)
    return {
        "qty_step": float(self.qty_steps[symbol]),
        "price_step": float(self.price_steps[symbol]),
        "min_qty": float(self.min_qtys[symbol]),
        "min_cost": float(self.min_costs[symbol]),
        "c_mult": float(self.c_mults[symbol]),
        "maker_fee_fraction": float(maker_fee_fraction),
        "taker_fee_fraction": float(taker_fee_fraction),
    }
