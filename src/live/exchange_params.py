"""Exchange settings shared by ordinary and protective order construction."""
import math


def _get_exchange_fee_rates(self, symbol: str) -> tuple[float, float]:
    market = self.markets_dict[symbol]
    maker_fee = market.get("maker_fee")
    if maker_fee is None:
        maker_fee = market.get("maker")
    taker_fee = market.get("taker_fee")
    if taker_fee is None:
        taker_fee = market.get("taker")
    if maker_fee is None:
        raise ValueError(f"missing maker_fee for {symbol}")
    if taker_fee is None:
        raise ValueError(f"missing taker_fee for {symbol}")
    maker_fee = float(maker_fee)
    taker_fee = float(taker_fee)
    if not math.isfinite(maker_fee):
        raise ValueError(f"maker_fee must be finite for {symbol}, got {maker_fee}")
    if not math.isfinite(taker_fee):
        raise ValueError(f"taker_fee must be finite for {symbol}, got {taker_fee}")
    return maker_fee, taker_fee


def _orchestrator_exchange_params(self, symbol: str) -> dict:
    maker_fee, taker_fee = self._get_exchange_fee_rates(symbol)
    return {
        "qty_step": float(self.qty_steps[symbol]),
        "price_step": float(self.price_steps[symbol]),
        "min_qty": float(self.min_qtys[symbol]),
        "min_cost": float(self.min_costs[symbol]),
        "c_mult": float(self.c_mults[symbol]),
        "maker_fee": float(maker_fee),
        "taker_fee": float(taker_fee),
    }
