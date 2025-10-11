import logging

from candlestick_manager import CandlestickManager


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_debug_level_network_only_filters_non_ccxt():
    cm = CandlestickManager(exchange=None, exchange_name="ex", debug=1)
    cap = _Capture()
    cm.log.addHandler(cap)
    cm.log.setLevel(logging.DEBUG)

    # Non-network debug should be filtered at level 1
    cm._log("debug", "saved_range", symbol="X")
    # Network debug should pass
    cm._log("debug", "ccxt_fetch_ohlcv", symbol="X")

    msgs = cap.messages
    assert any("event=ccxt_fetch_ohlcv" in m for m in msgs)
    assert not any("event=saved_range" in m for m in msgs)


def test_debug_level_full_allows_all_debug():
    cm = CandlestickManager(exchange=None, exchange_name="ex", debug=2)
    cap = _Capture()
    cm.log.addHandler(cap)
    cm.log.setLevel(logging.DEBUG)

    cm._log("debug", "saved_range", foo=1)
    cm._log("debug", "ccxt_fetch_ohlcv", bar=2)

    msgs = cap.messages
    assert any("event=ccxt_fetch_ohlcv" in m for m in msgs)
    assert any("event=saved_range" in m for m in msgs)
