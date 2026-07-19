import ccxt.async_support as ccxt_async
import ccxt.pro as ccxt_pro

import utils


def test_gateio_ccxt_id_resolves_current_rest_and_pro_clients():
    ccxt_id = utils.to_ccxt_exchange_id("gateio")

    assert ccxt_id == "gate"
    assert getattr(ccxt_async, ccxt_id).__name__ == "gate"
    assert getattr(ccxt_pro, ccxt_id).__name__ == "gate"
    assert utils.to_standard_exchange_name(ccxt_id) == "gateio"
