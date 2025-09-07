import asyncio
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [29, 18]   


from utils import load_ccxt_instance, coin_to_symbol, utc_ms, ts_to_date
from candlestick_manager import CandlestickManager
ONE_DAY_MS = 1000 * 60 * 60 * 24

async def test_suite(cm, ex):
    print("test 1 basic fetch, 2 days\n")
    
    coin = "BTC"
    symbol = coin_to_symbol(coin, ex)
    end_ts = int(utc_ms() - ONE_DAY_MS * 2)
    start_ts = end_ts - ONE_DAY_MS * 2
    candles = await cm.get_candles(symbol, start_ts=start_ts, end_ts=end_ts, strict=True)
    df = pd.DataFrame(candles)
    print(len(df), df.ts.diff().max())
    print('given start date   ', ts_to_date(start_ts))
    print('returned start date', ts_to_date(df.iloc[0].ts))
    print('given end date     ', ts_to_date(end_ts))
    print('returned end date  ', ts_to_date(df.iloc[-1].ts))
    print('biggest gap in minutes', df.ts.diff().max() / 60000)
    zero_candles = df[df.bv == 0.0]
    print('zero_candles', zero_candles)

    print("test 2 basic fetch, end_ts=None\n")
    start_ts = int(utc_ms() - ONE_DAY_MS * 2)
    end_ts = None
    candles = await cm.get_candles(symbol, start_ts=start_ts, end_ts=end_ts, strict=True)
    df = pd.DataFrame(candles)
    print('n candles fetched', len(df))
    print('max ts diff', df.ts.diff().max())
    print(len(df), df.ts.diff().max())
    print('given start date   ', ts_to_date(start_ts))
    print('returned start date', ts_to_date(df.iloc[0].ts))
    print('current end date   ', ts_to_date(utc_ms()))
    print('returned end date  ', ts_to_date(df.iloc[-1].ts))
    print('biggest gap in minutes', df.ts.diff().max() / 60000)
    zero_candles = df[df.bv == 0.0]
    print('zero_candles', zero_candles)

    print("test 3 EMAs\n")
    
    span = 1234.0
    ema_close = await cm.get_latest_ema_close(symbol, span)
    ema_nrr = await cm.get_latest_ema_nrr(symbol, span)
    ema_volume = await cm.get_latest_ema_volume(symbol, span)
    print('ema_close', ema_close)
    print('ema_nrr', ema_nrr)
    print('ema_volume', ema_volume)
    
    return df



async def main():
    # tests for candlestick manager
    exchanges = ['binance', 'bybit', 'bitget', 'okx', 'gateio', 'kucoin', 'hyperliquid']
    for ex in exchanges:
        cc = load_ccxt_instance(ex)
        cm = CandlestickManager(cc, debug=True)
        df = await test_suite(cm, ex)
        try:
            await cm.exchange.close()
        except:
            pass
        break



if __name__ == '__main__':
    asyncio.run(main())