import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch historical BTC price data from Binance
def fetch_historical_data(symbol='NOTUSDT', timeframe='1h', since=None, limit=1000):
    binance = ccxt.binance()
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, since, limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Fetch data for the last month
since = int((pd.Timestamp.now() - pd.DateOffset(months=1)).timestamp() * 1000)

# Fetch data for the last month
df = fetch_historical_data()

# Display the first few rows of the data
print(df.head())

# Define the configuration based on the provided parameters
config = {
    'grid_span': [0.012, 0.02],
    'initial_qty_pct': [0.005, 0.02],
    'n_close_orders': [2, 16]
}

def recursive_grid_trading(prices, config):
    balance = 10000  # Initial balance in USD
    btc_held = 0     # Initial BTC held
    trade_log = []   # Log of trades

    grid_levels = np.linspace(config['grid_span'][0], config['grid_span'][1], config['n_close_orders'][1])

    for price in prices:
        # Example logic: Buy BTC if price is below a certain level
        for level in grid_levels:
            if price < level:
                qty = balance * config['initial_qty_pct'][1] / price
                btc_held += qty
                balance -= qty * price
                trade_log.append(('BUY', price, qty))
        
        # Example logic: Sell BTC if price is above a certain level
        for level in grid_levels:
            if price > level:
                qty = btc_held * config['initial_qty_pct'][1]
                btc_held -= qty
                balance += qty * price
                trade_log.append(('SELL', price, qty))

    return balance, btc_held, trade_log

# Run the simulation with the BTC price data
balance, btc_held, trade_log = recursive_grid_trading(df['close'], config)

# Convert trade log to a DataFrame for analysis
trade_df = pd.DataFrame(trade_log, columns=['Action', 'Price', 'Quantity'])

# Calculate the profit/loss for each trade
trade_df['Value'] = trade_df['Price'] * trade_df['Quantity']
trade_df['Profit/Loss'] = trade_df.apply(lambda row: row['Value'] if row['Action'] == 'SELL' else -row['Value'], axis=1)

# Calculate cumulative profit/loss
trade_df['Cumulative P/L'] = trade_df['Profit/Loss'].cumsum()

# Display the trade log and final cumulative profit/loss
print(trade_df.head())

# Plot cumulative profit/loss over time
# Plot cumulative profit/loss over time
# plt.figure(figsize=(10, 5))
# plt.plot(trade_df['Cumulative P/L'], label='Cumulative Profit/Loss')
# plt.xlabel('Trades')
# plt.ylabel('Cumulative Profit/Loss (USD)')
# plt.title('Cumulative Profit/Loss Over Time')
# plt.legend()
# plt.grid(True)
# plt.show()

# Plot price data with buy and sell points
plt.figure(figsize=(15, 7))
plt.plot(trade_df['Price'], label='Price', color='blue')
buy_signals = trade_df[trade_df['Action'] == 'BUY']
sell_signals = trade_df[trade_df['Action'] == 'SELL']
plt.scatter(buy_signals.index, buy_signals['Price'], marker='^', color='green', label='Buy Signal', alpha=1)
plt.scatter(sell_signals.index, sell_signals['Price'], marker='v', color='red', label='Sell Signal', alpha=1)
plt.title('Price with Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Display results
print(f'Final Balance: ${balance:.2f}')
print(f'BTC Held: {btc_held:.4f} BTC')
print(f'Final Cumulative P/L: ${trade_df["Cumulative P/L"].iloc[-1]:.2f}')
