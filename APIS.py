from pycoingecko import CoinGeckoAPI
import pandas as pd

cg = CoinGeckoAPI()

bitcoin_data = cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='inr', days=2)

# You don’t need bitcoin_data[] — just convert the 'prices' part directly
df = pd.DataFrame(bitcoin_data['prices'], columns=['time', 'price'])

# ✅ Use pandas to convert time column properly (timestamps are in milliseconds)
df['time'] = pd.to_datetime(df['time'], unit='ms')

# Now you can iterate as you wanted
for time, price in zip(df['time'], df['price']):
    print(f"Time: {time}, Price: ₹{price:,.2f}\n")



