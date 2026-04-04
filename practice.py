from pycoingecko import CoinGeckoAPI
import pandas as pd
import plotly.graph_objects as go


cg = CoinGeckoAPI()


bitcoin_data = cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='inr', days=30)

df = pd.DataFrame(bitcoin_data['prices'], columns=['timestamp', 'price'])
df['time'] = pd.to_datetime(df['timestamp'], unit='ms')

df['date'] = df['time'].dt.date
candles = df.groupby('date')['price'].agg(['first', 'max', 'min', 'last']).reset_index()

# Create candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=candles['date'],
    open=candles['first'],
    high=candles['max'],
    low=candles['min'],
    close=candles['last']
)])

fig.update_layout(
    title='Bitcoin Candlestick Chart Over Past 30 Days',
    xaxis_title='Date',
    yaxis_title='Price (INR RS)',
    xaxis_rangeslider_visible=False
)

fig.show()
