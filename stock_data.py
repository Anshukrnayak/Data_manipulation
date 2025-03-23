import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime

st.title('SMA Trading Strategy Backtest')

# User inputs
st.sidebar.header('Strategy Parameters')
ticker = st.sidebar.text_input('Stock Ticker', 'AAPL')
start_date = st.sidebar.date_input('Start Date', datetime(2020, 1, 1))
end_date = st.sidebar.date_input('End Date', datetime(2023, 1, 1))
short_sma = st.sidebar.number_input('Short SMA', min_value=1, value=50)
long_sma = st.sidebar.number_input('Long SMA', min_value=1, value=200)

st.write(f'**Analyzing {ticker} from {start_date} to {end_date} with SMA {short_sma}/{long_sma}**')

# 1. Get historical data with error handling
try:
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error('No data downloaded - check ticker and date range')
    else:
        # Drop rows with NaN or non-numeric data
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        data = data.astype(float)  # Ensure all values are float
        st.write(f'Data Loaded: {len(data)} rows')
except Exception as e:
    st.error(f'Error downloading data: {e}')

# 2. Create strategy parameters with validation
if len(data) >= long_sma:
    data['SMA50'] = data['Close'].rolling(window=short_sma).mean()
    data['SMA200'] = data['Close'].rolling(window=long_sma).mean()
    data['Signal'] = np.where(data['SMA50'] > data['SMA200'], 1, 0)
    data['Position'] = data['Signal'].diff().fillna(0)

    # 3. Backtesting the strategy
    data['Market Returns'] = data['Close'].pct_change().fillna(0)
    data['Strategy Returns'] = data['Market Returns'] * data['Signal'].shift(1).fillna(0)
    data['Cumulative Market Returns'] = (1 + data['Market Returns']).cumprod()
    data['Cumulative Strategy Returns'] = (1 + data['Strategy Returns']).cumprod()

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Cumulative Market Returns'], label='Buy & Hold', linewidth=2)
    ax.plot(data['Cumulative Strategy Returns'], label='SMA Strategy', linewidth=2)
    ax.set_title('Strategy Backtest Results')
    ax.set_ylabel('Cumulative Returns')
    ax.set_xlabel('Date')
    ax.legend()
    st.pyplot(fig)

    # Candlestick Chart
    buy_signals = data[data['Position'] == 1]
    sell_signals = data[data['Position'] == -1]
    apd = [
        mpf.make_addplot(data['SMA50'], color='blue'),
        mpf.make_addplot(data['SMA200'], color='orange')
    ]
    if not buy_signals.empty:
        apd.append(mpf.make_addplot(buy_signals['Low'] * 0.99, type='scatter', markersize=100, marker='^', color='g'))
    if not sell_signals.empty:
        apd.append(mpf.make_addplot(sell_signals['High'] * 1.01, type='scatter', markersize=100, marker='v', color='r'))
    try:
        fig, ax = mpf.plot(data, type='candle', style='charles', title=f'{ticker} Price with {short_sma}/{long_sma} SMA Strategy', ylabel='Price ($)', addplot=apd, volume=False, returnfig=True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f'Plotting error: {e}')
else:
    st.warning(f'Not enough data points for {long_sma}-day SMA')
