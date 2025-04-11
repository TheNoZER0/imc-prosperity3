# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Basket Analysis Dashboard")

@st.cache_data
def load_data():
    # Read the orderbook CSV with semicolon delimiter.
    df = pd.read_csv('../data/round2/prices_combined_r2.csv', sep=';')
    df['day'] = df['day'].astype(int)
    df['timestamp'] = df['timestamp'].astype(int)
    df['mid_price'] = df['mid_price'].astype(float)
    return df

def get_mid_price_series(df, product):
    sub = df[df['product'] == product][['day','timestamp','mid_price']].copy()
    sub['time'] = sub['day'].astype(str) + '-' + sub['timestamp'].astype(str)
    sub = sub.set_index('time')
    return sub['mid_price']

# Load data
df = load_data()

# Build time series for components and baskets
mid_CROISSANTS   = get_mid_price_series(df, 'CROISSANTS')
mid_JAMS         = get_mid_price_series(df, 'JAMS')
mid_DJEMBES      = get_mid_price_series(df, 'DJEMBES')
mid_BASKET1 = get_mid_price_series(df, 'PICNIC_BASKET1')
mid_BASKET2 = get_mid_price_series(df, 'PICNIC_BASKET2')

# Merge into a DataFrame
df_prices = pd.DataFrame({
    'Croissants': mid_CROISSANTS,
    'Jams': mid_JAMS,
    'Djembes': mid_DJEMBES,
    'Basket1': mid_BASKET1,
    'Basket2': mid_BASKET2
}).dropna()

# Calculate theoretical fair values:
df_prices['Theo_Basket1'] = 6 * df_prices['Croissants'] + 3 * df_prices['Jams'] + 1 * df_prices['Djembes']
df_prices['Theo_Basket2'] = 4 * df_prices['Croissants'] + 2 * df_prices['Jams']
df_prices['Spread_Basket1'] = df_prices['Basket1'] - df_prices['Theo_Basket1']
df_prices['Spread_Basket2'] = df_prices['Basket2'] - df_prices['Theo_Basket2']

# User controls
st.sidebar.header("User Controls")
selected_basket = st.sidebar.selectbox("Select Basket", ["PICNIC_BASKET1", "PICNIC_BASKET2"])
z_threshold = st.sidebar.slider("Z-Score Entry Threshold", 1.0, 3.0, 2.0, 0.1)
window = st.sidebar.number_input("Rolling Window Size", value=500, min_value=10, max_value=10000)

# Calculate rolling z-score
def rolling_zscore(series, window=500):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

if selected_basket == "PICNIC_BASKET1":
    df_prices['Zscore'] = rolling_zscore(df_prices['Spread_Basket1'], window)
    actual = df_prices['Basket1']
    theo = df_prices['Theo_Basket1']
    spread = df_prices['Spread_Basket1']
else:
    df_prices['Zscore'] = rolling_zscore(df_prices['Spread_Basket2'], window)
    actual = df_prices['Basket2']
    theo = df_prices['Theo_Basket2']
    spread = df_prices['Spread_Basket2']

# Create Plotly figure with two subplots: price panel and spread z-score panel.
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.08, row_heights=[0.6, 0.4])
times = df_prices.index

# Price plot
fig.add_trace(go.Scatter(x=times, y=actual, mode='lines', name='Actual Price'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=times, y=theo, mode='lines', name='Theoretical Price', line=dict(dash='dash')),
              row=1, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)

# Spread z-score plot
fig.add_trace(go.Scatter(x=times, y=df_prices['Zscore'], mode='lines', name='Z-Score'),
              row=2, col=1)
fig.add_hline(y=z_threshold, line=dict(color='red', dash='dot'), row=2, col=1)
fig.add_hline(y=-z_threshold, line=dict(color='white', dash='dot'), row=2, col=1)
fig.update_yaxes(title_text="Z-Score", row=2, col=1)

fig.update_layout(title_text=f"{selected_basket}: Actual vs Theoretical Price and Spread Z-Score",
                  template='plotly_dark', xaxis=dict(tickangle=45))
st.plotly_chart(fig, use_container_width=True)

# Show current recommendation
current_z = df_prices['Zscore'].dropna().iloc[-1]
if current_z > z_threshold:
    rec = f"{selected_basket} appears overpriced (Z-score = {current_z:.2f}). Consider shorting the basket and buying components."
elif current_z < -z_threshold:
    rec = f"{selected_basket} appears underpriced (Z-score = {current_z:.2f}). Consider buying the basket and shorting components."
else:
    rec = f"{selected_basket} is within the normal range (Z-score = {current_z:.2f}). No strong trade signal."
st.markdown("### Trading Recommendation")
st.info(rec)

# Compute and display ADF test for stationarity on the spread
from statsmodels.tsa.stattools import adfuller
spread_series = spread.dropna()
adf_result = adfuller(spread_series)
st.write("**ADF Statistic:** {:.4f}".format(adf_result[0]))
st.write("**ADF p-value:** {:.4f}".format(adf_result[1]))

# (Optional) display correlation heatmap for all instruments using st.pyplot
st.markdown("### Correlation Matrix")
import matplotlib.pyplot as plt
import seaborn as sns
df_returns = np.log(df_prices[['Croissants', 'Jams', 'Djembes', 'Basket1', 'Basket2']]).diff().dropna()
plt.figure(figsize=(6,5))
sns.heatmap(df_returns.corr(), annot=True, cmap='coolwarm', center=0)
st.pyplot(plt.gcf())

# (Optional) You can include further panels such as a backtest performance chart.
st.markdown("### Note")
st.write("This dashboard is designed to mimic a Bloomberg‐style terminal for analysing basket trading opportunities. "
         "It uses theoretical fair value models, spread z-scores, and stationarity tests to suggest trade signals. "
         "Feel free to adjust parameters and backtest your strategy using the Jupyter Notebook.")
