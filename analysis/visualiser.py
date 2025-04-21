import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

st.set_page_config(page_title="IMC Basket Visualiser", layout="wide")

# ---------------------------------------
# Helper Functions
# ---------------------------------------

def hurst_exponent(ts):
    """Return the Hurst Exponent of the time series ts."""
    lags = range(2, min(100, len(ts)//2))
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return 2.0 * poly[0]

def rolling_zscore(series, window=50):
    rolling_mean = series.rolling(window).mean()
    rolling_std  = series.rolling(window).std()
    zscore = (series - rolling_mean) / rolling_std
    return zscore

# Functions to extract order book information
def get_best_bid(row):
    """Return best bid from bid_price_1-3."""
    bid_prices = [row.get(f'bid_price_{i}', np.nan) for i in range(1, 4)]
    valid_bids = [p for p in bid_prices if not pd.isna(p)]
    return max(valid_bids) if valid_bids else np.nan

def get_best_ask(row):
    """Return best ask from ask_price_1-3."""
    ask_prices = [row.get(f'ask_price_{i}', np.nan) for i in range(1, 4)]
    valid_asks = [p for p in ask_prices if not pd.isna(p)]
    return min(valid_asks) if valid_asks else np.nan

def calculate_spread(row):
    """Compute the spread = best ask - best bid for a row."""
    best_bid = get_best_bid(row)
    best_ask = get_best_ask(row)
    if pd.isna(best_bid) or pd.isna(best_ask):
        return np.nan
    return best_ask - best_bid

def get_mid_price_series(df, product):
    sub = df[df['product'] == product][['day','timestamp','mid_price']].copy()
    sub['time'] = sub['day'].astype(str) + '-' + sub['timestamp'].astype(str)
    sub = sub.sort_values(['day','timestamp']).set_index('time')
    return sub['mid_price']

def get_spread_series(df, product):
    """
    For a given product, compute the order book spread (best ask minus best bid)
    and return a series indexed by a combined time key.
    """
    sub = df[df['product'] == product].copy()
    sub['spread'] = sub.apply(calculate_spread, axis=1)
    sub['time'] = sub['day'].astype(str) + '-' + sub['timestamp'].astype(str)
    sub = sub.sort_values(['day','timestamp']).set_index('time')
    return sub['spread']

def get_best_bid_series(df, product):
    sub = df[df['product'] == product].copy()
    sub['time'] = sub['day'].astype(str) + '-' + sub['timestamp'].astype(str)
    sub = sub.sort_values(['day','timestamp']).set_index('time')
    sub['best_bid'] = sub.apply(get_best_bid, axis=1)
    return sub['best_bid']

def get_best_ask_series(df, product):
    sub = df[df['product'] == product].copy()
    sub['time'] = sub['day'].astype(str) + '-' + sub['timestamp'].astype(str)
    sub = sub.sort_values(['day','timestamp']).set_index('time')
    sub['best_ask'] = sub.apply(get_best_ask, axis=1)
    return sub['best_ask']

def get_pnl_series(df, product):
    """
    Extract the profit_and_loss (PnL) series for a given product.
    Assumes the CSV column is named 'profit_and_loss'.
    """
    sub = df[df['product'] == product][['day', 'timestamp', 'profit_and_loss']].copy()
    sub['time'] = sub['day'].astype(str) + '-' + sub['timestamp'].astype(str)
    sub = sub.sort_values(['day','timestamp']).set_index('time')
    return sub['profit_and_loss']

# ---------------------------------------
# Statistical Test Functions
# ---------------------------------------

def run_adf(series, series_name="Series"):
    """
    Run Augmented Dickey-Fuller test on a series.
    Returns a string with test statistic, p-value, and critical values.
    """
    result = adfuller(series.dropna())
    out = (f"ADF Test for {series_name}:\n"
           f"Test Statistic: {result[0]:.4f}\n"
           f"p-value: {result[1]:.4f}\n"
           f"Critical Values: {result[4]}\n")
    return out

def run_johansen(prices_df):
    """
    Run Johansen's cointegration test on a DataFrame of prices.
    Returns the test output along with the cointegrating vectors.
    """
    result = coint_johansen(prices_df, det_order=0, k_ar_diff=1)
    out = "Johansen Cointegration Test:\n"
    for i, trace_stat in enumerate(result.lr1):
        cv = result.cvt[i]  # critical values for 90%, 95%, 99%
        out += (f"Rank <= {i}: Trace Statistic = {trace_stat:.4f}, Critical Values = {cv}\n")
    out += "\nCointegrating Vectors (columns):\n"
    for i in range(result.evec.shape[1]):
        vec = result.evec[:, i]
        # Normalize such that the first element is 1
        if vec[0] != 0:
            vec_norm = vec / vec[0]
        else:
            vec_norm = vec
        out += f"Relation {i+1}: {vec_norm}\n"
    return out

def compute_correlation(prices_df):
    """
    Compute and return the correlation matrix (log returns) of the price DataFrame.
    """
    returns = np.log(prices_df).diff().dropna()
    corr = returns.corr()
    return corr

# ---------------------------------------
# Plotting Functions (each returns one figure)
# ---------------------------------------

def plot_theoretical_actual_basket1(df_prices):
    """
    Produce one figure showing actual vs. theoretical prices for PICNIC_BASKET1.
    Theo_Basket1 = 6 * Croissants + 3 * Jams + 1 * Djembes.
    """
    df_plot = df_prices.copy()
    df_plot['Theo_Basket1'] = 6 * df_plot['Croissants'] + 3 * df_plot['Jams'] + 1 * df_plot['Djembes']
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df_plot.index, df_plot['Basket1'], label='Basket1 Actual', color='cyan')
    ax.plot(df_plot.index, df_plot['Theo_Basket1'], label='Basket1 Theoretical', linestyle='--', color='orange')
    ax.set_title('PICNIC_BASKET1: Actual vs. Theoretical Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def plot_theoretical_spread_basket1(df_prices):
    """
    Produce one figure for the price spread (Actual - Theoretical) for PICNIC_BASKET1.
    """
    df_plot = df_prices.copy()
    df_plot['Theo_Basket1'] = 6 * df_plot['Croissants'] + 3 * df_plot['Jams'] + 1 * df_plot['Djembes']
    df_plot['Spread_Basket1'] = df_plot['Basket1'] - df_plot['Theo_Basket1']
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df_plot.index, df_plot['Spread_Basket1'], label='Basket1 Price Spread', color='magenta')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title('PICNIC_BASKET1: Price Spread (Actual - Theoretical)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Spread')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def plot_theoretical_actual_basket2(df_prices):
    """
    Produce one figure showing actual vs. theoretical prices for PICNIC_BASKET2.
    Theo_Basket2 = 4 * Croissants + 2 * Jams.
    """
    df_plot = df_prices.copy()
    df_plot['Theo_Basket2'] = 4 * df_plot['Croissants'] + 2 * df_plot['Jams']
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df_plot.index, df_plot['Basket2'], label='Basket2 Actual', color='cyan')
    ax.plot(df_plot.index, df_plot['Theo_Basket2'], label='Basket2 Theoretical', linestyle='--', color='orange')
    ax.set_title('PICNIC_BASKET2: Actual vs. Theoretical Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def plot_theoretical_spread_basket2(df_prices):
    """
    Produce one figure for the price spread (Actual - Theoretical) for PICNIC_BASKET2.
    """
    df_plot = df_prices.copy()
    df_plot['Theo_Basket2'] = 4 * df_plot['Croissants'] + 2 * df_plot['Jams']
    df_plot['Spread_Basket2'] = df_plot['Basket2'] - df_plot['Theo_Basket2']
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df_plot.index, df_plot['Spread_Basket2'], label='Basket2 Price Spread', color='magenta')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title('PICNIC_BASKET2: Price Spread (Actual - Theoretical)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Spread')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def plot_zscore_basket1(df_prices, window=500):
    """
    Produce one figure for the rolling z-score of PICNIC_BASKET1's spread.
    """
    df_plot = df_prices.copy()
    if 'Spread_Basket1' not in df_plot.columns:
        df_plot['Theo_Basket1'] = 6 * df_plot['Croissants'] + 3 * df_plot['Jams'] + 1 * df_plot['Djembes']
        df_plot['Spread_Basket1'] = df_plot['Basket1'] - df_plot['Theo_Basket1']
        
    df_plot['Zscore_Basket1'] = rolling_zscore(df_plot['Spread_Basket1'], window)
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df_plot.index, df_plot['Zscore_Basket1'], label='Basket1 Spread Z-Score', color='black')
    ax.axhline(2, color='red', linestyle='--', label='Upper Threshold (2)')
    ax.axhline(-2, color='green', linestyle='--', label='Lower Threshold (-2)')
    ax.set_title('PICNIC_BASKET1 Spread Z-Score')
    ax.set_xlabel('Time')
    ax.set_ylabel('Z-Score')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def plot_zscore_basket2(df_prices, window=500):
    """
    Produce one figure for the rolling z-score of PICNIC_BASKET2's spread.
    """
    df_plot = df_prices.copy()
    if 'Spread_Basket2' not in df_plot.columns:
        df_plot['Theo_Basket2'] = 4 * df_plot['Croissants'] + 2 * df_plot['Jams']
        df_plot['Spread_Basket2'] = df_plot['Basket2'] - df_plot['Theo_Basket2']
        
    df_plot['Zscore_Basket2'] = rolling_zscore(df_plot['Spread_Basket2'], window)
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df_plot.index, df_plot['Zscore_Basket2'], label='Basket2 Spread Z-Score', color='blue')
    ax.axhline(2, color='red', linestyle='--', label='Upper Threshold (2)')
    ax.axhline(-2, color='green', linestyle='--', label='Lower Threshold (-2)')
    ax.set_title('PICNIC_BASKET2 Spread Z-Score')
    ax.set_xlabel('Time')
    ax.set_ylabel('Z-Score')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def plot_zscore_spread(df_prices, window=500):
    """
    Produce one figure for the spread (difference) between the z-scores of PICNIC_BASKET1 and PICNIC_BASKET2.
    """
    df_plot = df_prices.copy()
    df_plot['Theo_Basket1'] = 6 * df_plot['Croissants'] + 3 * df_plot['Jams'] + 1 * df_plot['Djembes']
    df_plot['Spread_Basket1'] = df_plot['Basket1'] - df_plot['Theo_Basket1']
    df_plot['Theo_Basket2'] = 4 * df_plot['Croissants'] + 2 * df_plot['Jams']
    df_plot['Spread_Basket2'] = df_plot['Basket2'] - df_plot['Theo_Basket2']
    
    df_plot['Zscore_Basket1'] = rolling_zscore(df_plot['Spread_Basket1'], window)
    df_plot['Zscore_Basket2'] = rolling_zscore(df_plot['Spread_Basket2'], window)
    df_plot['Zscore_Spread'] = df_plot['Zscore_Basket1'] - df_plot['Zscore_Basket2']
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df_plot.index, df_plot['Zscore_Spread'], label="Z-Score Spread (Basket1 - Basket2)", 
            color="purple", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title('Spread between Z-Scores of PICNIC_BASKET1 and PICNIC_BASKET2')
    ax.set_xlabel('Time')
    ax.set_ylabel('Z-Score Spread')
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def plot_pnl(df, product):
    """
    Produce one figure for the profit_and_loss (PnL) over time for a given product.
    Assumes the CSV column for PnL is 'profit_and_loss'.
    """
    sub = df[df['product'] == product].copy()
    sub['time'] = sub['day'].astype(str) + '-' + sub['timestamp'].astype(str)
    sub = sub.sort_values(['day','timestamp']).set_index('time')
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(sub.index, sub['profit_and_loss'], label=f'{product} PnL', color='darkorange')
    ax.set_xlabel('Time')
    ax.set_ylabel('Profit and Loss')
    ax.set_title(f'{product} Profit and Loss Over Time')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def plot_mid_price(df, product):
    """Plot mid_price over time for any product."""
    series = get_mid_price_series(df, product)
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(series.index, series, label=f'{product} Mid Price')
    ax.set_title(f'{product} Mid Price Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mid Price')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# def plot_cumulative_pnl(df, product):
#     """Plot cumulative PnL over time for any product."""
#     pnl = get_pnl_series(df, product)
#     cum_pnl = pnl.cumsum()
#     fig, ax = plt.subplots(figsize=(12,4))
#     ax.plot(cum_pnl.index, cum_pnl, label=f'{product} Cumulative PnL')
#     ax.set_title(f'{product} Cumulative PnL Over Time')
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Cumulative PnL')
#     ax.legend()
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     return fig


def get_vwap_for_row(row):
    """
    Compute the volume weighted average price (VWAP) for a given row of order book data
    by iterating through levels 1 to 3 for both bid and ask sides.
    
    For each side, we sum (price * volume) over all levels (ignoring NaNs)
    and then divide by the total volume across all levels.
    The overall VWAP is computed as the weighted average of both sides.
    """
    bid_total_value = 0.0
    bid_total_volume = 0.0
    for i in range(1, 4):
        price = row.get(f'bid_price_{i}', np.nan)
        volume = row.get(f'bid_volume_{i}', np.nan)
        if not pd.isna(price) and not pd.isna(volume):
            bid_total_value += price * volume
            bid_total_volume += volume

    ask_total_value = 0.0
    ask_total_volume = 0.0
    for i in range(1, 4):
        price = row.get(f'ask_price_{i}', np.nan)
        volume = row.get(f'ask_volume_{i}', np.nan)
        if not pd.isna(price) and not pd.isna(volume):
            ask_total_value += price * volume
            ask_total_volume += volume

    total_value = bid_total_value + ask_total_value
    total_volume = bid_total_volume + ask_total_volume
    if total_volume == 0:
        return np.nan
    return total_value / total_volume

def get_vwap_series(df, product):
    """
    Filters the DataFrame for a given product, computes the VWAP for each row 
    (using all bid/ask levels 1-3), and returns a Series indexed by a combined time key.
    """
    sub = df[df['product'] == product].copy()
    # Compute VWAP for each row using our custom function
    sub['vwap'] = sub.apply(get_vwap_for_row, axis=1)
    # Create a combined time key (e.g., "day-timestamp")
    sub['time'] = sub['day'].astype(str) + '-' + sub['timestamp'].astype(str)
    sub = sub.sort_values(['day', 'timestamp']).set_index('time')
    return sub['vwap']



def plot_vwap_and_spread_product(df, product):
    """
    Produce a figure with two subplots for a given product:
    - Top: The volume weighted average price (VWAP).
    - Bottom: The bid/ask spread (best ask minus best bid).
    
    Args:
        df (DataFrame): DataFrame containing the order book and price data.
        product (str): The product identifier (e.g. "RAINFOREST_RESIN").
    
    Returns:
        fig: The matplotlib figure object.
    """
    # Extract the VWAP and the spread series using existing helper functions.
    vwap_series = get_vwap_series(df, product)
    spread_series = get_spread_series(df, product)
    
    # Create a figure with two subplots (one for VWAP, one for spread).
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot the VWAP series.
    ax1.plot(vwap_series.index, vwap_series, label=f'{product} VWAP', color='blue')
    ax1.set_title(f'{product} Volume Weighted Average Price')
    ax1.set_ylabel('VWAP')
    ax1.legend()
    
    # Plot the spread series.
    ax2.plot(spread_series.index, spread_series, label=f'{product} Bid/Ask Spread', color='red')
    ax2.set_title(f'{product} Bid/Ask Spread')
    ax2.set_ylabel('Spread')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

# Wrapper functions for the three products:
def plot_resin_vwap_spread(df):
    """Plot VWAP and bid/ask spread for RAINFOREST_RESIN."""
    return plot_vwap_and_spread_product(df, "RAINFOREST_RESIN")

def plot_squid_vwap_spread(df):
    """Plot VWAP and bid/ask spread for SQUID_INK."""
    return plot_vwap_and_spread_product(df, "SQUID_INK")

def plot_kelp_vwap_spread(df):
    """Plot VWAP and bid/ask spread for KELP."""
    return plot_vwap_and_spread_product(df, "KELP")


def plot_zscore_product(df, product, window=50):
    """
    Produce a figure that plots the rolling z-score of the product's price (VWAP)
    relative to its rolling mean, computed over the given window.
    
    Args:
        df (DataFrame): DataFrame containing the data.
        product (str): The product identifier.
        window (int): The rolling window size.
    
    Returns:
        fig: The matplotlib figure object.
    """
    # Extract the price series using VWAP.
    price_series = get_vwap_series(df, product)
    # Compute the rolling z-score using the provided helper function.
    zscore_series = rolling_zscore(price_series, window)
    
    # Create the figure.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(price_series.index, zscore_series, label=f'{product} Rolling Z-Score', color='purple')
    ax.axhline(2, color='red', linestyle='--', label='Upper Threshold (2)')
    ax.axhline(-2, color='green', linestyle='--', label='Lower Threshold (-2)')
    ax.set_title(f'{product} Price Rolling Z-Score (Window = {window})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Z-Score')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

# Wrapper functions for z-score plots for each product:
def plot_zscore_resin(df, window=50):
    """Plot rolling z-score for RAINFOREST_RESIN."""
    return plot_zscore_product(df, "RAINFOREST_RESIN", window)

def plot_zscore_squid(df, window=50):
    """Plot rolling z-score for SQUID_INK."""
    return plot_zscore_product(df, "SQUID_INK", window)

def plot_zscore_kelp(df, window=50):
    """Plot rolling z-score for KELP."""
    return plot_zscore_product(df, "KELP", window)


# ---------------------------------------
# Statistical Test Plot / Output Functions
# ---------------------------------------

def display_adf_results(series, series_name="Series"):
    result = adfuller(series.dropna())
    st.markdown(f"**ADF Test for {series_name}:**")
    st.text(f"Test Statistic: {result[0]:.4f}")
    st.text(f"p-value: {result[1]:.4f}")
    st.text(f"Critical Values: {result[4]}")
    if result[1] < 0.05:
        st.success(f"{series_name} is likely stationary (reject H0).")
    else:
        st.error(f"{series_name} is likely non-stationary (fail to reject H0).")

def display_johansen_results(prices_df):
    result = coint_johansen(prices_df, det_order=0, k_ar_diff=1)
    st.markdown("**Johansen Cointegration Test Results:**")
    out = ""
    for i, trace_stat in enumerate(result.lr1):
        cv = result.cvt[i]  # critical values
        out += f"Rank <= {i}: Trace Statistic = {trace_stat:.4f}, Critical Values = {cv}\n"
    st.text(out)
    out_vec = "Cointegrating Vectors (normalized):\n"
    for i in range(result.evec.shape[1]):
        vec = result.evec[:, i]
        if vec[0] != 0:
            vec_norm = vec / vec[0]
        else:
            vec_norm = vec
        out_vec += f"Relation {i+1}: {vec_norm}\n"
    st.text(out_vec)

def display_correlation(prices_df):
    # Compute log returns and then correlation
    returns = np.log(prices_df).diff().dropna()
    corr = returns.corr()
    st.markdown("**Correlation Matrix (Log Returns):**")
    st.dataframe(corr)


# ---------------------------------------
# NEW: Trade Analysis Functions
# ---------------------------------------

def load_trade_data(filepath):
    """Loads the trade CSV, handling potential errors."""
    try:
        df = pd.read_csv(filepath, sep=';')
        # Basic cleaning and type conversion
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'price', 'quantity', 'buyer', 'seller', 'symbol'])

        # *** FIX: Calculate trade value HERE ***
        df['value'] = df['price'] * df['quantity'] # Moved this line up

        # Add a combined time column for sorting/plotting if 'day' exists
        if 'day' in df.columns:
            df['day'] = pd.to_numeric(df['day'], errors='coerce').fillna(0).astype(int)
            df['time_key'] = df['day'] * 1_000_000 + df['timestamp']
            df = df.sort_values('time_key')
            df['time_str'] = df['day'].astype(str) + '-' + df['timestamp'].astype(str)
        else:
            # If no 'day', just sort by timestamp
            df = df.sort_values('timestamp')
            df['time_str'] = df['timestamp'].astype(str)
            df['time_key'] = df['timestamp'] # Use timestamp if day is missing

        # Return AFTER calculating value
        return df
    except FileNotFoundError:
        st.error(f"Error: Trade data file not found at {filepath}")
        return None
    except Exception as e:
        st.error(f"Error loading trade data: {e}")
        return None

def analyze_trader_volume(trades_df):
    """Calculates total buy, sell, and net volume per trader per asset."""
    # Calculate Buy Volume
    buy_volume = trades_df.groupby(['buyer', 'symbol'])['quantity'].sum().reset_index()
    buy_volume = buy_volume.rename(columns={'buyer': 'trader', 'quantity': 'buy_volume'})

    # Calculate Sell Volume
    sell_volume = trades_df.groupby(['seller', 'symbol'])['quantity'].sum().reset_index()
    sell_volume = sell_volume.rename(columns={'seller': 'trader', 'quantity': 'sell_volume'})

    # Merge buy and sell volumes
    volume_analysis = pd.merge(buy_volume, sell_volume, on=['trader', 'symbol'], how='outer')
    volume_analysis = volume_analysis.fillna(0)

    # Calculate Total and Net Volume
    volume_analysis['total_volume'] = volume_analysis['buy_volume'] + volume_analysis['sell_volume']
    volume_analysis['net_volume'] = volume_analysis['buy_volume'] - volume_analysis['sell_volume']

    return volume_analysis.sort_values(by=['trader', 'total_volume'], ascending=[True, False])

def get_most_traded_assets(volume_analysis):
    """Identifies the most traded asset (by total volume) for each trader."""
    most_traded = volume_analysis.loc[volume_analysis.groupby('trader')['total_volume'].idxmax()]
    return most_traded[['trader', 'symbol', 'total_volume']].sort_values('total_volume', ascending=False)

def plot_most_traded_per_trader(most_traded_df):
    """Creates a bar chart of the most traded asset per trader."""
    if most_traded_df.empty:
        st.warning("No trade volume data to plot.")
        return None

    fig = px.bar(most_traded_df,
                 x='trader',
                 y='total_volume',
                 color='symbol',
                 title='Most Traded Asset per Trader (by Total Volume)',
                 labels={'total_volume': 'Total Volume Traded', 'trader': 'Trader', 'symbol': 'Asset Symbol'},
                 hover_data=['symbol', 'total_volume'])
    fig.update_layout(xaxis_title="Trader", yaxis_title="Total Volume Traded")
    return fig

def plot_trade_dynamics(trades_df, selected_trader, selected_asset):
    """Plots buy and sell trades for a specific trader and asset."""
    trader_asset_trades = trades_df[trades_df['symbol'] == selected_asset]
    buy_trades = trader_asset_trades[trader_asset_trades['buyer'] == selected_trader]
    sell_trades = trader_asset_trades[trader_asset_trades['seller'] == selected_trader]

    if buy_trades.empty and sell_trades.empty:
        st.warning(f"No trades found for Trader '{selected_trader}' in Asset '{selected_asset}'.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    if not buy_trades.empty:
        ax.scatter(buy_trades['timestamp'], buy_trades['price'],
                   s=buy_trades['quantity']*5, # Size by quantity
                   label=f'Buys ({selected_trader})', color='green', alpha=0.7, edgecolors='w')

    if not sell_trades.empty:
        ax.scatter(sell_trades['timestamp'], sell_trades['price'],
                   s=sell_trades['quantity']*5, # Size by quantity
                   label=f'Sells ({selected_trader})', color='red', alpha=0.7, marker='x') # Use 'x' for sells

    ax.set_title(f'Trade Dynamics for {selected_trader} - Asset: {selected_asset}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Price (SEASHELLS)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    return fig

# *** NEW: PNL Calculation Function ***
def calculate_trader_asset_pnl(trades_df):
    """Calculates approximate realized PNL per trader per asset."""
    if trades_df is None or trades_df.empty or 'value' not in trades_df.columns:
        st.warning("Trade data is missing or incomplete for PNL calculation.")
        return pd.DataFrame()

    # Calculate total value BOUGHT by each trader for each asset
    buy_value = trades_df.groupby(['buyer', 'symbol'])['value'].sum().reset_index()
    buy_value = buy_value.rename(columns={'buyer': 'trader', 'value': 'total_value_bought'})

    # Calculate total value SOLD by each trader for each asset
    sell_value = trades_df.groupby(['seller', 'symbol'])['value'].sum().reset_index()
    sell_value = sell_value.rename(columns={'seller': 'trader', 'value': 'total_value_sold'})

    # Merge buy and sell values
    pnl_analysis = pd.merge(buy_value, sell_value, on=['trader', 'symbol'], how='outer')
    pnl_analysis = pnl_analysis.fillna(0) # Fill NaN with 0 for traders who only bought or only sold

    # Calculate PNL (Value Sold - Value Bought)
    pnl_analysis['pnl'] = pnl_analysis['total_value_sold'] - pnl_analysis['total_value_bought']

    return pnl_analysis.sort_values(by=['trader', 'pnl'], ascending=[True, False])

# *** NEW: PNL Plotting Function ***
def plot_trader_asset_pnl(pnl_df):
    """Creates a bar chart of PNL per asset for each trader."""
    if pnl_df is None or pnl_df.empty:
        st.warning("No PNL data to plot.")
        return None

    # Ensure PNL is numeric
    pnl_df['pnl'] = pd.to_numeric(pnl_df['pnl'], errors='coerce')
    pnl_df = pnl_df.dropna(subset=['pnl'])

    if pnl_df.empty:
        st.warning("PNL data contains non-numeric values or is empty after cleaning.")
        return None

    # Sort for better visualization (optional, e.g., by total PNL per trader)
    pnl_summary = pnl_df.groupby('trader')['pnl'].sum().sort_values(ascending=False).index
    pnl_df['trader'] = pd.Categorical(pnl_df['trader'], categories=pnl_summary, ordered=True)
    pnl_df = pnl_df.sort_values('trader')

    fig = px.bar(pnl_df,
                 x='trader',
                 y='pnl',
                 color='symbol',
                 title='Approximate Realized PNL per Trader per Asset',
                 labels={'pnl': 'Profit and Loss (Value Sold - Value Bought)', 'trader': 'Trader', 'symbol': 'Asset'},
                 hover_data=['symbol', 'total_value_bought', 'total_value_sold', 'pnl'])

    fig.update_layout(xaxis_title="Trader",
                      yaxis_title="Approximate PNL",
                      xaxis={'categoryorder':'array', 'categoryarray': pnl_summary}) # Keep sorted order
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
    return fig

# --- In Trade Analysis Functions section (add this new function) ---

def calculate_trader_avg_pnl_per_unit(trades_df):
    """Calculates PNL based on average buy/sell prices per trader/asset."""
    if trades_df is None or trades_df.empty or 'value' not in trades_df.columns:
        st.warning("Trade data is missing or incomplete for Average PNL calculation.")
        return pd.DataFrame()

    # Calculate total value & quantity BOUGHT
    buy_stats = trades_df.groupby(['buyer', 'symbol']).agg(
        buy_value_sum=('value', 'sum'),
        buy_qty_sum=('quantity', 'sum')
    ).reset_index().rename(columns={'buyer': 'trader'})

    # Calculate total value & quantity SOLD
    sell_stats = trades_df.groupby(['seller', 'symbol']).agg(
        sell_value_sum=('value', 'sum'),
        sell_qty_sum=('quantity', 'sum')
    ).reset_index().rename(columns={'seller': 'trader'})

    # Merge stats
    avg_pnl_analysis = pd.merge(buy_stats, sell_stats, on=['trader', 'symbol'], how='outer')
    avg_pnl_analysis = avg_pnl_analysis.fillna(0)

    # Calculate average prices (handle division by zero)
    avg_pnl_analysis['avg_buy_price'] = np.where(
        avg_pnl_analysis['buy_qty_sum'] > 0,
        avg_pnl_analysis['buy_value_sum'] / avg_pnl_analysis['buy_qty_sum'],
        0
    )
    avg_pnl_analysis['avg_sell_price'] = np.where(
        avg_pnl_analysis['sell_qty_sum'] > 0,
        avg_pnl_analysis['sell_value_sum'] / avg_pnl_analysis['sell_qty_sum'],
        0
    )

    # Calculate PNL per unit traded (only where both buy and sell occurred)
    avg_pnl_analysis['avg_pnl_per_unit'] = np.where(
        (avg_pnl_analysis['buy_qty_sum'] > 0) & (avg_pnl_analysis['sell_qty_sum'] > 0),
        avg_pnl_analysis['avg_sell_price'] - avg_pnl_analysis['avg_buy_price'],
        0 # Assign 0 if only bought or only sold (no round trip implied)
    )

    # Calculate total approximate PNL based on the volume matched
    avg_pnl_analysis['matched_volume'] = np.minimum(
        avg_pnl_analysis['buy_qty_sum'], avg_pnl_analysis['sell_qty_sum']
    )
    avg_pnl_analysis['total_approx_pnl'] = avg_pnl_analysis['avg_pnl_per_unit'] * avg_pnl_analysis['matched_volume']

    # Select and rename columns for clarity
    result = avg_pnl_analysis[[
        'trader', 'symbol', 'avg_buy_price', 'buy_qty_sum',
        'avg_sell_price', 'sell_qty_sum', 'avg_pnl_per_unit', 'total_approx_pnl'
    ]].rename(columns={
        'buy_qty_sum': 'total_buy_qty',
        'sell_qty_sum': 'total_sell_qty'
    })

    return result.sort_values(by=['trader', 'total_approx_pnl'], ascending=[True, False])


# --- In Trade Analysis Functions section (add this new function) ---

def plot_trader_avg_pnl_per_unit(avg_pnl_df):
    """Creates a bar chart of average PNL per unit per asset for each trader."""
    if avg_pnl_df is None or avg_pnl_df.empty:
        st.warning("No Average PNL per Unit data to plot.")
        return None

    # Filter out cases where PNL per unit is 0 (e.g., only buys or only sells)
    plot_df = avg_pnl_df[avg_pnl_df['avg_pnl_per_unit'] != 0].copy()

    if plot_df.empty:
        st.info("No traders found with both buy and sell activity for any single asset.")
        return None

    # Sort traders for consistent plotting (e.g., by sum of their total approx PNL)
    trader_order = plot_df.groupby('trader')['total_approx_pnl'].sum().sort_values(ascending=False).index
    plot_df['trader'] = pd.Categorical(plot_df['trader'], categories=trader_order, ordered=True)
    plot_df = plot_df.sort_values('trader')

    fig = px.bar(plot_df,
                 x='trader',
                 y='avg_pnl_per_unit',
                 color='symbol',
                 title='Average Realized PNL per Unit Traded (Avg Sell Price - Avg Buy Price)',
                 labels={'avg_pnl_per_unit': 'Avg PNL per Unit', 'trader': 'Trader', 'symbol': 'Asset'},
                 hover_data=['symbol', 'avg_buy_price', 'avg_sell_price', 'total_buy_qty', 'total_sell_qty', 'total_approx_pnl'])

    fig.update_layout(xaxis_title="Trader",
                      yaxis_title="Average PNL per Unit",
                      xaxis={'categoryorder': 'array', 'categoryarray': trader_order}) # Keep sorted order
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
    return fig

def plot_trader_asset_pnl_lines(avg_pnl_df):
    """Creates a line plot of total approximate PNL per asset for each trader."""
    if avg_pnl_df is None or avg_pnl_df.empty:
        st.warning("No Average PNL data to plot as lines.")
        return None

    # Ensure PNL is numeric
    plot_df = avg_pnl_df.copy()
    plot_df['total_approx_pnl'] = pd.to_numeric(plot_df['total_approx_pnl'], errors='coerce')
    plot_df = plot_df.dropna(subset=['total_approx_pnl'])

    if plot_df.empty:
        st.warning("PNL data contains non-numeric values or is empty after cleaning.")
        return None

    # Sort traders for consistent plotting (e.g., by sum of their total approx PNL)
    trader_order = plot_df.groupby('trader')['total_approx_pnl'].sum().sort_values(ascending=False).index
    plot_df['trader'] = pd.Categorical(plot_df['trader'], categories=trader_order, ordered=True)
    plot_df = plot_df.sort_values(['trader', 'symbol']) # Sort for consistent lines

    fig = px.line(plot_df,
                  x='trader',
                  y='total_approx_pnl',
                  color='symbol',         # Each asset gets a line
                  markers=True,           # Add markers to see points clearly
                  title='Total Approximate Realized PNL per Trader per Asset (Line Plot)',
                  labels={'total_approx_pnl': 'Total Approx. PNL', 'trader': 'Trader', 'symbol': 'Asset'},
                  hover_data=['symbol', 'avg_pnl_per_unit', 'total_approx_pnl']) # Add relevant hover data

    fig.update_layout(xaxis_title="Trader",
                      yaxis_title="Total Approximate PNL",
                      xaxis={'categoryorder': 'array', 'categoryarray': trader_order}) # Keep sorted order
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
    return fig


def plot_spread_product(df, product):
    """Plots the best bid-ask spread over time for a given product."""
    spread_series = get_spread_series(df, product) # Uses your existing function
    if spread_series is None or spread_series.empty or spread_series.isnull().all():
        st.warning(f"No spread data found or calculable for {product}.")
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(spread_series.index, spread_series, label=f'{product} Bid-Ask Spread', color='red')
    ax.set_title(f'{product} Best Bid-Ask Spread Over Time')
    ax.set_xlabel('Time (Day-Timestamp)')
    ax.set_ylabel('Spread')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

def plot_zscore_spread_product(df, product, window=50):
    """Plots the rolling z-score of the best bid-ask spread for a given product."""
    spread_series = get_spread_series(df, product)
    if spread_series is None or spread_series.empty or spread_series.isnull().all():
         st.warning(f"Cannot calculate Spread Z-Score for {product} due to missing spread data.")
         return None

    # Drop NaNs before calculating rolling Z-score
    spread_series = spread_series.dropna()
    if spread_series.empty:
         st.warning(f"Spread series for {product} is empty after dropping NaNs.")
         return None

    zscore_series = rolling_zscore(spread_series, window)
    zscore_series = zscore_series.dropna() # Drop NaNs generated by rolling window

    if zscore_series.empty:
        st.warning(f"Cannot plot Spread Z-Score for {product} (possibly insufficient data for window size {window}).")
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(zscore_series.index, zscore_series, label=f'{product} Spread Rolling Z-Score', color='orange')
    ax.axhline(2, color='red', linestyle='--', label='Upper Threshold (2)')
    ax.axhline(-2, color='green', linestyle='--', label='Lower Threshold (-2)')
    ax.set_title(f'{product} Spread Rolling Z-Score (Window = {window})')
    ax.set_xlabel('Time (Day-Timestamp)')
    ax.set_ylabel('Z-Score')
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    plt.tight_layout()
    return fig


def plot_single_trader_pnl_lines(avg_pnl_df, selected_trader):
    """Creates a line plot of total approximate PNL per asset for a SINGLE selected trader."""
    if avg_pnl_df is None or avg_pnl_df.empty:
        st.warning("No Average PNL data available.")
        return None
    if not selected_trader:
        st.info("Please select a trader.")
        return None

    # Filter data for the selected trader
    trader_df = avg_pnl_df[avg_pnl_df['trader'] == selected_trader].copy()

    if trader_df.empty:
        st.info(f"No PNL data found for trader: {selected_trader}")
        return None

    # Ensure PNL is numeric
    trader_df['total_approx_pnl'] = pd.to_numeric(trader_df['total_approx_pnl'], errors='coerce')
    trader_df = trader_df.dropna(subset=['total_approx_pnl'])

    if trader_df.empty:
        st.info(f"No valid PNL data to plot for trader: {selected_trader}")
        return None

    # Sort by asset symbol for consistent plotting
    trader_df = trader_df.sort_values('symbol')

    fig = px.line(trader_df,
                  x='symbol',             # Assets on the x-axis
                  y='total_approx_pnl',   # PNL on the y-axis
                  markers=True,           # Add markers to see points clearly
                  title=f'Total Approximate Realized PNL per Asset for {selected_trader}',
                  labels={'total_approx_pnl': 'Total Approx. PNL', 'symbol': 'Asset Symbol'},
                  hover_data=['symbol', 'avg_pnl_per_unit', 'total_approx_pnl']) # Add relevant hover data

    fig.update_layout(xaxis_title="Asset Symbol",
                      yaxis_title="Total Approximate PNL")
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
    # Ensure x-axis labels don't overlap if many assets
    fig.update_xaxes(tickangle=45)

    return fig

# ---------------------------------------
# Main function to load data and display graphs and statistical tests in Streamlit
# ---------------------------------------
def main():
    st.title("IMC Prosperity Visualiser (Round 5)")
    st.markdown("Dashboard for analysing order book data and trade activities.")

    # --- File Paths ---
    # Make sure these paths are correct relative to where you run the script
    order_book_file = '../data/submission_data/zainold.csv' # Keep existing or update if needed
    trade_file = '../data/round5/trades_combined_r5.csv' # Use the specified path

    # --- Define Traders and Assets ---
    TRADERS = ['Caesar', 'Charlie', 'Paris', 'Camilla', 'Pablo', 'Penelope', 'Gary', 'Peter', 'Gina', 'Olivia']
    ASSETS = ['CROISSANTS', 'RAINFOREST_RESIN', 'JAMS', 'KELP', 'SQUID_INK', 'PICNIC_BASKET1',
              'PICNIC_BASKET2', 'MAGNIFICENT_MACARONS', 'DJEMBES', 'VOLCANIC_ROCK_VOUCHER_10000',
              'VOLCANIC_ROCK_VOUCHER_10250', 'VOLCANIC_ROCK_VOUCHER_10500', 'VOLCANIC_ROCK_VOUCHER_9500',
              'VOLCANIC_ROCK_VOUCHER_9750', 'VOLCANIC_ROCK']


    # --- Load Data ---
    st.sidebar.header("Data Loading")
    # Load Order Book Data
    orders_df = None # Initialize
    try:
        orders_df = pd.read_csv(order_book_file, sep=';')
        orders_df['day'] = pd.to_numeric(orders_df['day'], errors='coerce').fillna(0).astype(int)
        orders_df['timestamp'] = pd.to_numeric(orders_df['timestamp'], errors='coerce').fillna(0).astype(int)
        # Convert price/volume columns safely
        price_cols = [f'{side}_price_{i}' for side in ['bid', 'ask'] for i in range(1, 4)] + ['mid_price']
        vol_cols = [f'{side}_volume_{i}' for side in ['bid', 'ask'] for i in range(1, 4)]
        for col in price_cols + vol_cols + ['profit_and_loss']:
            if col in orders_df.columns:
                 orders_df[col] = pd.to_numeric(orders_df[col], errors='coerce')

        st.sidebar.success(f"Order book data loaded ({len(orders_df)} rows)")

        # Pre-calculate best bid/ask/spread/vwap only if columns exist
        if all(col in orders_df.columns for col in [f'bid_price_{i}' for i in range(1, 4)] + [f'ask_price_{i}' for i in range(1, 4)]):
             orders_df['best_bid'] = orders_df.apply(get_best_bid, axis=1)
             orders_df['best_ask'] = orders_df.apply(get_best_ask, axis=1)
             orders_df['spread'] = orders_df.apply(calculate_spread, axis=1)
             orders_df['vwap'] = orders_df.apply(get_vwap_for_row, axis=1)
        else:
             st.sidebar.warning("Order book price/volume columns missing, skipping best_bid/ask/spread/vwap calculation.")

    except FileNotFoundError:
        st.sidebar.error(f"Order book file not found: {order_book_file}")
        # Don't stop execution, allow trade analysis if trades load
    except Exception as e:
        st.sidebar.error(f"Error loading order book data: {e}")


    # Load Trade Data
    trades_df = load_trade_data(trade_file)
    if trades_df is None:
        st.sidebar.warning("Trade data could not be loaded. Trade analysis section will be limited.")
    else:
        st.sidebar.success(f"Trade data loaded ({len(trades_df)} rows)")


    # --- Sidebar Navigation ---
    st.sidebar.header("Analysis Sections")
    # Define available sections based on loaded data
    available_sections = []
    if trades_df is not None:
        available_sections.append("Trade Analysis")
    if orders_df is not None:
        available_sections.extend(["Order Book Analysis", "Statistical Tests", "Asset Price Plots"])

    if not available_sections:
        st.error("No data loaded successfully. Cannot display analysis.")
        st.stop()

    # Set default section
    default_section_index = 0
    if "Trade Analysis" in available_sections:
         default_section_index = available_sections.index("Trade Analysis")
    elif "Order Book Analysis" in available_sections:
         default_section_index = available_sections.index("Order Book Analysis")


    analysis_section = st.sidebar.radio(
        "Choose Analysis:",
        available_sections,
        index=default_section_index
    )

    # =======================================
    # SECTION: Trade Analysis
    # =======================================
    if analysis_section == "Trade Analysis":
        st.header("Trade Activity Analysis")
        if trades_df is None:
            st.warning("Trade data not available.")
        else:
            st.markdown("Analysis of buy/sell volumes, realized PNL, and specific trade dynamics.")
            st.info("PNL is calculated as Total Value Sold - Total Value Bought per trader/asset. This is an approximation of realized PNL.")

            # --- Trader Volume Analysis ---
            with st.expander("Trader Volume Summary", expanded=False):
                volume_analysis_df = analyze_trader_volume(trades_df)
                if not volume_analysis_df.empty:
                    st.dataframe(volume_analysis_df.style.format({
                        "buy_volume": "{:,.0f}",
                        "sell_volume": "{:,.0f}",
                        "total_volume": "{:,.0f}",
                        "net_volume": "{:,.0f}"
                    }))
                else:
                    st.info("No volume data calculated.")

            # --- Trader PNL Analysis ---
            st.subheader("Trader PNL Analysis (Based on Trades)")

            # --- Method 1: Total Value Sold - Total Value Bought ---
            st.markdown("**Method 1: Total Value PNL**")
            st.markdown("_Calculated as Total Value Sold - Total Value Bought per trader/asset._")
            pnl_analysis_df = calculate_trader_asset_pnl(trades_df) # Uses the original function
            fig_pnl_trader_asset = plot_trader_asset_pnl(pnl_analysis_df) # Uses the original plot function
            if fig_pnl_trader_asset:
                st.plotly_chart(fig_pnl_trader_asset, use_container_width=True)
            else:
                 st.info("Could not generate Total Value PNL plot.")
            with st.expander("Show Total Value PNL Data Table", expanded=False):
                 if not pnl_analysis_df.empty:
                     st.dataframe(pnl_analysis_df.style.format({
                         "total_value_bought": "{:,.2f}",
                         "total_value_sold": "{:,.2f}",
                         "pnl": "{:,.2f}"
                     }))
                 else:
                     st.info("No Total Value PNL data calculated.")

            # --- Method 2: Average PNL per Unit ---
            st.markdown("**Method 2: Average PNL per Unit Traded**")
            st.markdown("_Calculated as (Average Sell Price - Average Buy Price) per trader/asset. Only shown for assets where both buys and sells occurred._")
            avg_pnl_df = calculate_trader_avg_pnl_per_unit(trades_df) # Call the NEW calculation
            fig_avg_pnl = plot_trader_avg_pnl_per_unit(avg_pnl_df) # Call the NEW plot function
            if fig_avg_pnl:
                st.plotly_chart(fig_avg_pnl, use_container_width=True)
            else:
                 st.info("Could not generate Average PNL per Unit plot (or no round trips found).")

            with st.expander("Show Average PNL per Unit Data Table", expanded=False):
                 if not avg_pnl_df.empty:
                     st.dataframe(avg_pnl_df.style.format({
                         "avg_buy_price": "{:,.2f}",
                         "total_buy_qty": "{:,.0f}",
                         "avg_sell_price": "{:,.2f}",
                         "total_sell_qty": "{:,.0f}",
                         "avg_pnl_per_unit": "{:,.2f}",
                         "total_approx_pnl": "{:,.2f}"
                     }))
                 else:
                     st.info("No Average PNL per Unit data calculated.")

            st.markdown("**PNL Distribution per Trader (Line Plot)**")
            st.markdown("_Select a trader to view their Total Approximate PNL per asset._")

            # Check if avg_pnl_df exists and is not empty
            if 'avg_pnl_df' in locals() and not avg_pnl_df.empty:
                # Get unique traders who have PNL data
                available_traders = sorted(avg_pnl_df['trader'].unique())

                if available_traders:
                    # Add a selectbox to choose the trader
                    selected_trader_pnl = st.selectbox(
                        "Select Trader for PNL Line Plot:",
                        options=available_traders
                    )

                    if selected_trader_pnl:
                        # Call the NEW plotting function for the selected trader
                        fig_single_trader_pnl = plot_single_trader_pnl_lines(avg_pnl_df, selected_trader_pnl)

                        if fig_single_trader_pnl:
                            st.plotly_chart(fig_single_trader_pnl, use_container_width=True)
                        # else: # plot_single_trader_pnl_lines already shows messages if no plot
                            # st.info(f"Could not generate PNL line plot for {selected_trader_pnl}.")
                else:
                    st.info("No traders found with calculated PNL data.")
            else:
                st.warning("Average PNL data not available for line plot.")


            # --- Most Traded Asset per Trader ---
            st.subheader("Most Traded Asset per Trader")
            # Recalculate volume if needed (might be empty if expander wasn't opened)
            if 'volume_analysis_df' not in locals() or volume_analysis_df.empty:
                 volume_analysis_df = analyze_trader_volume(trades_df)

            most_traded_df = get_most_traded_assets(volume_analysis_df)
            fig_most_traded = plot_most_traded_per_trader(most_traded_df)
            if fig_most_traded:
                st.plotly_chart(fig_most_traded, use_container_width=True)
            else:
                 st.info("No data for 'Most Traded Asset' plot.")

            # --- Specific Trade Dynamics ---
            st.subheader("Individual Trade Dynamics")
            st.markdown("Select a trader and asset to see their buy/sell activity over time.")
            col1, col2 = st.columns(2)
            # Get available traders and assets from PNL or Volume data if available
            if not pnl_analysis_df.empty:
                available_traders = sorted(pnl_analysis_df['trader'].unique())
            elif not volume_analysis_df.empty:
                 available_traders = sorted(volume_analysis_df['trader'].unique())
            else:
                 available_traders = sorted(TRADERS) # Fallback

            with col1:
                selected_trader = st.selectbox("Select Trader:", available_traders)
            with col2:
                # Filter assets traded by the selected trader from PNL or Volume data
                if selected_trader:
                    if not pnl_analysis_df.empty:
                        trader_assets = sorted(pnl_analysis_df[pnl_analysis_df['trader'] == selected_trader]['symbol'].unique())
                    elif not volume_analysis_df.empty:
                        trader_assets = sorted(volume_analysis_df[volume_analysis_df['trader'] == selected_trader]['symbol'].unique())
                    else:
                        trader_assets = sorted(ASSETS) # Fallback
                else:
                    trader_assets = sorted(ASSETS) # Fallback

                if not trader_assets: # Handle case where selected trader might have no assets listed
                     trader_assets = sorted(ASSETS)

                selected_asset_trade = st.selectbox("Select Asset:", trader_assets)

            if selected_trader and selected_asset_trade:
                fig_trade_dynamics = plot_trade_dynamics(trades_df, selected_trader, selected_asset_trade)
                if fig_trade_dynamics:
                    st.pyplot(fig_trade_dynamics)


    # =======================================
    # SECTION: Order Book Analysis
    # =======================================
    elif analysis_section == "Order Book Analysis":
        st.header("Order Book Analysis")
        if orders_df is None:
             st.warning("Order book data not available.")
        else:
            st.markdown("Analysis based on order book snapshots (mid-price, VWAP, spreads, theoretical baskets).")

            # --- Prepare DataFrames for Analysis ---
            df_prices = None # Initialize
            try:
                 # Use VWAP where available, fallback to mid_price if VWAP calc failed or is NaN
                 mid_CROISSANTS   = get_vwap_series(orders_df, 'CROISSANTS').fillna(get_mid_price_series(orders_df, 'CROISSANTS'))
                 mid_JAMS         = get_vwap_series(orders_df, 'JAMS').fillna(get_mid_price_series(orders_df, 'JAMS'))
                 mid_DJEMBES      = get_vwap_series(orders_df, 'DJEMBES').fillna(get_mid_price_series(orders_df, 'DJEMBES'))
                 mid_PICNIC_BASKET1 = get_mid_price_series(orders_df, 'PICNIC_BASKET1') # Baskets use mid_price
                 mid_PICNIC_BASKET2 = get_mid_price_series(orders_df, 'PICNIC_BASKET2') # Baskets use mid_price

                 # Align indexes before creating DataFrame
                 idx = mid_CROISSANTS.index.union(mid_JAMS.index).union(mid_DJEMBES.index).union(mid_PICNIC_BASKET1.index).union(mid_PICNIC_BASKET2.index)

                 df_prices = pd.DataFrame({
                     'Croissants': mid_CROISSANTS.reindex(idx),
                     'Jams': mid_JAMS.reindex(idx),
                     'Djembes': mid_DJEMBES.reindex(idx),
                     'Basket1': mid_PICNIC_BASKET1.reindex(idx),
                     'Basket2': mid_PICNIC_BASKET2.reindex(idx)
                 })
                 # Forward fill to handle missing timestamps before dropping NaNs used in calculations
                 df_prices = df_prices.ffill()#.dropna() # Drop rows with any NaNs AFTER ffill - careful not to drop too much

                 if df_prices.empty:
                      st.warning("Could not create DataFrame for basket analysis (likely missing data for components).")
                      df_prices = None # Set to None to skip dependent plots

            except Exception as e:
                 st.error(f"Error preparing data for basket analysis: {e}")
                 df_prices = None # Set to None


            st.subheader("VWAP and Bid/Ask Spread Analysis (Selected Products)")
            with st.expander("RAINFOREST_RESIN VWAP/Spread"):
                fig_resin = plot_resin_vwap_spread(orders_df)
                if fig_resin: st.pyplot(fig_resin)
            with st.expander("SQUID_INK VWAP/Spread"):
                fig_squid = plot_squid_vwap_spread(orders_df)
                if fig_squid: st.pyplot(fig_squid)
            with st.expander("KELP VWAP/Spread"):
                fig_kelp = plot_kelp_vwap_spread(orders_df)
                if fig_kelp: st.pyplot(fig_kelp)


            st.subheader("Rolling Z-Score Analysis (VWAP, Window=50)")
            with st.expander("RAINFOREST_RESIN Z-Score"):
                fig_z_resin = plot_zscore_resin(orders_df, window=50)
                if fig_z_resin: st.pyplot(fig_z_resin)
            with st.expander("SQUID_INK Z-Score"):
                fig_z_squid = plot_zscore_squid(orders_df, window=50)
                if fig_z_squid: st.pyplot(fig_z_squid)
            with st.expander("KELP Z-Score"):
                fig_z_kelp = plot_zscore_kelp(orders_df, window=50)
                if fig_z_kelp: st.pyplot(fig_z_kelp)

            if df_prices is not None:
                st.subheader("Theoretical Picnic Basket Price Analysis")
                with st.expander("Basket 1: Actual vs Theoretical & Spread"):
                    fig_theory_b1 = plot_theoretical_actual_basket1(df_prices)
                    if fig_theory_b1: st.pyplot(fig_theory_b1)
                    fig_spread_b1 = plot_theoretical_spread_basket1(df_prices)
                    if fig_spread_b1: st.pyplot(fig_spread_b1)

                with st.expander("Basket 2: Actual vs Theoretical & Spread"):
                    fig_theory_b2 = plot_theoretical_actual_basket2(df_prices) # Make sure this function is defined/adjusted
                    if fig_theory_b2: st.pyplot(fig_theory_b2)
                    fig_spread_b2 = plot_theoretical_spread_basket2(df_prices) # Make sure this function is defined/adjusted
                    if fig_spread_b2: st.pyplot(fig_spread_b2)

                st.subheader("Basket Spread Rolling Z-Score Analysis (Window=500)")
                with st.expander("Basket 1 Spread Z-Score"):
                    fig_z_b1 = plot_zscore_basket1(df_prices, window=500)
                    if fig_z_b1: st.pyplot(fig_z_b1)
                with st.expander("Basket 2 Spread Z-Score"):
                    fig_z_b2 = plot_zscore_basket2(df_prices, window=500) # Make sure this function is defined/adjusted
                    if fig_z_b2: st.pyplot(fig_z_b2)
                with st.expander("Spread between Basket Z-Scores"):
                    fig_z_spread = plot_zscore_spread(df_prices, window=500) # Make sure this function is defined/adjusted
                    if fig_z_spread: st.pyplot(fig_z_spread)
            else:
                st.info("Skipping Basket analysis due to data preparation issues.")
            # --- Add Macarons Spread Analysis ---
            st.subheader("MAGNIFICENT_MACARONS Specific Analysis")
            macaron_product = "MAGNIFICENT_MACARONS"

            with st.expander(f"{macaron_product} - Mid Price", expanded=False):
                fig_m_mid = plot_mid_price(orders_df, macaron_product)
                if fig_m_mid: st.pyplot(fig_m_mid)

            with st.expander(f"{macaron_product} - Bid-Ask Spread", expanded=True): # Expand this one by default maybe
                fig_m_spread = plot_spread_product(orders_df, macaron_product)
                if fig_m_spread: st.pyplot(fig_m_spread)

            with st.expander(f"{macaron_product} - Spread Z-Score (Window=50)", expanded=True): # Expand this too
                fig_m_z_spread = plot_zscore_spread_product(orders_df, macaron_product, window=50)
                if fig_m_z_spread: st.pyplot(fig_m_z_spread)


    # =======================================
    # SECTION: Statistical Tests
    # =======================================
    elif analysis_section == "Statistical Tests":
        st.header("Statistical Tests")
        if orders_df is None:
             st.warning("Order book data not available for statistical tests.")
        else:
            st.markdown("Stationarity (ADF) and Cointegration (Johansen) tests.")

            # --- Prepare DataFrames for Tests ---
            df_prices_test = None # Initialize
            try:
                 mid_CROISSANTS   = get_vwap_series(orders_df, 'CROISSANTS').fillna(get_mid_price_series(orders_df, 'CROISSANTS'))
                 mid_JAMS         = get_vwap_series(orders_df, 'JAMS').fillna(get_mid_price_series(orders_df, 'JAMS'))
                 mid_DJEMBES      = get_vwap_series(orders_df, 'DJEMBES').fillna(get_mid_price_series(orders_df, 'DJEMBES'))
                 mid_PICNIC_BASKET1 = get_mid_price_series(orders_df, 'PICNIC_BASKET1')
                 mid_PICNIC_BASKET2 = get_mid_price_series(orders_df, 'PICNIC_BASKET2')

                 # Align indexes
                 idx = mid_CROISSANTS.index.union(mid_JAMS.index).union(mid_DJEMBES.index).union(mid_PICNIC_BASKET1.index).union(mid_PICNIC_BASKET2.index)

                 df_prices_test = pd.DataFrame({
                     'Croissants': mid_CROISSANTS.reindex(idx), 'Jams': mid_JAMS.reindex(idx), 'Djembes': mid_DJEMBES.reindex(idx),
                     'Basket1': mid_PICNIC_BASKET1.reindex(idx), 'Basket2': mid_PICNIC_BASKET2.reindex(idx)
                 }).ffill() # Fill forward first

                 # Calculate spreads only if components and baskets exist
                 if all(c in df_prices_test.columns for c in ['Croissants', 'Jams', 'Djembes', 'Basket1']):
                     df_prices_test['Theo_Basket1'] = 6 * df_prices_test['Croissants'] + 3 * df_prices_test['Jams'] + 1 * df_prices_test['Djembes']
                     df_prices_test['Spread_Basket1'] = df_prices_test['Basket1'] - df_prices_test['Theo_Basket1']
                 else: df_prices_test['Spread_Basket1'] = np.nan

                 if all(c in df_prices_test.columns for c in ['Croissants', 'Jams', 'Basket2']):
                     df_prices_test['Theo_Basket2'] = 4 * df_prices_test['Croissants'] + 2 * df_prices_test['Jams']
                     df_prices_test['Spread_Basket2'] = df_prices_test['Basket2'] - df_prices_test['Theo_Basket2']
                 else: df_prices_test['Spread_Basket2'] = np.nan

                 # Calculate Z-scores only if spreads are available and non-empty
                 if 'Spread_Basket1' in df_prices_test and not df_prices_test['Spread_Basket1'].isnull().all():
                      df_prices_test['Zscore_Basket1'] = rolling_zscore(df_prices_test['Spread_Basket1'].dropna(), window = 500) # Dropna before rolling
                 else: df_prices_test['Zscore_Basket1'] = np.nan

                 if 'Spread_Basket2' in df_prices_test and not df_prices_test['Spread_Basket2'].isnull().all():
                      df_prices_test['Zscore_Basket2'] = rolling_zscore(df_prices_test['Spread_Basket2'].dropna(), window = 500) # Dropna before rolling
                 else: df_prices_test['Zscore_Basket2'] = np.nan

            except Exception as e:
                 st.error(f"Error preparing data for statistical tests: {e}")
                 df_prices_test = pd.DataFrame() # Empty df if error


            if df_prices_test is not None and not df_prices_test.empty:
                st.subheader("ADF Test on Basket Spreads")
                if 'Spread_Basket1' in df_prices_test and not df_prices_test['Spread_Basket1'].isnull().all(): display_adf_results(df_prices_test['Spread_Basket1'], "PICNIC_BASKET1 Spread")
                if 'Spread_Basket2' in df_prices_test and not df_prices_test['Spread_Basket2'].isnull().all(): display_adf_results(df_prices_test['Spread_Basket2'], "PICNIC_BASKET2 Spread")

                st.subheader("Johansen Cointegration Tests")
                with st.expander("Test on Basket Prices (Basket1, Basket2)"):
                     prices_for_johansen = df_prices_test[['Basket1', 'Basket2']].dropna()
                     if not prices_for_johansen.empty and prices_for_johansen.shape[0] >= prices_for_johansen.shape[1]:
                          display_johansen_results(prices_for_johansen)
                     else: st.info("Insufficient data for Johansen test on Basket Prices.")

                with st.expander("Test on Basket Spread Z-Scores (Zscore_Basket1, Zscore_Basket2)"):
                     z_scores = df_prices_test[['Zscore_Basket1', 'Zscore_Basket2']].dropna()
                     if not z_scores.empty and z_scores.shape[0] >= z_scores.shape[1]:
                         display_johansen_results(z_scores)
                     else: st.info("Insufficient data for Johansen test on Z-Scores.")

                with st.expander("Test on Basket Spreads (Spread_Basket1, Spread_Basket2)"):
                     spread_series = df_prices_test[['Spread_Basket1', 'Spread_Basket2']].dropna()
                     if not spread_series.empty and spread_series.shape[0] >= spread_series.shape[1]:
                         display_johansen_results(spread_series)
                     else: st.info("Insufficient data for Johansen test on Spreads.")

                st.subheader("Correlation Matrix (Log Returns)")
                cols_for_corr = ['Basket1', 'Basket2', 'Croissants', 'Jams', 'Djembes']
                valid_cols_for_corr = [col for col in cols_for_corr if col in df_prices_test.columns and not df_prices_test[col].isnull().all()]
                if valid_cols_for_corr:
                     display_correlation(df_prices_test[valid_cols_for_corr])
                else:
                     st.warning("No valid columns found for correlation analysis.")
            else:
                st.warning("Statistical tests skipped due to data preparation issues.")

    # =======================================
    # SECTION: Asset Price Plots
    # =======================================
    elif analysis_section == "Asset Price Plots":
        st.header("Asset Price Plots")
        if orders_df is None:
            st.warning("Order book data not available for asset price plots.")
        else:
            all_order_book_assets = sorted(orders_df['product'].unique())

            # --- Mid Price Plots ---
            st.subheader("Mid Price Plots")
            mid_price_asset = st.selectbox("Select Asset for Mid Price Plot:", all_order_book_assets)
            if mid_price_asset:
                fig_mid = plot_mid_price(orders_df, mid_price_asset)
                if fig_mid: st.pyplot(fig_mid)

            # --- PnL Plots (from Order Book) ---
            st.subheader("Profit and Loss (PnL) Plots (from Order Book)")
            st.markdown("_Note: This PnL data comes from the order book file ('profit_and_loss' column), if available. It may differ from PnL calculated from trades._")
            pnl_asset = st.selectbox("Select Asset for Order Book PnL Plot:", all_order_book_assets)
            if pnl_asset:
                fig_pnl = plot_pnl(orders_df, pnl_asset)
                if fig_pnl: st.pyplot(fig_pnl)


    st.markdown("---")
    st.markdown("### End of Visualisation")



if __name__ == '__main__':
    main()