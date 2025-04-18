import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
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
# Main function to load data and display graphs and statistical tests in Streamlit
# ---------------------------------------
def main():
    st.title("IMC Prosperity Basket Visualiser")
    st.markdown("This dashboard displays various plots and statistical tests derived from the order book data, "
                "including theoretical price analysis, rolling z-scores, stationarity tests, cointegration tests, "
                "and profit and loss (PnL) for individual assets.")
    
    # Load CSV file (update path as needed)
    orders_df = pd.read_csv('../data/submission_data/zainalgo.csv', sep=';')
    orders_df['day'] = orders_df['day'].astype(int)
    orders_df['timestamp'] = orders_df['timestamp'].astype(int)
    orders_df['mid_price'] = orders_df['mid_price'].astype(float)
    
    # Pre-calculate best bid/ask and spreads
    orders_df['best_bid'] = orders_df.apply(get_best_bid, axis=1)
    orders_df['best_ask'] = orders_df.apply(get_best_ask, axis=1)
    orders_df['spread'] = orders_df.apply(calculate_spread, axis=1)
    
    # Extract mid-price series for components and baskets
    mid_SQUINK = get_vwap_series(orders_df, 'SQUID_INK')
    mid_RESIN = get_vwap_series(orders_df, 'RAINFOREST_RESIN')
    mid_KELP = get_vwap_series(orders_df, 'KELP')
    mid_CROISSANTS   = get_vwap_series(orders_df, 'CROISSANTS')
    mid_JAMS         = get_vwap_series(orders_df, 'JAMS')
    mid_DJEMBES      = get_vwap_series(orders_df, 'DJEMBES')
    mid_PICNIC_BASKET1 = get_mid_price_series(orders_df, 'PICNIC_BASKET1')
    mid_PICNIC_BASKET2 = get_mid_price_series(orders_df, 'PICNIC_BASKET2')
    
    # Combine into a DataFrame for theoretical analysis.

    df_round1 = pd.DataFrame({
        'Squink': mid_SQUINK,
        'Resin': mid_RESIN,
        'Kelp': mid_KELP,
    }).dropna()




    df_prices = pd.DataFrame({
        'Croissants': mid_CROISSANTS,
        'Jams': mid_JAMS,
        'Djembes': mid_DJEMBES,
        'Basket1': mid_PICNIC_BASKET1,
        'Basket2': mid_PICNIC_BASKET2
    }).dropna()

    st.header("VWAP and Bid/Ask Spread Analysis")
    fig_resin = plot_resin_vwap_spread(orders_df)
    st.pyplot(fig_resin)

    fig_squid = plot_squid_vwap_spread(orders_df)
    st.pyplot(fig_squid)

    fig_kelp = plot_kelp_vwap_spread(orders_df)
    st.pyplot(fig_kelp)

    st.header("Rolling Z-Score Analysis for Individual Products")
    fig_z_resin = plot_zscore_resin(orders_df, window=50)
    st.pyplot(fig_z_resin)

    fig_z_squid = plot_zscore_squid(orders_df, window=50)
    st.pyplot(fig_z_squid)

    fig_z_kelp = plot_zscore_kelp(orders_df, window=50)
    st.pyplot(fig_z_kelp)

    
    st.header("Theoretical Price Analysis")
    st.markdown("Comparison of actual vs. theoretical prices and the corresponding price spread for each basket.")
    fig_theory_b1 = plot_theoretical_actual_basket1(df_prices)
    st.pyplot(fig_theory_b1)
    fig_spread_b1 = plot_theoretical_spread_basket1(df_prices)
    st.pyplot(fig_spread_b1)
    fig_theory_b2 = plot_theoretical_actual_basket2(df_prices)
    st.pyplot(fig_theory_b2)
    fig_spread_b2 = plot_theoretical_spread_basket2(df_prices)
    st.pyplot(fig_spread_b2)
    
    st.header("Rolling Z-Score Analysis")
    st.markdown("Rolling z-scores for the price spread for each basket and the spread between the two z-score series.")
    fig_z_b1 = plot_zscore_basket1(df_prices, window=500)
    st.pyplot(fig_z_b1)
    fig_z_b2 = plot_zscore_basket2(df_prices, window=500)
    st.pyplot(fig_z_b2)
    fig_z_spread = plot_zscore_spread(df_prices, window=500)
    st.pyplot(fig_z_spread)

    
    st.header("Statistical Tests")
    st.markdown("Results and interpretations of stationarity and cointegration tests.")
    # Run ADF tests on Basket1 Spread and Basket2 Spread
    df_temp = df_prices.copy()
    df_temp['Theo_Basket1'] = 6 * df_temp['Croissants'] + 3 * df_temp['Jams'] + 1 * df_temp['Djembes']
    df_temp['Spread_Basket1'] = df_temp['Basket1'] - df_temp['Theo_Basket1']
    df_temp['Theo_Basket2'] = 4 * df_temp['Croissants'] + 2 * df_temp['Jams']
    df_temp['Spread_Basket2'] = df_temp['Basket2'] - df_temp['Theo_Basket2']
    df_temp['Zscore_Basket1'] = rolling_zscore(df_temp['Spread_Basket1'], window = 500)
    df_temp['Zscore_Basket2'] = rolling_zscore(df_temp['Spread_Basket2'], window = 500)
    
    st.subheader("ADF Test on Basket Spreads")
    adf_result_b1 = run_adf(df_temp['Spread_Basket1'], "PICNIC_BASKET1 Spread")
    adf_result_b2 = run_adf(df_temp['Spread_Basket2'], "PICNIC_BASKET2 Spread")
    st.text(adf_result_b1)
    st.text(adf_result_b2)
    
    # Run Johansen cointegration test on the baskets.
    st.subheader("Johansen Cointegration Test on Baskets (price of PICNIC_BASKET1 and PICNIC_BASKET2)")
    prices_for_johansen = df_prices[['Basket1', 'Basket2']]
    johansen_out = display_johansen_results(prices_for_johansen)

    st.subheader("Johansens Test on the Z-Scores (Z-Score of Basket1 and Basket2)")
    z_scores = df_temp[['Zscore_Basket1', 'Zscore_Basket2']].dropna()
    johansen_out_z = display_johansen_results(z_scores)

    st.subheader("Johansens Test on the Spread (Spread of theoretical and actual of basket1 on basket2)")
    spread_series = df_temp[['Spread_Basket1', 'Spread_Basket2']]
    johansen_out_spread = display_johansen_results(spread_series)
    
    # Display correlation matrix of log returns.
    st.subheader("Correlation Matrix (Log Returns)")
    display_correlation(df_prices[['Basket1', 'Basket2', 'Croissants', 'Jams', 'Djembes']])

    st.header("Volcanic Rock & Vouchers")
    volcanic_products = [
        "VOLCANIC_ROCK",
        "VOLCANIC_ROCK_VOUCHER_9500",
        "VOLCANIC_ROCK_VOUCHER_9750",
        "VOLCANIC_ROCK_VOUCHER_10000",
        "VOLCANIC_ROCK_VOUCHER_10250",
        "VOLCANIC_ROCK_VOUCHER_10500",
    ]
    for prod in volcanic_products:
        st.subheader(prod)
        st.pyplot(plot_mid_price(orders_df, prod))

    st.header("Macarons")
    macarons = "MAGNIFICENT_MACARONS"
    st.subheader(macarons)
    st.pyplot(plot_mid_price(orders_df, macarons))
   
    
    st.header("Profit and Loss (PnL)")
    st.markdown("PnL (profit_and_loss) over time for individual assets.")
    asset = st.selectbox("Select asset for PnL plot:", ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2", "SQUID_INK", "RAINFOREST_RESIN", 
                                                        "KELP", "VOLCANIC_ROCK", "MAGNIFICENT_MACARONS", "VOLCANIC_ROCK_VOUCHER_9500",
                                                        "VOLCANIC_ROCK_VOUCHER_9750", "VOLCANIC_ROCK_VOUCHER_10000",
                                                        "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500"])

    pnl_fig = plot_pnl(orders_df, asset)
    st.pyplot(pnl_fig)


    
    st.markdown("### End of Visualisation")
    
if __name__ == '__main__':
    main()
