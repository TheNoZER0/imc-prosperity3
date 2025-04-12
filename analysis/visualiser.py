import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
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
    Assumes the CSV column is named 'profit_and_loss'
    """
    sub = df[df['product'] == product][['day', 'timestamp', 'profit_and_loss']].copy()
    sub['time'] = sub['day'].astype(str) + '-' + sub['timestamp'].astype(str)
    sub = sub.sort_values(['day','timestamp']).set_index('time')
    return sub['profit_and_loss']

# ---------------------------------------
# Plotting Functions (each returns one figure)
# ---------------------------------------

def plot_theoretical_actual_basket1(df_prices):
    """
    Produce one figure showing actual vs. theoretical prices for PICNIC_BASKET1.
    The theoretical price is computed as:
       Theo_Basket1 = 6 * Croissants + 3 * Jams + 1 * Djembes
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
    The theoretical price is computed as:
       Theo_Basket2 = 4 * Croissants + 2 * Jams
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
    If Spread_Basket1 is missing, it is computed.
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
    If Spread_Basket2 is missing, it is computed.
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
    # Ensure spreads are computed.
    if 'Spread_Basket1' not in df_plot.columns:
        df_plot['Theo_Basket1'] = 6 * df_plot['Croissants'] + 3 * df_plot['Jams'] + 1 * df_plot['Djembes']
        df_plot['Spread_Basket1'] = df_plot['Basket1'] - df_plot['Theo_Basket1']
    if 'Spread_Basket2' not in df_plot.columns:
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

# ---------------------------------------
# Main function to load data and display graphs in Streamlit
# ---------------------------------------
def main():
    st.title("IMC Prosperity Basket Visualiser")
    st.markdown("This dashboard displays various plots derived from the order book data, including theoretical price analysis, rolling z-scores and their spreads, and the profit and loss (PnL) for individual assets.")
    
    # Load the CSV file (update the path as needed)
    orders_df = pd.read_csv('../data/submission_data/downloaded_mishi_algo.csv', sep=';')
    # Convert data types appropriately.
    orders_df['day'] = orders_df['day'].astype(int)
    orders_df['timestamp'] = orders_df['timestamp'].astype(int)
    orders_df['mid_price'] = orders_df['mid_price'].astype(float)
    
    # Apply best bid, best ask, and spread functions (if needed elsewhere)
    orders_df['best_bid'] = orders_df.apply(get_best_bid, axis=1)
    orders_df['best_ask'] = orders_df.apply(get_best_ask, axis=1)
    orders_df['spread'] = orders_df.apply(calculate_spread, axis=1)
    
    # Extract mid-price series for components and baskets.
    mid_CROISSANTS   = get_mid_price_series(orders_df, 'CROISSANTS')
    mid_JAMS         = get_mid_price_series(orders_df, 'JAMS')
    mid_DJEMBES      = get_mid_price_series(orders_df, 'DJEMBES')
    mid_PICNIC_BASKET1 = get_mid_price_series(orders_df, 'PICNIC_BASKET1')
    mid_PICNIC_BASKET2 = get_mid_price_series(orders_df, 'PICNIC_BASKET2')
    
    # Combine into a DataFrame for theoretical analysis.
    df_prices = pd.DataFrame({
        'Croissants': mid_CROISSANTS,
        'Jams': mid_JAMS,
        'Djembes': mid_DJEMBES,
        'Basket1': mid_PICNIC_BASKET1,
        'Basket2': mid_PICNIC_BASKET2
    }).dropna()
    
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
    st.markdown("Rolling z-scores for the price spread for each basket and the difference between the two z-score series.")
    fig_z_b1 = plot_zscore_basket1(df_prices, window=500)
    st.pyplot(fig_z_b1)
    fig_z_b2 = plot_zscore_basket2(df_prices, window=500)
    st.pyplot(fig_z_b2)
    fig_z_spread = plot_zscore_spread(df_prices, window=500)
    st.pyplot(fig_z_spread)
    
    st.header("Profit and Loss (PnL)")
    st.markdown("PnL (profit_and_loss) over time for individual assets.")
    asset = st.selectbox("Select asset for PnL plot:", ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"])
    pnl_fig = plot_pnl(orders_df, asset)
    st.pyplot(pnl_fig)
    
    st.markdown("### End of Visualisation")
    
if __name__ == '__main__':
    main()
