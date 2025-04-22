#!/usr/bin/env python3
"""
hidden_value_estimator.py

This script reads two CSV files (an orderbook and a trades dataset) whose paths
and parameters are embedded in the file. It estimates the fair value of a given product
by computing:
  
  - An EMA of the mid_price from the orderbook data.
  - An EMA of the volume–weighted average price (VWAP) from the trades data.

It then combines these estimates (using a simple average) to produce a hidden fair value.

Additionally, it calculates the theoretical maximum return obtainable from the asset using the 
orderbook mid_price series. In addition to the maximum profit from a single trade 
(buy low, sell high), it also calculates the cumulative profit (i.e. the sum of all positive 
price differences available if one traded on every upward move).

Data CSVs are expected to be semicolon‑delimited.
"""

import pandas as pd
import numpy as np

# --- Configuration (update these paths and parameters as needed) ---
ORDERBOOK_CSV = "data/round2/basket/prices_round_2_day_-1.csv"  # Path to your orderbook (prices) dataset
TRADES_CSV = "data/round2/basket/trades_round_2_day_-1.csv"       # Path to your trades dataset
PRODUCT = "JAMS"              # Product to estimate (e.g. "KELP" or "SQUID_INK")
EMA_SPAN = 15                 # Span used in EMA smoothing

# --- Data Loading Functions ---
def load_orderbook_data(file_path: str) -> pd.DataFrame:
    """Load the orderbook CSV (semicolon delimited) and derive best bid/ask."""
    df = pd.read_csv(file_path, delimiter=';')
    # Convert timestamp to datetime if possible
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Identify columns for bid and ask prices.
    bid_cols = [col for col in df.columns if col.startswith('bid_price')]
    ask_cols = [col for col in df.columns if col.startswith('ask_price')]
    
    # Derive best bid: maximum of bid prices (or simply bid_price_1 if pre-sorted)
    df['best_bid'] = df[bid_cols].max(axis=1, numeric_only=True)
    # Derive best ask: minimum of ask prices
    df['best_ask'] = df[ask_cols].min(axis=1, numeric_only=True)
    
    # Use provided mid_price if available; otherwise compute as the average.
    if 'mid_price' not in df.columns or df['mid_price'].isnull().all():
        df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2

    return df

def load_trades_data(file_path: str) -> pd.DataFrame:
    """Load the trades CSV (semicolon delimited) and convert timestamp."""
    df = pd.read_csv(file_path, delimiter=';')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

# --- Fair Value Estimation Functions ---
def calculate_orderbook_fair_value(df: pd.DataFrame, product: str, span: int = EMA_SPAN) -> float:
    """
    Calculate a smoothed EMA of the mid_price from orderbook data for a given product.
    """
    prod_df = df[df['product'] == product].sort_values('timestamp')
    if prod_df.empty:
        raise ValueError(f"No orderbook data available for product '{product}'.")
    prod_df['ema_mid'] = prod_df['mid_price'].ewm(span=span, adjust=False).mean()
    return prod_df['ema_mid'].iloc[-1]

def calculate_trade_fair_value(df: pd.DataFrame, product: str, span: int = EMA_SPAN) -> float:
    """
    Calculate a smoothed volume-weighted average price (VWAP) from trades data for a given product.
    """
    prod_df = df[df['symbol'] == product].sort_values('timestamp')
    if prod_df.empty:
        raise ValueError(f"No trades data available for product '{product}'.")
    
    # Compute total trade value and cumulative volume, then VWAP.
    prod_df['total_trade_value'] = prod_df['price'] * prod_df['quantity']
    prod_df['cum_volume'] = prod_df['quantity'].cumsum()
    prod_df['cum_trade_value'] = prod_df['total_trade_value'].cumsum()
    prod_df['vwap'] = prod_df['cum_trade_value'] / prod_df['cum_volume']
    
    # Smooth the VWAP with an EMA
    prod_df['ema_vwap'] = prod_df['vwap'].ewm(span=span, adjust=False).mean()
    return prod_df['ema_vwap'].iloc[-1]

def combine_fair_values(ob_value: float, trade_value: float) -> float:
    """Combine the two fair value estimates (orderbook and trades) using a simple average."""
    return (ob_value + trade_value) / 2

# --- Theoretical Maximum Returns Functions ---
def calculate_theoretical_max_return(df: pd.DataFrame, product: str) -> dict:
    """
    Calculate the theoretical maximum return obtainable from the asset using its historical mid_price series.
    
    This function calculates two measures:
      1. Single Trade Max Profit: The maximum profit per unit possible by buying at the lowest price 
         and selling at a later highest price.
      2. Cumulative Profit: Sum of all positive price differences (i.e. the sum of all upward movements)
         if you were allowed to trade on every increase.
    
    Returns a dictionary with:
      - max_profit: Maximum profit from a single trade.
      - buy_price: The lowest price at which to buy for maximum single-trade profit.
      - sell_price: The corresponding later price at which to sell.
      - percent_return: Percentage return from the single best trade.
      - cumulative_profit: Total profit obtainable by summing every positive price movement.
      - cumulative_percent_return: Cumulative percentage return relative to the first price.
    """
    prod_df = df[df['product'] == product].sort_values('timestamp')
    if prod_df.empty:
        raise ValueError(f"No data available for product '{product}'.")
    
    prices = prod_df['mid_price'].values
    if len(prices) == 0:
        raise ValueError("No mid_price data available.")

    min_price = float('inf')
    max_profit = 0
    buy_price = None
    sell_price = None

    # Compute single trade maximum profit
    for price in prices:
        if price < min_price:
            min_price = price
        profit = price - min_price
        if profit > max_profit:
            max_profit = profit
            buy_price = min_price
            sell_price = price

    percent_return = (max_profit / buy_price) * 100 if buy_price and buy_price != 0 else 0

    # Compute cumulative profit (maximum profit if you traded on every positive price movement)
    cumulative_profit = sum(max(prices[i+1] - prices[i], 0) for i in range(len(prices) - 1))
    cumulative_percent_return = (cumulative_profit / prices[0]) * 100 if prices[0] != 0 else 0

    return {
        "max_profit": max_profit,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "percent_return": percent_return,
        "cumulative_profit": cumulative_profit,
        "cumulative_percent_return": cumulative_percent_return,
    }

# --- Main Execution ---
def main():
    # Load the datasets from the hardcoded file paths.
    orderbook_df = load_orderbook_data(ORDERBOOK_CSV)
    trades_df = load_trades_data(TRADES_CSV)
    
    # Compute fair value estimates from orderbook and trades data
    try:
        ob_fair_value = calculate_orderbook_fair_value(orderbook_df, PRODUCT, EMA_SPAN)
    except Exception as e:
        print(f"Error calculating orderbook fair value: {e}")
        ob_fair_value = None
        
    try:
        trade_fair_value = calculate_trade_fair_value(trades_df, PRODUCT, EMA_SPAN)
    except Exception as e:
        print(f"Error calculating trades fair value: {e}")
        trade_fair_value = None

    if ob_fair_value is None or trade_fair_value is None:
        print("Insufficient data to compute fair value estimates.")
    else:
        combined_value = combine_fair_values(ob_fair_value, trade_fair_value)
        print(f"Estimated Fair Value for {PRODUCT}:")
        print(f"  Orderbook EMA mid_price: {ob_fair_value:.2f}")
        print(f"  Trades EMA (VWAP): {trade_fair_value:.2f}")
        print(f"  Combined Estimate: {combined_value:.2f}")

    # Calculate theoretical maximum return using the orderbook data
    try:
        max_return_info = calculate_theoretical_max_return(orderbook_df, PRODUCT)
        print("\nTheoretical Maximum Return from Orderbook data:")
        print(f"  Single Best Trade:")
        print(f"    Buy at: {max_return_info['buy_price']:.2f}")
        print(f"    Sell at: {max_return_info['sell_price']:.2f}")
        print(f"    Maximum Profit: {max_return_info['max_profit']:.2f}")
        print(f"    Percent Return: {max_return_info['percent_return']:.2f}%")
        print(f"  Cumulative Trading Profit (all positive moves):")
        print(f"    Cumulative Profit: {max_return_info['cumulative_profit']:.2f}")
        print(f"    Cumulative Percent Return: {max_return_info['cumulative_percent_return']:.2f}%")
    except Exception as e:
        print(f"Error calculating theoretical maximum return: {e}")

if __name__ == '__main__':
    main()
