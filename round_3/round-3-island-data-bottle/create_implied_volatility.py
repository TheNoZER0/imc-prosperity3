import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import brentq

# === Configuration ===
DATA_FOLDER = r"C:\Users\Roger\Cloud\Prosperity\combined_data"
VOLCANIC_FILE = "volcanic_rock.xlsx"
VOUCHER_FILES = [
    "volcanic_rock_voucher_9500.xlsx",
    "volcanic_rock_voucher_9750.xlsx",
    "volcanic_rock_voucher_10000.xlsx",
    "volcanic_rock_voucher_10250.xlsx",
    "volcanic_rock_voucher_10500.xlsx"
]
STRIKE_PRICES = {
    "volcanic_rock_voucher_9500.xlsx": 9500,
    "volcanic_rock_voucher_9750.xlsx": 9750,
    "volcanic_rock_voucher_10000.xlsx": 10000,
    "volcanic_rock_voucher_10250.xlsx": 10250,
    "volcanic_rock_voucher_10500.xlsx": 10500
}
RISK_FREE_RATE = 0.0

# === Black-Scholes Call Pricing ===
def norm_cdf(x):
    """More accurate standard normal CDF using the Abramowitz and Stegun approximation."""
    # Constants for the approximation
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    p  = 0.2316419
    c  = 0.3989422804014337  # 1 / sqrt(2 * pi)

    # Sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # Calculation of the approximation
    t = 1 / (1 + p * x)
    poly = a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
    result = 1 - c * math.exp(-x**2 / 2) * poly

    return 0.5 * (1 + sign * result)

def bs_call_price(spot, strike, T, r, sigma):
    """Black-Scholes Call Price."""
    if sigma == 0 or T == 0:  # Handle edge cases to prevent division by zero
        return 0  # At-the-money or deep in-the-money
    sqrt_T = np.sqrt(T)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return spot * norm_cdf(d1) - strike * np.exp(-r * T) * norm_cdf(d2)

# External objective function for implied volatility calculation
def objective(sigma, market_price, spot, strike, T, r):
    return bs_call_price(spot, strike, T, r, sigma) - market_price

def implied_volatility_call(market_price, spot, strike, T, r, tol=1e-20, max_iter=15000):
    """Find implied volatility using Brent's method."""
    try:
        implied_vol = brentq(objective, 1e-15, 10.0, args=(market_price, spot, strike, T, r), xtol=tol, maxiter=max_iter)
        return implied_vol
    except ValueError:
        return np.nan  # Return NaN if no solution is found in the range

# === Load Volcanic Rock Prices ===
vol_df = pd.read_excel(os.path.join(DATA_FOLDER, VOLCANIC_FILE))
vol_mid = vol_df['mid_price'].reset_index(drop=True)
log_returns = np.log(vol_mid / vol_mid.shift(1)).dropna()
est_sigma = np.std(log_returns) * np.sqrt(252)

# === Plot Setup ===
fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
colors = plt.cm.viridis_r(np.linspace(0.2, 0.9, len(VOUCHER_FILES)))

# Plot Volcanic Rock (just for reference)
axes[0].plot(vol_mid, label="Volcanic Rock Mid Price", color="brown")
axes[0].set_ylabel("Price")
axes[0].set_title("Volcanic Rock")
axes[0].legend()
axes[0].grid(True)

# === Filter Out Invalid Volatility ===
def filter_volatility(implied_vols, window=5, threshold=0.0001):
    """Filter out values close to zero and those more than 1 standard deviation away from the moving average."""
    # Calculate rolling mean and standard deviation (window = 5)
    rolling_mean = pd.Series(implied_vols).rolling(window=window, min_periods=1).mean().to_numpy()
    rolling_std = pd.Series(implied_vols).rolling(window=window, min_periods=1).std().to_numpy()

    # Create a mask for valid values
    valid_mask = (np.abs(implied_vols) > threshold) & \
                 (np.abs(implied_vols - rolling_mean) <= rolling_std)

    return valid_mask

# === Process Each Voucher ===
for i, file_name in enumerate(VOUCHER_FILES):
    print(f"starting {file_name}")
    strike = STRIKE_PRICES[file_name]
    voucher_df = pd.read_excel(os.path.join(DATA_FOLDER, file_name))
    voucher_mid = voucher_df['mid_price'].reset_index(drop=True)

    # Align lengths
    min_len = min(len(vol_mid), len(voucher_mid))
    spot_trim = vol_mid[:min_len].to_numpy()
    premium = voucher_mid[:min_len].to_numpy()
    indices = np.arange(min_len)

    # Calculate time to expiry (in years) for each index
    T_array = ((80000 - indices) / 10000) / 365.25

    # Compute Implied Volatility for each point
    implied_vols = np.array([
        implied_volatility_call(premium[t], spot_trim[t], strike, T_array[t], RISK_FREE_RATE)
        if premium[t] > 0 and spot_trim[t] > 0 else np.nan
        for t in range(min_len)
    ])

    # Apply filtering
    valid_indices = filter_volatility(implied_vols)

    # === Plot Implied Volatility ===
    ax = axes[i + 1]
    ax.scatter(indices[valid_indices], implied_vols[valid_indices], label=f"Implied Vol (K={strike})", color=colors[i], s=1)
    ax.set_ylabel("Implied Volatility")
    ax.legend(loc="upper left")
    ax.grid(True)

# === Finalize Plot ===
axes[-1].set_xlabel("Time Index")
plt.tight_layout()
plt.show()
