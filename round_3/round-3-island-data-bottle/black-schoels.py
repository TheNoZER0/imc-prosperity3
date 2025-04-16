import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
import math

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
@njit
def norm_cdf(x):
    """Approximate standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


@njit
def bs_call_price(spot, strike, T, r, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T + 1e-10)
    d2 = d1 - sigma * sqrt_T
    return spot * norm_cdf(d1) - strike * np.exp(-r * T) * norm_cdf(d2)


@njit
def implied_volatility_call(market_price, spot, strike, T, r, tol=1e-6, max_iter=100):
    low, high = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        price = bs_call_price(spot, strike, T, r, mid)
        diff = price - market_price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            high = mid
        else:
            low = mid
    return np.nan


# === Load Volcanic Rock Prices ===
vol_df = pd.read_excel(os.path.join(DATA_FOLDER, VOLCANIC_FILE))
vol_mid = vol_df['mid_price'].reset_index(drop=True)
log_returns = np.log(vol_mid / vol_mid.shift(1)).dropna()
est_sigma = np.std(log_returns) * np.sqrt(252)

# === Plot Setup ===
fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
colors = plt.cm.viridis_r(np.linspace(0.2, 0.9, len(VOUCHER_FILES)))

# Plot Volcanic Rock
axes[0].plot(vol_mid, label="Volcanic Rock Mid Price", color="brown")
axes[0].set_ylabel("Price")
axes[0].set_title("Volcanic Rock")
axes[0].legend()
axes[0].grid(True)

# === Process Each Voucher ===
for i, file_name in enumerate(VOUCHER_FILES):
    strike = STRIKE_PRICES[file_name]
    voucher_df = pd.read_excel(os.path.join(DATA_FOLDER, file_name))
    voucher_mid = voucher_df['mid_price'].reset_index(drop=True)

    # Align lengths
    min_len = min(len(vol_mid), len(voucher_mid))
    spot_trim = vol_mid[:min_len].to_numpy()
    premium = voucher_mid[:min_len].to_numpy()
    indices = np.arange(min_len)
    T_array = np.clip((90000 - indices) / 365, 0.0001, None)

    # Compute Black-Scholes Prices and Implied Volatility
    bs_prices = np.array([bs_call_price(spot_trim[t], strike, T_array[t], RISK_FREE_RATE, est_sigma)
                          for t in range(min_len)])

    implied_vols = np.array([
        implied_volatility_call(premium[t], spot_trim[t], strike, T_array[t], RISK_FREE_RATE)
        if premium[t] > 0 and spot_trim[t] > 0 else np.nan
        for t in range(min_len)
    ])

    # === Plot Prices ===
    ax = axes[i + 1]
    ax.plot(premium, label=f"Voucher Mid (K={strike})", color=colors[i])
    ax.plot(bs_prices, label="BS Price", linestyle='--', color=colors[i])
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(True)

    # === Plot Implied Volatility ===
    ax2 = ax.twinx()
    ax2.plot(implied_vols, label="Implied Vol", color="black", alpha=0.3)
    ax2.set_ylabel("IV", color="black")
    ax2.tick_params(axis='y', labelcolor="black")

# === Finalize Plot ===
axes[-1].set_xlabel("Time Index")
plt.tight_layout()
plt.show()
