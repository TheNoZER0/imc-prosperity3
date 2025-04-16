import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Folder where the Excel files are stored
data_folder = r"C:\Users\Roger\Cloud\Prosperity\combined_data"

# Files
volcanic_file = "volcanic_rock.xlsx"
voucher_files = [
    "volcanic_rock_voucher_9500.xlsx",
    "volcanic_rock_voucher_9750.xlsx",
    "volcanic_rock_voucher_10000.xlsx",
    "volcanic_rock_voucher_10250.xlsx",
    "volcanic_rock_voucher_10500.xlsx"
]

# Load volcanic rock data
vol_path = os.path.join(data_folder, volcanic_file)
vol_df = pd.read_excel(vol_path)
vol_mid = vol_df['mid_price'].reset_index(drop=True)

# Set up subplots: 1 for volcanic rock + 5 for deltas
fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
colors = plt.cm.viridis_r(np.linspace(0.2, 0.9, len(voucher_files)))

# Plot volcanic rock mid price at the top
axes[0].plot(vol_mid, color='brown', label='Volcanic Rock Mid Price')
axes[0].set_ylabel("Mid Price")
axes[0].set_title("Volcanic Rock")
axes[0].legend()
axes[0].grid(True)

# Loop over vouchers to compute and plot delta
for i, file_name in enumerate(voucher_files):
    file_path = os.path.join(data_folder, file_name)
    voucher_df = pd.read_excel(file_path)
    voucher_mid = voucher_df['mid_price'].reset_index(drop=True)

    # Align lengths
    min_len = min(len(voucher_mid), len(vol_mid))
    voucher_mid = voucher_mid[:min_len]
    vol_trimmed = vol_mid[:min_len]

    # Compute delta using finite difference
    delta = np.diff(voucher_mid) / np.diff(vol_trimmed)
    delta = np.insert(delta, 0, np.nan)  # pad to match length

    # Plot in the corresponding subplot
    axes[i + 1].plot(delta, label=file_name.replace(".xlsx", ""), color=colors[i])
    axes[i + 1].set_ylabel("Delta")
    axes[i + 1].legend()
    axes[i + 1].grid(True)

# Final label
axes[-1].set_xlabel("Index (Time)")
plt.tight_layout()
plt.show()
