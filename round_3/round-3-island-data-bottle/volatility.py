import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Folder where the Excel files are stored
data_folder = r"C:\Users\Roger\Cloud\Prosperity\combined_data"

# File names
volcanic_file = "volcanic_rock.xlsx"
voucher_files = [
    "volcanic_rock_voucher_9500.xlsx",
    "volcanic_rock_voucher_9750.xlsx",
    "volcanic_rock_voucher_10000.xlsx",
    "volcanic_rock_voucher_10250.xlsx",
    "volcanic_rock_voucher_10500.xlsx"
]
voucher_levels = {
    "volcanic_rock_voucher_9500.xlsx": 9500,
    "volcanic_rock_voucher_9750.xlsx": 9750,
    "volcanic_rock_voucher_10000.xlsx": 10000,
    "volcanic_rock_voucher_10250.xlsx": 10250,
    "volcanic_rock_voucher_10500.xlsx": 10500
}

# Read volcanic rock mid prices
volcanic_path = os.path.join(data_folder, volcanic_file)
volcanic_mid = None
if os.path.exists(volcanic_path):
    try:
        df_volcanic = pd.read_excel(volcanic_path)
        if 'mid_price' in df_volcanic.columns:
            volcanic_mid = df_volcanic['mid_price'].reset_index(drop=True)
        else:
            print(f"⚠️ No 'mid_price' in {volcanic_file}, skipping.")
    except Exception as e:
        print(f"❌ Error reading {volcanic_file}: {e}")
else:
    print(f"❌ File not found: {volcanic_file}")

# Check if volcanic_mid was read successfully
if volcanic_mid is None:
    raise RuntimeError("Missing volcanic rock mid prices. Cannot compute volatility.")

# Calculate volatility (rolling standard deviation) for volcanic rock
window_size = 20  # You can adjust this window size
volcanic_volatility = volcanic_mid.rolling(window=window_size).std()

# Set up plots
n_vouchers = len(voucher_files)
fig, axes = plt.subplots(n_vouchers + 1, 1, figsize=(14, 3 * (n_vouchers + 1)), sharex=True)
colors = plt.cm.plasma(np.linspace(0.2, 0.9, n_vouchers))

# Plot volatility for each voucher
for i, file_name in enumerate(voucher_files):
    file_path = os.path.join(data_folder, file_name)
    strike_price = voucher_levels[file_name]
    color = colors[i]
    ax = axes[i]

    if os.path.exists(file_path):
        try:
            df_voucher = pd.read_excel(file_path)
            if 'mid_price' in df_voucher.columns:
                premium = df_voucher['mid_price'].reset_index(drop=True)
                min_len = min(len(volcanic_mid), len(premium))

                # Calculate volatility (rolling standard deviation) for the voucher
                voucher_volatility = premium[:min_len].rolling(window=window_size).std()

                # Plot volatility for the voucher
                ax.plot(voucher_volatility, label=f"{file_name.replace('.xlsx', '')} Volatility", color=color)

                ax.set_ylabel("Volatility")
                ax.set_title(f"Volatility: {file_name.replace('.xlsx', '')}")
                ax.grid(True)
                ax.legend(loc='upper right')
            else:
                print(f"⚠️ No 'mid_price' in {file_name}, skipping.")
        except Exception as e:
            print(f"❌ Error reading {file_name}: {e}")
    else:
        print(f"❌ File not found: {file_name}")

# Plot volcanic rock volatility in its own subplot
volcanic_ax = axes[-1]
volcanic_ax.plot(volcanic_volatility, label="Volcanic Rock Volatility", color='black', linestyle='--')
volcanic_ax.set_ylabel("Volatility")
volcanic_ax.set_title("Volcanic Rock Volatility")
volcanic_ax.grid(True)
volcanic_ax.legend(loc='upper right')

# Set xlabel for the bottom plot
axes[-1].set_xlabel("Index (Time)")

plt.tight_layout()
plt.show()
