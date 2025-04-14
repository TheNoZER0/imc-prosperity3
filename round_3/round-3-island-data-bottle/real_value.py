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
    raise RuntimeError("Missing volcanic rock mid prices. Cannot compute real values.")

# Set up subplots: one for volcanic rock + one per voucher
n_plots = 1 + len(voucher_files)
fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(voucher_files)))

# Plot volcanic rock price on top
axes[0].plot(volcanic_mid, color='brown')
axes[0].set_title("Volcanic Rock Mid Price")
axes[0].set_ylabel("Price")
axes[0].grid(True)

# Plot real value and premium for each voucher
for i, file_name in enumerate(voucher_files):
    file_path = os.path.join(data_folder, file_name)
    strike_price = voucher_levels[file_name]
    color = colors[i]
    ax = axes[i + 1]

    if os.path.exists(file_path):
        try:
            df_voucher = pd.read_excel(file_path)
            if 'mid_price' in df_voucher.columns:
                premium = df_voucher['mid_price'].reset_index(drop=True)
                min_len = min(len(volcanic_mid), len(premium))

                intrinsic_value = np.maximum(volcanic_mid[:min_len] - strike_price, 0)
                real_value = intrinsic_value - premium[:min_len]

                ax.plot(real_value, color=color, label="Real Value")
                ax.set_ylabel("Real Value")
                ax.set_title(f"{file_name.replace('.xlsx', '')} (Strike: {strike_price})")
                ax.grid(True)

                # Twin axis for premium
                ax2 = ax.twinx()
                ax2.plot(premium[:min_len], color='gray', linestyle='--', alpha=0.6, label="Premium")
                ax2.set_ylabel("Premium")

                # Optional: add legend combining both
                lines_1, labels_1 = ax.get_legend_handles_labels()
                lines_2, labels_2 = ax2.get_legend_handles_labels()
                ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

            else:
                print(f"⚠️ No 'mid_price' in {file_name}, skipping.")
        except Exception as e:
            print(f"❌ Error reading {file_name}: {e}")
    else:
        print(f"❌ File not found: {file_name}")

# Final styling
axes[-1].set_xlabel("Index (Time)")
plt.tight_layout()
plt.show()
