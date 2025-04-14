import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Folder where the Excel files are stored
data_folder = r"C:\Users\Roger\Cloud\Prosperity\combined_data"

# Volcanic rock + vouchers
volcanic_file = "volcanic_rock.xlsx"
voucher_files = [
    "volcanic_rock_voucher_9500.xlsx",
    "volcanic_rock_voucher_9750.xlsx",
    "volcanic_rock_voucher_10000.xlsx",
    "volcanic_rock_voucher_10250.xlsx",
    "volcanic_rock_voucher_10500.xlsx"
]

# Corresponding values for vouchers (for horizontal lines)
voucher_levels = {
    "volcanic_rock_voucher_9500.xlsx": 9500,
    "volcanic_rock_voucher_9750.xlsx": 9750,
    "volcanic_rock_voucher_10000.xlsx": 10000,
    "volcanic_rock_voucher_10250.xlsx": 10250,
    "volcanic_rock_voucher_10500.xlsx": 10500
}

# Create two vertically stacked subplots with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[1, 2])

# Plot volcanic_rock.xlsx on ax1
volcanic_path = os.path.join(data_folder, volcanic_file)
if os.path.exists(volcanic_path):
    try:
        df = pd.read_excel(volcanic_path)
        if 'mid_price' in df.columns:
            ax1.plot(df['mid_price'].reset_index(drop=True), label="volcanic_rock", color='brown')
            ax1.set_ylabel("Mid Price")
            ax1.set_title("Volcanic Rock")
        else:
            print(f"⚠️ No 'mid_price' in {volcanic_file}, skipping.")
    except Exception as e:
        print(f"❌ Error reading {volcanic_file}: {e}")
else:
    print(f"❌ File not found: {volcanic_file}")

# Colors for vouchers
colors = plt.cm.viridis_r(np.linspace(0.2, 0.9, len(voucher_files)))

# Plot vouchers on ax2 and add horizontal lines to ax1
for i, file_name in enumerate(voucher_files):
    file_path = os.path.join(data_folder, file_name)
    color = colors[i]
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
            if 'mid_price' in df.columns:
                ax2.plot(df['mid_price'].reset_index(drop=True), label=file_name.replace('.xlsx', ''), color=color)

                # Draw transparent horizontal line on top plot
                y_value = voucher_levels[file_name]
                ax1.axhline(y=y_value, color=color, linestyle='--', alpha=0.7, label=f"{y_value} level")
            else:
                print(f"⚠️ No 'mid_price' in {file_name}, skipping.")
        except Exception as e:
            print(f"❌ Error reading {file_name}: {e}")
    else:
        print(f"❌ File not found: {file_name}")

# Final settings
ax1.legend()
ax2.set_title("Volcanic Rock Vouchers")
ax2.set_xlabel("Index (Time)")
ax2.set_ylabel("Mid Price")
ax2.legend()
ax1.grid(True)
ax2.grid(True)

plt.tight_layout()
plt.show()
