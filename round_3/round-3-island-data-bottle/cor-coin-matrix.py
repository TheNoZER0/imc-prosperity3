import os

import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import coint
import seaborn as sns
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
# Load all series into a single DataFrame
price_data = {}

# Load volcanic rock
volcanic_df = pd.read_excel(os.path.join(data_folder, volcanic_file))
if 'mid_price' in volcanic_df.columns:
    price_data['volcanic_rock'] = volcanic_df['mid_price'].reset_index(drop=True)

# Load vouchers
for file_name in voucher_files:
    file_path = os.path.join(data_folder, file_name)
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        if 'mid_price' in df.columns:
            key = file_name.replace('.xlsx', '')
            price_data[key] = df['mid_price'].reset_index(drop=True)

# Combine into one DataFrame (align lengths)
combined_df = pd.DataFrame(price_data).dropna()

# ---------- CORRELATION ----------
corr_matrix = combined_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="rocket_r", fmt=".4f", vmin=0, vmax=1)
plt.title("Correlation Matrix (Pearson)")
plt.tight_layout()
plt.show()

# ---------- COINTEGRATION ----------
cointegration_matrix = pd.DataFrame(index=combined_df.columns, columns=combined_df.columns)
pval_matrix = pd.DataFrame(index=combined_df.columns, columns=combined_df.columns)

for i in combined_df.columns:
    for j in combined_df.columns:
        if i == j:
            cointegration_matrix.loc[i, j] = 1.0
            pval_matrix.loc[i, j] = 0.0
        else:
            score, pvalue, _ = coint(combined_df[i], combined_df[j])
            cointegration_matrix.loc[i, j] = round(score, 2)
            pval_matrix.loc[i, j] = round(pvalue, 4)

# Plot p-values (lower = stronger evidence of cointegration)
plt.figure(figsize=(10, 8))
sns.heatmap(pval_matrix.astype(float), annot=True, cmap="YlGnBu", fmt=".4f")
plt.title("Cointegration Test (p-values)")
plt.tight_layout()
plt.show()
