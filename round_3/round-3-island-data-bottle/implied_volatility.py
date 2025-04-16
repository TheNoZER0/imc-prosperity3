import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === Load Data ===
file_path = r"C:\Users\Roger\Cloud\Prosperity\round_3\round-3-island-data-bottle\implied_volatilities.xlsx"
df = pd.read_excel(file_path)

# === Identify volatility columns ===
vol_cols = [col for col in df.columns if col.startswith("voucher_")]
num_plots = len(vol_cols)

# === Set Up Subplots ===
fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3.5 * num_plots))

if num_plots == 1:
    axes = [axes]  # Ensure iterable for one subplot

for i, col in enumerate(vol_cols):
    series = df[col].dropna()

    # === Stats ===
    q25, q75 = np.percentile(series, [25, 75])
    iqr = q75 - q25
    mean = series.mean()
    median = series.median()

    # === Freedman–Diaconis bin width ===
    bin_width = 2 * iqr * (len(series) ** (-1 / 3))
    if bin_width == 0: bin_width = 1e-4
    bins = int(np.ceil((series.max() - series.min()) / bin_width))

    ax = axes[i]
    ax.hist(series, bins=bins, color='tab:orange', edgecolor='black')

    # === Plot Mean, Median, IQR ===
    ax.axvline(mean, color='blue', linestyle='-', label='Mean')
    ax.axvline(median, color='green', linestyle='--', label='Median')
    ax.axvline(q25, color='red', linestyle=':', label='Q1 / Q3')
    ax.axvline(q75, color='red', linestyle=':')

    ax.set_title(f"{col.replace('_', ' ').title()} - Histogram")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel("Implied Volatility")
plt.tight_layout()

# === Save Plot ===
output_plot_path = os.path.join(os.path.dirname(file_path), "implied_volatility_histograms.png")
plt.savefig(output_plot_path)
plt.show()
print(f"Saved plot to: {output_plot_path}")

import pandas as pd
import numpy as np
import os

# === Load Data ===
file_path = r"C:\Users\Roger\Cloud\Prosperity\round_3\round-3-island-data-bottle\implied_volatilities.xlsx"
df = pd.read_excel(file_path)

# === Identify volatility columns ===
vol_cols = [col for col in df.columns if col.startswith("voucher_")]

# === Summary Stats ===
print("Summary statistics:\n")

for col in vol_cols:
    series = df[col].dropna()
    mean = series.mean()
    median = series.median()
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1
    std = series.std()
    var = series.var()
    count = series.count()
    min_val = series.min()
    max_val = series.max()

    print(f"--- {col} ---")
    print(f"Count   : {count}")
    print(f"Mean    : {mean:.8f}")
    print(f"Median  : {median:.8f}")
    print(f"Q1      : {q1:.8f}")
    print(f"Q3      : {q3:.8f}")
    print(f"IQR     : {iqr:.8f}")
    print(f"Std Dev : {std:.8f}")
    print(f"Variance: {var:.10f}")
    print(f"Min     : {min_val:.8f}")
    print(f"Max     : {max_val:.8f}")
    print()
