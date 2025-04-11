import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

# List of file paths
file_paths = [
    r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\basket1_diff.xlsx",
    r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\basket2_diff.xlsx",
    r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\croissants.xlsx",
    r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\djembes.xlsx",
    r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\jams.xlsx",
    r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\picnic_basket1.xlsx",
    r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\picnic_basket2.xlsx",
]

# Load the data
data = {}
for file_path in file_paths:
    df = pd.read_excel(file_path)
    data[file_path] = df['mid_price']

# Cross-correlation function
def compute_cross_correlation(series1, series2, max_lag=2000):
    correlation = []
    for lag in range(-max_lag, max_lag + 1):
        shifted_series1 = series1.shift(lag)
        correlation.append(series2.corr(shifted_series1.dropna()))
    return correlation

# Ensure the "png" directory exists
png_dir = r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\png"
os.makedirs(png_dir, exist_ok=True)

# Initialize a list to store results
correlation_results = []

# Compare all pairs of datasets
for file_path1, series1 in data.items():
    for file_path2, series2 in data.items():
        if file_path1 != file_path2:
            print("starting cross-correlation between", file_path1, "and", file_path2)
            # Compute cross-correlation between series1 and series2
            correlation = compute_cross_correlation(series1, series2, max_lag=2000)
            max_correlation = max(correlation)
            max_lag = correlation.index(max_correlation) - 2000  # Adjusting index for range(-2000, 2001)

            # Store results
            correlation_results.append({
                'Dataset 1': file_path1,
                'Dataset 2': file_path2,
                'Max Correlation': max_correlation,
                'Lag': max_lag
            })

            # Plot the cross-correlation and save to PNG
            plt.figure(figsize=(10, 6))
            plt.plot(range(-2000, 2001), correlation)
            plt.title(f'Cross-correlation between {file_path1} and {file_path2}')
            plt.xlabel('Lag')
            plt.ylabel('Correlation')
            plt.grid(True)
            plt.ylim(0, 1)  # Enforce y-axis from 0 to 1
            plot_filename = os.path.join(png_dir, f"cross_correlation_{file_path1.split('\\')[-1]}_vs_{file_path2.split('\\')[-1]}.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved cross-correlation plot: {plot_filename}")

# Convert results to a DataFrame
correlation_df = pd.DataFrame(correlation_results)

# Save results to an Excel file
results_file = r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\cross_correlation_results.xlsx"
with pd.ExcelWriter(results_file) as writer:
    correlation_df.to_excel(writer, sheet_name='Correlation Results', index=False)

print(f"Cross-correlation results saved to {results_file}")
print(f"Graphs saved in {png_dir}")
