import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Load data
cros_data = pd.read_excel(r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\croissants.xlsx")
djem_data = pd.read_excel(r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\djembes.xlsx")
jams_data = pd.read_excel(r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\jams.xlsx")
pic1_data = pd.read_excel(r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\picnic_basket1.xlsx")
pic2_data = pd.read_excel(r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\picnic_basket2.xlsx")

# Extract mid_price columns
cros_price = cros_data['mid_price']
djem_price = djem_data['mid_price']
jams_price = jams_data['mid_price']
basket1_price = pic1_data['mid_price']
basket2_price = pic2_data['mid_price']

# Compute real prices
def basket1_real_price(cros_price, djem_price, jams_price):
    return 6 * cros_price + djem_price + 3 * jams_price

def basket2_real_price(cros_price, jams_price):
    return 4 * cros_price + 2 * jams_price

basket1_real = basket1_real_price(cros_price, djem_price, jams_price)
basket2_real = basket2_real_price(cros_price, jams_price)

# Compute differences between market and real prices
basket1_diff = basket1_price - basket1_real

# Cross-correlation to test for time offset (between Basket 1 Market Price and Real Price)
def compute_cross_correlation(series1, series2, max_lag=2000):
    correlation = []
    for lag in range(-max_lag, max_lag + 1):
        shifted_series1 = series1.shift(lag)
        # Drop NaN values (due to the shift)
        correlation.append(series2.corr(shifted_series1.dropna()))
    return correlation

# Perform cross-correlation on Basket 1 market price vs real price
correlation_basket1 = compute_cross_correlation(basket1_price, basket1_real, max_lag=2000)

# Plot cross-correlation results
plt.figure(figsize=(10, 6))
plt.plot(range(-2000, 2001), correlation_basket1)
plt.title('Cross-correlation between Basket 1 Market Price and Real Price')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.grid(True)
plt.show()

# Find the lag with the maximum correlation
max_correlation = max(correlation_basket1)
max_lag = correlation_basket1.index(max_correlation) - 2000  # Adjusting index for range(-2000, 2001)

print(f"Maximum correlation of {max_correlation:.4f} found at lag {max_lag}")

# Save the results of the differences and correlation to Excel
diff_data = {
    'Basket 1 Diff': basket1_diff
}

diff_df = pd.DataFrame(diff_data)

# Save to Excel
diff_file = r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\basket1_diff_and_correlation.xlsx"
with pd.ExcelWriter(diff_file) as writer:
    diff_df.to_excel(writer, sheet_name='Differences', index=False)
    pd.DataFrame(correlation_basket1, columns=['Correlation']).to_excel(writer, sheet_name='Cross-correlation', index=False)

print(f"Results saved to {diff_file}")
