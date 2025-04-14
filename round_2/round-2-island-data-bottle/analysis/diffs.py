import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# List of file paths
r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\basket1_diff.xlsx",
r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\basket2_diff.xlsx",
diff1 = pd.read_excel(r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\basket1_diff.xlsx")
diff1_data = diff1['mid_price']

diff2 = pd.read_excel(r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\basket2_diff.xlsx")
diff2_data = diff2['mid_price']


plt.plot(diff1_data, label='Basket 1 Diff', color='blue')
plt.plot(diff2_data, label='Basket 2 Diff', color='orange')
plt.title('Difference between Market and Real Prices')
plt.xlabel('Timestamp (Index)')
plt.ylabel('Difference')
plt.legend()
plt.grid()
plt.show()
