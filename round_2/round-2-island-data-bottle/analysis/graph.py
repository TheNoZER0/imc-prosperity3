import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Plot setup
fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

axs[0].plot(cros_price.index, cros_price, color='saddlebrown')
axs[0].set_ylabel('Croissants')
axs[0].grid()

axs[1].plot(djem_price.index, djem_price, color='steelblue')
axs[1].set_ylabel('Djembes')
axs[1].grid()

axs[2].plot(jams_price.index, jams_price, color='orchid')
axs[2].set_ylabel('Jams')
axs[2].grid()

axs[3].plot(basket1_price.index, basket1_price, label='Market', color='gold')
axs[3].plot(basket1_real.index, basket1_real, label='Real', color='darkorange')
axs[3].set_ylabel('Basket 1')
axs[3].legend()
axs[3].grid()

axs[4].plot(basket2_price.index, basket2_price, label='Market', color='seagreen')
axs[4].plot(basket2_real.index, basket2_real, label='Real', color='darkgreen')
axs[4].set_ylabel('Basket 2')
axs[4].set_xlabel('Timestamp (Index)')
axs[4].legend()
axs[4].grid()

fig.suptitle("Round 2 Prices", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



