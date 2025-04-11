import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
cros = pd.read_excel(r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\croissants.xlsx")['mid_price']
djem = pd.read_excel(r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\djembes.xlsx")['mid_price']
jams = pd.read_excel(r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\jams.xlsx")['mid_price']
pic1 = pd.read_excel(r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\picnic_basket1.xlsx")['mid_price']
pic2 = pd.read_excel(r"C:\Users\Roger\Cloud\Prosperity\round_2\round-2-island-data-bottle\picnic_basket2.xlsx")['mid_price']

# Combine into one DataFrame
df = pd.DataFrame({
    'Croissants': cros,
    'Djembes': djem,
    'Jams': jams,
    'Basket 1': pic1,
    'Basket 2': pic2,
})

# Compute correlation matrix
corr = df.corr()
print(corr)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='magma_r', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Mid Prices")
plt.tight_layout()
plt.show()
