import pandas as pd
import os

# Folder where your xlsx files are
directory = '.'

# List of product names (as they appear in the "product" column)
products = ['RAINFOREST_RESIN', 'KELP', 'SQUID_INK',
            "DJEMBES", 'CROISSANTS', 'JAMS', 'PICNIC_BASKET1', 'PICNIC_BASKET2']

# Dict to hold DataFrames for each product
product_data = {product: [] for product in products}

# Loop through all files in the folder
for filename in os.listdir(directory):
    if filename.endswith('.xlsx') and filename.startswith('prices_round'):
        path = os.path.join(directory, filename)
        try:
            df = pd.read_excel(path)

            for product in products:
                filtered = df[df['product'] == product]
                if not filtered.empty:
                    product_data[product].append(filtered)

            print(f"✅ Processed: {filename}")
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

# Save each product’s combined and sorted data
for product, dfs in product_data.items():
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values(by=['day', 'timestamp'])
        out_filename = f"{product.lower()}.xlsx"
        combined_df.to_excel(out_filename, index=False)
        print(f"📦 Saved {out_filename}")
