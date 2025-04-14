import pandas as pd
import os

# Folder where your xlsx files are
directory = '.'

# Dict to hold DataFrames for each product
product_data = {}

# Loop through all files in the folder
for filename in os.listdir(directory):
    if filename.endswith('.xlsx') and filename.startswith('prices_round'):
        path = os.path.join(directory, filename)
        try:
            # Read the Excel file
            df = pd.read_excel(path)

            # Get unique product names directly from the "product" column
            products = df['product'].unique()

            # Initialize an empty list for each product in the dict if not already there
            for product in products:
                if product not in product_data:
                    product_data[product] = []

            # Filter and append data for each product
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
