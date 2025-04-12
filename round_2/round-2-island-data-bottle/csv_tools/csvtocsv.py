import os
import pandas as pd

# Set your directory (change '.' to a specific path if needed)
directory = '.'

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        csv_path = os.path.join(directory, filename)
        xlsx_path = os.path.join(directory, filename.replace('.csv', '.xlsx'))

        # Read CSV with semicolon separator
        try:
            df = pd.read_csv(csv_path, sep=';')
            df.to_excel(xlsx_path, index=False)
            print(f"✅ Converted: {filename} -> {xlsx_path}")
        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")
