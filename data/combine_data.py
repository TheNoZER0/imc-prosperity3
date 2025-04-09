import pandas as pd
import os

data_type = input("Hii, Enter 'p' for prices or 't' for trades: ").strip().lower()
round_number = input("Enter the round number (e.g., 1): ").strip()

if data_type not in ['p', 't']:
    print("Invalid input. Use 'p' or 't'.")
    exit()

prefix = "prices" if data_type == "p" else "trades"
id_col = "product" if data_type == "p" else "symbol"
folder = f"round{round_number}"

files = {
    f"{folder}/{prefix}_round_{round_number}_day_0.csv": "d0",
    f"{folder}/{prefix}_round_{round_number}_day_-1.csv": "d-1",
    f"{folder}/{prefix}_round_{round_number}_day_-2.csv": "d-2",
}

dataframes = []

for file, suffix in files.items():
    print(f"Reading {file}...")
    if not os.path.exists(file):
        print(f"File {file} not found, skipping.")
        continue

    try:
        df = pd.read_csv(file, sep=";")
    except Exception as e:
        print(f"Failed to read {file}: {e}")
        continue

    df.columns = df.columns.str.strip().str.lower()

    if 'timestamp' not in df.columns or id_col not in df.columns:
        print(f"Skipping {file} — missing 'timestamp' or '{id_col}' column.")
        continue

    df['key'] = df['timestamp'].astype(str) + "_" + df[id_col]
    df = df.drop_duplicates(subset='key', keep='first')
    df = df.set_index('key')
    df = df.drop(columns=['timestamp', id_col])

    df = df.add_suffix(f"_{suffix}")
    dataframes.append(df)

if dataframes:
    combined_df = pd.concat(dataframes, axis=1)
    combined_df = combined_df.reset_index()
    combined_df[['timestamp', id_col]] = combined_df['key'].str.rsplit("_", n=1, expand=True)
    combined_df = combined_df.drop(columns=['key'])

    output_filename = f"{folder}/combined_{prefix}_round_{round_number}.csv"
    combined_df.to_csv(output_filename, index=False)
    print(f"Combined data saved to '{output_filename}'")
else:
    print("No valid dataframes to combine.")
