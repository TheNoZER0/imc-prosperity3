import pandas as pd

# --- Process the Trades Data ---

# Read each trades file and add a corresponding 'day' column.
trades_day_minus2 = pd.read_csv('data/round1/trades_round_1_day_-2.csv', delimiter=';')
trades_day_minus2['day'] = -2

trades_day_minus1 = pd.read_csv('data/round1/trades_round_1_day_-1.csv', delimiter=';')
trades_day_minus1['day'] = -1

trades_day_0 = pd.read_csv('data/round1/trades_round_1_day_0.csv', delimiter=';')
trades_day_0['day'] = 0


trades_df = pd.concat([trades_day_minus2, trades_day_minus1, trades_day_0], ignore_index=True)


trades_df = trades_df.drop(columns=['buyer', 'seller'])

trades_order = ['day', 'timestamp'] + [col for col in trades_df.columns if col not in ['day', 'timestamp']]
trades_df = trades_df[trades_order]


trades_df.to_csv('trades_combined.csv', index=False, sep=';')


# --- Process the Prices Data ---

df_day_minus2 = pd.read_csv('data/round1/prices_round_1_day_-2.csv', delimiter=';')
df_day_minus1 = pd.read_csv('data/round1/prices_round_1_day_-1.csv', delimiter=';')
df_day_0 = pd.read_csv('data/round1/prices_round_1_day_0.csv', delimiter=';')

# Combine the prices data vertically.
prices_df = pd.concat([df_day_minus2, df_day_minus1, df_day_0], ignore_index=True)
prices_df = prices_df.sort_values(by=['day', 'timestamp']).reset_index(drop=True)

prices_order = ['day', 'timestamp'] + [col for col in prices_df.columns if col not in ['day', 'timestamp']]
prices_df = prices_df[prices_order]


prices_df.to_csv('prices_combined.csv', index=False, sep=';')


print(prices_df.head())
