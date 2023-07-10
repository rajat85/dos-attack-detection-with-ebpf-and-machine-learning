import pandas as pd
import random

FILE_1 = 'valid_few_requests_through_lua_for_tcp_states_07_07.csv'
FILE_2 = 'invalid_few_requests_through_lua_for_tcp_states_07_07.csv'
COMBINED_FILE_NAME = 'combined.csv'
# Read the CSV files
df1 = pd.read_csv(f"data/output/{FILE_1}")
df2 = pd.read_csv(f"data/output/{FILE_2}")

# Combine the dataframes
combined_df = pd.concat([df1, df2], ignore_index=True)

SEED_VALUE = 42
random.seed(SEED_VALUE)
# Shuffle the rows randomly
combined_df = combined_df.sample(frac=1)

# Reset the index
combined_df = combined_df.reset_index(drop=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv(f"data/output/{COMBINED_FILE_NAME}", index=False)

other_tcp_traces = pd.read_csv(f"data/output/{COMBINED_FILE_NAME}",
                               encoding='latin-1', sep=',', keep_default_na=False)
# print(other_tcp_traces.head(10))