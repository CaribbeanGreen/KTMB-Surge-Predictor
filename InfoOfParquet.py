import pandas as pd

# Load the parquet file
df = pd.read_parquet("ets_2026.parquet")

# Show first 5 rows
print(df.head())

# Show column names
print(df.columns)

# Show data types
print(df.dtypes)

# Show number of rows and columns
print(df.shape)
