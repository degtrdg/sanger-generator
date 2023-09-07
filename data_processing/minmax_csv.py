import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Define the path to the csv files
csv_folder = 'data/traces'

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Create a list to store all data
all_data = []

# Loop over all csv files in the folder
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(csv_folder, filename))
        all_data.append(df)

# Concatenate all data into a single DataFrame
all_data_df = pd.concat(all_data)

# Fit the scaler to the entire dataset
scaler.fit(all_data_df)

# Loop over all csv files in the folder again to transform each file
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(csv_folder, filename))
        # Skip empty dataframes
        if df.empty:
            continue
        
        # Apply Min-Max scaling
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
        
        # Save the scaled data back to the csv file
        df_scaled.to_csv(os.path.join(csv_folder, filename), index=False)