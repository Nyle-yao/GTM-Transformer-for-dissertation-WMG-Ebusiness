import torch

state_dict = torch.load("./results/Run1_12.pth")
print(state_dict)

# with open("/home/hfut3304/data/workspace/SXS/GTM-Transformer-main/wandb/run-20240926_100806-u3k1ezk5/files/output.log","r")as file:
#     logs = file.readlines()
#     print(logs)
#
#
#
# with open("/home/hfut3304/data/workspace/SXS/GTM-Transformer-main/wandb/offline-run-20240926_103347-fymgx6lw/files/output.log","r")as file:
#     logs = file.readlines()
#     print(logs)
#
# with open("/home/hfut3304/data/workspace/SXS/GTM-Transformer-main/wandb/offline-run-20240926_103347-fymgx6lw/files/output.log","r")as file:
#     logs = file.readlines()
#     print(logs)
#
# with open("/home/hfut3304/data/workspace/SXS/GTM-Transformer-main/wandb/run-20240926_100806-u3k1ezk5/logs/debug.log","r")as file:
#     logs = file.readlines()
#     print(logs)


import torch
import pandas as pd

# Step 1: Load the .pth file
pth_file_path = 'results/Run1_12.pth'  # Replace with the correct path if needed
data = torch.load(pth_file_path)

# Step 2: Extract the data
results = data['results']  # Forecasted values
gts = data['gts']  # Ground truth values
codes = data['codes']  # Item codes

# Step 3: Convert to a pandas DataFrame
# Assuming the results and gts arrays have the same shape
df = pd.DataFrame({
    'Item Code': codes,
    'Forecasted': [list(forecast) for forecast in results],  # Convert arrays to lists for CSV
    'Ground Truth': [list(gt) for gt in gts]  # Convert arrays to lists for CSV
})

# Step 4: Save to a CSV file
csv_file_path = 'results/Run1_12.csv'  # Replace with your desired CSV file path
df.to_csv(csv_file_path, index=False)

print(f"Data saved to {csv_file_path}")

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


# Function to calculate WAPE
def calculate_wape(gt, forecast):
    return 100 * np.sum(np.abs(gt - forecast)) / np.sum(gt)


# Step 1: Load the .pth file
pth_file_path = 'results/Run1_12.pth'  # Replace with the correct path if needed
data = torch.load(pth_file_path)

# Step 2: Extract the data
results = data['results']  # Forecasted values (numpy array)
gts = data['gts']  # Ground truth values (numpy array)
codes = data['codes']  # Item codes (list of strings)

# Step 3: Initialize lists for metrics
mae_list = []
wape_list = []

# Step 4: Calculate MAE and WAPE for each item
for i in range(len(results)):
    forecast = results[i]
    gt = gts[i]

    # Calculate MAE and WAPE for this item
    mae = mean_absolute_error(gt, forecast)
    wape = calculate_wape(gt, forecast)

    # Append metrics to lists
    mae_list.append(mae)
    wape_list.append(wape)

# Step 5: Convert to a pandas DataFrame
# Assuming results and gts have the same shape
df = pd.DataFrame({
    'Item Code': codes,
    'Forecasted': [list(forecast) for forecast in results],  # Convert arrays to lists for CSV
    'Ground Truth': [list(gt) for gt in gts],  # Convert arrays to lists for CSV
    'MAE': mae_list,
    'WAPE': wape_list
})

# Step 6: Save to a CSV file
csv_file_path = 'results/Run1_12_with_metrics.csv'  # Replace with your desired CSV file path
df.to_csv(csv_file_path, index=False)

print(f"Data with metrics saved to {csv_file_path}")