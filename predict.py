# Import necessary libraries
import os
import yaml
import pandas as pd
import torch
import numpy as np
import joblib
import warnings
from model.model import BiLSTMModel
from preprocessing.preprocess import prepare_inference_data
warnings.filterwarnings("ignore")

# Path definitions
config_path = r"C:\Users\MSI\model\config.yml"
data_path = r"C:\Users\MSI\model\extracted_data.csv"
scaler_path = r"C:\Users\MSI\model\preprocessing\scaler"
checkpoint_path = r"C:\Users\MSI\model\model\checkpoints\best_model .pth"

# Load configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Load data
df_new = pd.read_csv(data_path)

# Load and verify scaler
scaler = joblib.load(scaler_path)
if hasattr(scaler, 'scale_'):
    print("Scaler is fitted and ready to use.")
else:
    print("Scaler is not fitted. Something went wrong during loading.")

# Data preprocessing
window_size = 96
processed_data, timestamps = prepare_inference_data(df_new, scaler, window_size)
inference_tensor = torch.tensor(processed_data, dtype=torch.float32)

# Load model and set to evaluation mode
model = BiLSTMModel(**config["hyperparameters"])
print("Loading model from:", checkpoint_path)
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully and set to evaluation mode.")

# Inference
with torch.no_grad():
    predictions = model(inference_tensor)

# Function to expand timestamps for prediction output
def expand_timestamps(base_timestamps, periods):
    expanded_timestamps = []
    half_hour = pd.Timedelta(minutes=30)
    for base in base_timestamps:
        expanded_timestamps.extend([base + half_hour * i for i in range(1, periods + 1)])
    return expanded_timestamps

# Prepare results
all_timestamps = expand_timestamps(timestamps, periods=48)
predictions = predictions.numpy().flatten()
result_df = pd.DataFrame({
    'Timestamp': all_timestamps,
    'Prediction': predictions
})

# Sorting the results
result_df = result_df.sort_values(by='Timestamp').reset_index(drop=True)
result_df = result_df.groupby('Timestamp')['Prediction'].mean().reset_index()

# Optionally save the predictions to a CSV file
result_df.to_csv('result_df.csv', index=False)
result_df.to_json('predictions.json', orient='records', date_format='iso', lines=True)
