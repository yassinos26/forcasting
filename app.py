from flask import Flask, request, jsonify, render_template
import torch
import yaml
import joblib
import pandas as pd
import numpy as np
import torch.nn as nn
from flask_cors import CORS

# Initialisation du flask application
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------------------------------
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        out = self.fc(h_n)
        return out


# ---------------------------------------------------------------------------------------------------
# Paths

config_path = r"C:\Users\MSI\Parking Occupancy Forecasting\config.yaml"
tensor_path = r"C:\Users\MSI\Parking Occupancy Forecasting\tensor.pt"
checkpoint_path = r"C:\Users\MSI\Parking Occupancy Forecasting\best.pt"
# Load configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Load model and set to evaluation mode
model = BiLSTMModel(**config["hyperparameters"])
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict)
model.eval()
# Load processed data 

def load_tensor(tensor_path):
    """
    Load the tensor and timestamps from the given file path.
    """
    loaded_data = torch.load(tensor_path)
    return loaded_data['data'], loaded_data['timestamps']

# Helper function to expand timestamps
def expand_timestamps(base_timestamps, periods):
    expanded_timestamps = []
    half_hour = pd.Timedelta(minutes=30)
    for base in base_timestamps:
        expanded_timestamps.extend([base + half_hour * i for i in range(1, periods + 1)])
    return expanded_timestamps

# Route for rendering the template
@app.route('/')
def home():
    """Render the main template."""
    return render_template('index.html')

from flask import send_from_directory

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Run inference
        with torch.no_grad():
            inference_tensor, timestamps = load_tensor(tensor_path)
            predictions = model(inference_tensor).numpy().flatten()

        # Expand timestamps and prepare results
        all_timestamps = expand_timestamps(timestamps, periods=48)
        
        result_df = pd.DataFrame({
            'Timestamp': all_timestamps,
            'Prediction': predictions
        })

        # Sorting the results
        result_df = result_df.sort_values(by='Timestamp').reset_index(drop=True)
        result_df = result_df.groupby('Timestamp')['Prediction'].mean().reset_index()

        # Convert DataFrame to JSON
        result_json = result_df.to_json(orient='records', date_format='iso')
        
        return result_json, 200, {'ContentType':'application/json'}

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5020, host='0.0.0.0', debug=True)
