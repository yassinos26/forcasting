from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import yaml
import pandas as pd
import torch.nn as nn
from flask_cors import CORS
import os

# Initialisation de l'application Flask
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
config_path = 'config.yaml'
tensor_path = 'tensor.pt'
checkpoint_path = 'best.pt'
output_csv_path = 'predictions.csv'  # Chemin pour sauvegarder le fichier CSV

# Charger la configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Charger le modèle et le mettre en mode évaluation
model = BiLSTMModel(**config["hyperparameters"])
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict)
model.eval()

# Charger les données traitées
def load_tensor(tensor_path):
    """
    Charger le tenseur et les timestamps depuis le chemin donné.
    """
    loaded_data = torch.load(tensor_path)
    return loaded_data['data'], loaded_data['timestamps']

# Fonction utilitaire pour étendre les timestamps
def expand_timestamps(base_timestamps, periods):
    expanded_timestamps = []
    half_hour = pd.Timedelta(minutes=30)
    for base in base_timestamps:
        expanded_timestamps.extend([base + half_hour * i for i in range(1, periods + 1)])
    return expanded_timestamps

# Route pour afficher la page d'accueil
@app.route('/')
def home():
    """Rendre le template principal."""
    return render_template('index.html')

# Route pour les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Effectuer l'inférence
        with torch.no_grad():
            inference_tensor, timestamps = load_tensor(tensor_path)
            predictions = model(inference_tensor).numpy().flatten()

        # Étendre les timestamps et préparer les résultats
        all_timestamps = expand_timestamps(timestamps, periods=48)
        
        result_df = pd.DataFrame({
            'Timestamp': all_timestamps,
            'Prediction': predictions
        })

        # Trier les résultats
        result_df = result_df.sort_values(by='Timestamp').reset_index(drop=True)
        result_df = result_df.groupby('Timestamp')['Prediction'].mean().reset_index()

        # Sauvegarder les résultats en CSV
        result_df.to_csv(output_csv_path, index=False)

        # Convertir DataFrame en JSON pour la réponse
        result_json = result_df.to_json(orient='records', date_format='iso')
        return result_json, 200, {'ContentType': 'application/json'}

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route pour télécharger le fichier CSV
@app.route('/download', methods=['GET'])
def download_csv():
    try:
        # Vérifier si le fichier existe
        if not os.path.exists(output_csv_path):
            return jsonify({"error": "Le fichier des prédictions n'existe pas."}), 404
        
        # Envoyer le fichier CSV
        return send_from_directory(
            directory=os.path.dirname(output_csv_path),
            path=os.path.basename(output_csv_path),
            as_attachment=True
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5020, host='0.0.0.0', debug=True)
