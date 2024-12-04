import pandas as pd
import numpy as np
import torch

# ---------------------------------------------------------------------------------------------------
# Scaler configuration
# ---------------------------------------------------------------------------------------------------
def prepare_inference_data(df, scaler, input_window):
    """
    Prepare raw incoming data for inference, aligning with the training data preparation steps.
    """
    df['Date_Time_HalfHour'] = pd.to_datetime(df['Date'] + ' ' + df['Time_HalfHour'])
    df.set_index('Date_Time_HalfHour', inplace=True)

    df_agg = df.groupby(df.index).agg({
        'Occupancy': 'sum',
        'Capacity': 'first',
        'DayOfWeek': 'first'
    })
    df_agg['PercentOccupied'] = df_agg['Occupancy'] / df_agg['Capacity']

    df_agg['lag_1'] = df_agg['PercentOccupied'].shift(1)
    df_agg['lag_2'] = df_agg['PercentOccupied'].shift(2)
    df_agg['lag_3'] = df_agg['PercentOccupied'].shift(3)
    df_agg['rolling_mean'] = df_agg['PercentOccupied'].rolling(window=24).mean()
    df_agg['rolling_std'] = df_agg['PercentOccupied'].rolling(window=24).std()

    df_agg.bfill(inplace=True)
    df_agg.ffill(inplace=True)

    features = ['DayOfWeek', 'PercentOccupied', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'rolling_std']
    scaled_features = scaler.transform(df_agg[features])
    df_scaled = pd.DataFrame(scaled_features, index=df_agg.index, columns=features)
    
    # Sauvegarde du DataFrame normalisé dans un fichier .csv
    df_scaled.to_csv(output_csv_path)
    print(f"Data prepared and saved to {output_csv_path}")
    
    X = []
    timestamps = []
    for i in range(len(df_scaled) - input_window + 1):
        X.append(df_scaled.iloc[i:(i + input_window)].values)
        timestamps.append(df_scaled.index[i + input_window - 1])

    return np.array(X), pd.to_datetime(timestamps)

def predict_from_csv(model, csv_path, input_window, device):
    """
    Load prepared data from CSV and make predictions using the model.
    """
    # Chargement des données préparées
    df_scaled = pd.read_csv(csv_path, index_col=0)

    # Création des séquences pour le modèle
    X = []
    for i in range(len(df_scaled) - input_window + 1):
        X.append(df_scaled.iloc[i:(i + input_window)].values)
    X = np.array(X)

    # Conversion en tenseur PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Prédiction avec le modèle
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor)

    return predictions.cpu().numpy()

# Exemple d'utilisation :
output_csv_path = "prepared_data.csv"
X, timestamps = prepare_inference_data(df, scaler, input_window=24, output_csv_path=output_csv_path)

# Charger le modèle et effectuer des prédictions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMModel(input_size=7, hidden_size=32, output_size=1, num_layers=2).to(device)
model.load_state_dict(torch.load("model.pth"))  # Charger le modèle entraîné

predictions = predict_from_csv(model, output_csv_path, input_window=24, device=device)
print(predictions)
