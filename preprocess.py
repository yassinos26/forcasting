import pandas as pd
import numpy as np
import torch
import joblib

def load_scaler(path):
    """
    Load the scaler object from a given file path.
    """
    return joblib.load(path)

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

    X = []
    timestamps = []
    for i in range(len(df_scaled) - input_window + 1):
        X.append(df_scaled.iloc[i:(i + input_window)].values)
        timestamps.append(df_scaled.index[i + input_window - 1])

    return np.array(X), pd.to_datetime(timestamps)

def main():
    """
    Main function to load data, prepare it for inference, and save the processed tensor.
    """
    data_path = r"C:\Users\MSI\Parking Occupancy Forecasting\extracted_data.csv"
    scaler_path = r"C:\Users\MSI\Parking Occupancy Forecasting\scaler"
    output_path = r"C:\Users\MSI\Parking Occupancy Forecasting\tensor.pt"
    window_size = 96

    df = pd.read_csv(data_path)
    scaler = load_scaler(scaler_path)
    processed_data, timestamps = prepare_inference_data(df, scaler, window_size)
    inference_tensor = torch.tensor(processed_data, dtype=torch.float32)
    torch.save({'data': inference_tensor, 'timestamps': timestamps}, output_path)
    print("Saved tensor and timestamps to:", output_path)

if __name__ == "__main__":
    main()
