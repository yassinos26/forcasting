import requests, pandas as pd , plotly.express as px

# API URL
api_url = "http://127.0.0.1:5020/predict"

# Charger les données d'inférence
data_path = r"C:/Users/Yassine/Desktop/wided/final/extracted_data.csv"
df_new = pd.read_csv(data_path)

# Envoyer les données à l'API Flask
response = requests.post(api_url, json=df_new.to_dict(orient='records'))

if response.status_code == 200:
    # Convertir la réponse JSON en DataFrame
    predictions = pd.DataFrame(response.json())

    # Convertir les timestamps en datetime
    predictions['Timestamp'] = pd.to_datetime(predictions['Timestamp'])

    # Tracer le graphe
    fig = px.line(predictions, x='Timestamp', y='Prediction', title='Prédictions de BiLSTM')
    fig.update_xaxes(title='Timestamp')
    fig.update_yaxes(title='Prediction')

    # Afficher le graphe
    fig.show()

else:
    print(f"Erreur dans l'API Flask : {response.json()}")