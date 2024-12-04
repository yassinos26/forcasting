import requests
import pandas as pd
import plotly.express as px

# API URL
api_url = "http://127.0.0.1:5020/predict"

# Send a request to the Flask API
response = requests.post(api_url)

if response.status_code == 200:
    # Convert JSON response into DataFrame
    predictions = pd.DataFrame(response.json())

    # Convert timestamps to datetime
    predictions['Timestamp'] = pd.to_datetime(predictions['Timestamp'])

    # Plot the graph
    fig = px.line(predictions, x='Timestamp', y='Prediction', title='Prédictions de BiLSTM')
    fig.update_xaxes(title='Timestamp')
    fig.update_yaxes(title='Prediction')

    # Display the graph
    fig.show()

else:
    print(f"Erreur dans l'API Flask : {response.status_code} - {response.text}")
