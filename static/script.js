document.getElementById('predictButton').addEventListener('click', async () => {
    const dashboardDiv = document.getElementById('dashboard');
    const chartDiv = document.getElementById('chart');

    try {

        // Fetch the data from the API
        const response = await fetch('http://127.0.0.1:5020/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(await loadCSVData())
        });

        if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
        const predictions = await response.json();

        // Prepare data for Plotly
        const timestamps = predictions.map(p => p.Timestamp);
        const values = predictions.map(p => p.Prediction);
        
        const plotData = [
        {
            x: timestamps,
            y: values,
            mode: 'lines',
            name: 'Prediction'
        }
        ];

        // Render chart with Plotly
        Plotly.newPlot(chartDiv, plotData, {
        title: 'Prediction Results',
        xaxis: { title: 'Timestamp' },
        yaxis: { title: 'Prediction' }
        });

        console.log('ok')

        // Show dashboard section
        dashboardDiv.style.display = 'block';
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while fetching predictions.');
    }
    });

  // Load CSV data for the API request
    async function loadCSVData() {
    const response = await fetch('extracted_data.csv');
    const csvText = await response.text();
    const rows = csvText.split('\n').map(row => row.split(','));
    const headers = rows[0];
    return rows.slice(1).map(row => Object.fromEntries(row.map((cell, i) => [headers[i], cell])));
    }
