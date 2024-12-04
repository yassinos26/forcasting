document.addEventListener('DOMContentLoaded', () => {
    const predictButton = document.getElementById('predictButton');
    const dashboardDiv = document.getElementById('dashboard');
    const chartDiv = document.getElementById('chart');
    const statsDiv = document.getElementById('stats');
    const loadingSpinner = document.getElementById('loadingSpinner');

    predictButton.addEventListener('click', async () => {
        // Show the loading spinner
        loadingSpinner.style.display = 'block';

        try {
            // Fetch predictions from the API
            const response = await fetch('http://127.0.0.1:5020/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
            const predictions = await response.json();

            if (!Array.isArray(predictions)) {
                throw new Error("Invalid response format. Expected an array of predictions.");
            }

            // Extract data for the plots
            const timestamps = predictions.map(p => new Date(p.Timestamp)); // Parse timestamps
            const values = predictions.map(p => p.Prediction);

            // Group data by date
            const groupedData = timestamps.reduce((acc, timestamp, index) => {
                const date = timestamp.toISOString().split("T")[0]; // Extract the date part (YYYY-MM-DD)
                if (!acc[date]) acc[date] = [];
                acc[date].push({ timestamp, value: values[index] });
                return acc;
            }, {});

            // Calculate daily min and max
            const dailyStats = Object.entries(groupedData).map(([date, entries]) => {
                const minEntry = entries.reduce((min, curr) => (curr.value < min.value ? curr : min), entries[0]);
                const maxEntry = entries.reduce((max, curr) => (curr.value > max.value ? curr : max), entries[0]);
                return {
                    date,
                    min: { value: minEntry.value.toFixed(2), timestamp: minEntry.timestamp.toLocaleString() },
                    max: { value: maxEntry.value.toFixed(2), timestamp: maxEntry.timestamp.toLocaleString() }
                };
            });

            // Select the min and max for the first available day
            const { min, max } = dailyStats[0] || { min: null, max: null };

            // Compute global statistics
            const mean = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2);
            const median = [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)].toFixed(2);

            // Render prediction plot with updated background colors
            Plotly.newPlot(chartDiv, [{
                x: timestamps,
                y: values,
                mode: 'lines+markers',
                line: { color: '#2A9DF4', width: 3 },
                marker: { size: 6, color: '#F4A261' }
            }], {
                title: 'Parking Occupancy Forecast for Tomorrow',
                xaxis: {
                    title: 'Timestamp',
                    type: 'date',
                    color: '#E0E0E0' // Axis text color
                },
                yaxis: {
                    title: 'Percent Occupied (%)',
                    range: [0, 1],
                    color: '#E0E0E0' // Axis text color
                },
                hovermode: 'x unified',
                paper_bgcolor: '#1E1E1E', // Background of the entire chart
                plot_bgcolor: '#2B2F3E', // Background of the plot area
            });

            // Render statistics details
            statsDiv.innerHTML = `
            <h3>Parking Occupancy Insights</h3>
            <div style="text-align: center; margin: 20px 0;">
                <p><b>🅿️ Average Occupancy:</b> Tomorrow, parking is expected to be <b>${(mean * 100).toFixed(0)}%</b> full on average.</p>
                <p><b>🟢 Best Time to Park:</b> The best time to find parking is <b>${min?.timestamp || "Unknown"}</b>, with the lowest occupancy at just <b>${(min?.value * 100 || 0).toFixed(0)}%</b>.</p>
                <p><b>🔴 Busiest Time:</b> Parking is expected to be fullest at <b>${max?.timestamp || "Unknown"}</b>, reaching <b>${(max?.value * 100 || 0).toFixed(0)}%</b> occupancy.</p>
                <p><b>📊 Median Occupancy:</b> Typically, around <b>${(median * 100).toFixed(0)}%</b> of the parking lot will be occupied.</p>
            </div>
            <p style="text-align: center; font-size: 0.9rem; color: #ccc;">Plan your day to avoid peak hours and enjoy a stress-free parking experience!</p>
            `;

            // Show the dashboard
            dashboardDiv.style.display = 'block';
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while fetching predictions. Please check the console for details.');
        } finally {
            // Hide the loading spinner
            loadingSpinner.style.display = 'none';
        }
    });
    // Download Button Click Event
    downloadButton.addEventListener('click', () => {
        window.location.href = 'http://127.0.0.1:5020/download';
    });
});

