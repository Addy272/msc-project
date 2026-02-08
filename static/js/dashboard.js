// Dashboard JavaScript Functions
// MSc IT Project - Stock Price Forecasting System

// Show status message
function showStatus(message, type = 'info') {
    const statusDiv = document.getElementById('statusMessage');
    statusDiv.textContent = message;
    statusDiv.className = `status-message show ${type}`;
    
    setTimeout(() => {
        statusDiv.classList.remove('show');
    }, 5000);
}

// Fetch latest data
async function fetchData() {
    showStatus('Fetching latest data...', 'info');
    
    try {
        const response = await fetch('/api/fetch_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symbol: CURRENT_SYMBOL })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus(`Success! Fetched ${data.stock_records} stock records and ${data.sentiment_records} news articles`, 'success');
            
            // Reload visualizations
            setTimeout(() => {
                loadStockData();
                loadSentimentData();
            }, 1000);
        } else {
            showStatus('Error: ' + (data.error || 'Failed to fetch data'), 'error');
        }
    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
    }
}

// Train models
async function trainModels() {
    showStatus('Training models... This may take a few minutes.', 'info');
    
    try {
        const response = await fetch('/api/train_models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symbol: CURRENT_SYMBOL })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus('Models trained successfully!', 'success');
            
            // Show metrics
            document.getElementById('metricsSection').style.display = 'block';
            displayMetrics(data.rf_metrics, data.lstm_metrics);
            
            // Reload model metrics
            loadModelMetrics();
        } else {
            showStatus('Error: ' + (data.error || 'Failed to train models'), 'error');
        }
    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
    }
}

// Make prediction
async function makePrediction() {
    showStatus('Making prediction...', 'info');
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symbol: CURRENT_SYMBOL })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus('Prediction completed!', 'success');
            displayPredictions(data);
            document.getElementById('predictionSection').style.display = 'block';
        } else {
            showStatus('Error: ' + (data.error || 'Failed to make prediction'), 'error');
        }
    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
    }
}

// Display predictions
function displayPredictions(data) {
    // Current price
    document.getElementById('currentPrice').textContent = `$${data.current_price.toFixed(2)}`;
    
    // Random Forest
    document.getElementById('rfPrediction').textContent = `$${data.predictions.random_forest.toFixed(2)}`;
    const rfSignal = document.getElementById('rfSignal');
    rfSignal.textContent = data.signals.random_forest;
    rfSignal.className = `signal-badge ${data.signals.random_forest.toLowerCase()}`;
    
    // LSTM
    document.getElementById('lstmPrediction').textContent = `$${data.predictions.lstm.toFixed(2)}`;
    const lstmSignal = document.getElementById('lstmSignal');
    lstmSignal.textContent = data.signals.lstm;
    lstmSignal.className = `signal-badge ${data.signals.lstm.toLowerCase()}`;
    
    // Ensemble
    document.getElementById('ensemblePrediction').textContent = `$${data.predictions.ensemble.toFixed(2)}`;
    const ensembleSignal = document.getElementById('ensembleSignal');
    ensembleSignal.textContent = data.signals.ensemble;
    ensembleSignal.className = `signal-badge ${data.signals.ensemble.toLowerCase()}`;
}

// Display metrics
function displayMetrics(rfMetrics, lstmMetrics) {
    if (rfMetrics) {
        document.getElementById('rf-rmse').textContent = rfMetrics.rmse.toFixed(4);
        document.getElementById('rf-mae').textContent = rfMetrics.mae.toFixed(4);
        document.getElementById('rf-r2').textContent = rfMetrics.r2_score.toFixed(4);
    }
    
    if (lstmMetrics) {
        document.getElementById('lstm-rmse').textContent = lstmMetrics.rmse.toFixed(4);
        document.getElementById('lstm-mae').textContent = lstmMetrics.mae.toFixed(4);
        document.getElementById('lstm-r2').textContent = lstmMetrics.r2_score.toFixed(4);
    }
}

// Load stock data for visualization
async function loadStockData() {
    try {
        const response = await fetch(`/api/stock_data/${CURRENT_SYMBOL}`);
        const result = await response.json();
        
        if (result.success && result.data.length > 0) {
            const data = result.data;
            
            // Price chart
            const priceTrace = {
                x: data.map(d => d.date),
                y: data.map(d => d.close),
                type: 'scatter',
                mode: 'lines',
                name: 'Close Price',
                line: { color: '#667eea', width: 2 }
            };
            
            const priceLayout = {
                title: `${CURRENT_SYMBOL} Stock Price`,
                xaxis: { title: 'Date' },
                yaxis: { title: 'Price ($)' },
                hovermode: 'x unified'
            };
            
            Plotly.newPlot('priceChart', [priceTrace], priceLayout);
            
            // Volume chart
            const volumeTrace = {
                x: data.map(d => d.date),
                y: data.map(d => d.volume),
                type: 'bar',
                name: 'Volume',
                marker: { color: '#764ba2' }
            };
            
            const volumeLayout = {
                title: 'Trading Volume',
                xaxis: { title: 'Date' },
                yaxis: { title: 'Volume' },
                hovermode: 'x unified'
            };
            
            Plotly.newPlot('volumeChart', [volumeTrace], volumeLayout);
        }
    } catch (error) {
        console.error('Error loading stock data:', error);
    }
}

// Load sentiment data
async function loadSentimentData() {
    try {
        const response = await fetch(`/api/sentiment_data/${CURRENT_SYMBOL}`);
        const result = await response.json();
        
        if (result.success && result.data.length > 0) {
            const data = result.data;
            
            // Sentiment chart
            const sentimentTrace = {
                x: data.map(d => d.date),
                y: data.map(d => d.sentiment_score),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Sentiment Score',
                line: { color: '#11998e', width: 2 },
                marker: { size: 6 }
            };
            
            const sentimentLayout = {
                title: 'News Sentiment Over Time',
                xaxis: { title: 'Date' },
                yaxis: { title: 'Sentiment Score', range: [-1, 1] },
                shapes: [
                    {
                        type: 'line',
                        x0: data[0].date,
                        x1: data[data.length - 1].date,
                        y0: 0,
                        y1: 0,
                        line: { color: 'gray', width: 1, dash: 'dash' }
                    }
                ],
                hovermode: 'x unified'
            };
            
            Plotly.newPlot('sentimentChart', [sentimentTrace], sentimentLayout);
            
            // Display news list
            const newsList = document.getElementById('newsList');
            newsList.innerHTML = data.slice(0, 10).map(item => `
                <div class="news-item">
                    <h4>${item.headline}</h4>
                    <div class="news-meta">
                        ${item.date} | 
                        Sentiment: <span class="sentiment-${item.sentiment_label.toLowerCase()}">${item.sentiment_label}</span>
                        (Score: ${item.sentiment_score.toFixed(3)})
                    </div>
                </div>
            `).join('');
        } else {
            document.getElementById('sentimentChart').innerHTML = '<p style="text-align:center; padding:40px;">No sentiment data available. Fetch latest data to see sentiment analysis.</p>';
            document.getElementById('newsList').innerHTML = '<p style="text-align:center; padding:20px;">No news articles found.</p>';
        }
    } catch (error) {
        console.error('Error loading sentiment data:', error);
    }
}

// Load model metrics
async function loadModelMetrics() {
    try {
        const response = await fetch(`/api/model_metrics/${CURRENT_SYMBOL}`);
        const result = await response.json();
        
        if (result.success && (result.random_forest || result.lstm)) {
            document.getElementById('metricsSection').style.display = 'block';
            
            if (result.random_forest) {
                document.getElementById('rf-rmse').textContent = result.random_forest.rmse.toFixed(4);
                document.getElementById('rf-mae').textContent = result.random_forest.mae.toFixed(4);
                document.getElementById('rf-r2').textContent = result.random_forest.r2_score.toFixed(4);
            }
            
            if (result.lstm) {
                document.getElementById('lstm-rmse').textContent = result.lstm.rmse.toFixed(4);
                document.getElementById('lstm-mae').textContent = result.lstm.mae.toFixed(4);
                document.getElementById('lstm-r2').textContent = result.lstm.r2_score.toFixed(4);
            }
        }
    } catch (error) {
        console.error('Error loading model metrics:', error);
    }
}
