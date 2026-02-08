# Quick Start Guide
## Stock Price Forecasting System

This guide will help you get the system running in under 10 minutes.

---

## ⚡ Quick Setup (5 Steps)

### Step 1: Install Python Dependencies
```bash
cd stock_prediction_project
pip install -r requirements.txt
```

### Step 2: Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 3: Run the Application
```bash
python app.py
```

### Step 4: Open Browser
Navigate to: **http://localhost:5000**

### Step 5: Use the System
1. Click on a stock symbol (e.g., AAPL)
2. Click "📥 Fetch Latest Data"
3. Wait 30 seconds
4. Click "🎓 Train Models"
5. Wait 2-3 minutes
6. Click "🔮 Make Prediction"
7. View results!

---

## 🎯 First Time Usage

### What You'll See:

#### Home Page
- Welcome message
- Feature descriptions
- Stock selection buttons
- How it works section

#### Dashboard (After Selecting Stock)
- Control panel with 3 buttons
- Empty charts (until data is fetched)
- Status messages

---

## 📝 Step-by-Step First Run

### 1. Fetch Data (Required First Step)
```
Click: "📥 Fetch Latest Data"
⏱️ Takes: ~30 seconds
✅ Success: "Fetched X stock records and Y news articles"
```

**What Happens:**
- Downloads 2 years of stock price history
- Fetches recent financial news
- Performs sentiment analysis
- Saves to database

### 2. Train Models (Required Second Step)
```
Click: "🎓 Train Models"
⏱️ Takes: 2-5 minutes
✅ Success: "Models trained successfully!"
```

**What Happens:**
- Creates technical indicators
- Merges sentiment data
- Trains Random Forest model
- Trains LSTM neural network
- Saves models to disk
- Shows performance metrics

### 3. Make Predictions (Final Step)
```
Click: "🔮 Make Prediction"
⏱️ Takes: ~5 seconds
✅ Success: "Prediction completed!"
```

**What Happens:**
- Loads trained models
- Processes latest data
- Makes price predictions
- Generates trading signals
- Displays results

---

## 📊 Understanding Your Results

### Prediction Results Card

```
┌─────────────────────────────────┐
│ Current Price:     $150.25      │
│                                 │
│ Random Forest:     $152.80      │
│ Signal:            BUY 🟢       │
│                                 │
│ LSTM:              $151.50      │
│ Signal:            BUY 🟢       │
│                                 │
│ ENSEMBLE:          $152.15      │
│ Signal:            BUY 🟢       │
└─────────────────────────────────┘
```

**Interpretation:**
- If all models say **BUY**: Strong bullish signal
- If all models say **SELL**: Strong bearish signal
- If mixed signals: Market uncertainty, consider **HOLD**
- **Ensemble** is recommended (combines both models)

---

## 🎨 Visualizations

### 1. Stock Price History
- Blue line showing price trends
- Interactive: Hover for values
- Zoom: Click and drag
- Pan: Hold shift and drag

### 2. Trading Volume
- Purple bars showing daily volume
- Higher bars = More trading activity
- Useful for confirming price movements

### 3. Sentiment Analysis
- Green line showing news sentiment
- Above zero = Positive sentiment
- Below zero = Negative sentiment
- Correlates with price movements

---

## ⚡ Quick Tips

### For Best Results:
1. ✅ Use stocks with good news coverage (AAPL, GOOGL, TSLA)
2. ✅ Train models with at least 1 year of data
3. ✅ Compare predictions with actual market trends
4. ✅ Use ensemble prediction for final decision

### Common Mistakes to Avoid:
1. ❌ Making predictions before training models
2. ❌ Not fetching data first
3. ❌ Using predictions for real trading
4. ❌ Ignoring model performance metrics

---

## 🔍 Model Performance Metrics

### What They Mean:

**RMSE (Root Mean Squared Error)**
- Lower is better
- Measures average prediction error
- Good: < $5
- Excellent: < $2

**MAE (Mean Absolute Error)**
- Average absolute error
- More intuitive than RMSE
- Good: < $3
- Excellent: < $1.5

**R² Score (Coefficient of Determination)**
- Ranges from 0 to 1
- Higher is better
- Good: > 0.85
- Excellent: > 0.95

---

## 🎓 Demo Workflow

### Complete Demo (5 Minutes)

1. **Start Application**
   ```bash
   python app.py
   ```

2. **Open Browser**
   ```
   http://localhost:5000
   ```

3. **Select AAPL**
   ```
   Click "AAPL" button on home page
   ```

4. **Fetch Data**
   ```
   Dashboard → "📥 Fetch Latest Data" → Wait
   Status: "Fetched 500 stock records..."
   ```

5. **View Charts**
   ```
   Scroll down to see:
   - Stock price history
   - Trading volume
   - Sentiment analysis (if news available)
   ```

6. **Train Models**
   ```
   "🎓 Train Models" → Wait 2-3 minutes
   Status: "Models trained successfully!"
   Metrics appear showing RMSE, MAE, R²
   ```

7. **Make Prediction**
   ```
   "🔮 Make Prediction" → Wait 5 seconds
   See prediction cards with signals
   ```

8. **Analyze Results**
   ```
   - Check current price
   - Compare with predictions
   - Note trading signals
   - Review model confidence
   ```

---

## 🔧 Troubleshooting Quick Fixes

### Problem: "No module named 'flask'"
**Solution:**
```bash
pip install flask
```

### Problem: "Database not found"
**Solution:**
```bash
# Just run the app, it creates automatically
python app.py
```

### Problem: "Model not trained"
**Solution:**
```bash
# In dashboard, click "Train Models" first
```

### Problem: "Port 5000 already in use"
**Solution:**
Edit `app.py`, change last line to:
```python
app.run(debug=True, port=5001)
```

---

## 📱 Testing Different Stocks

Try these stocks for different behaviors:

**Tech Stocks** (High Volatility)
- TSLA (Tesla)
- NVDA (NVIDIA)
- META (Meta)

**Stable Stocks** (Low Volatility)
- JPM (JPMorgan)
- V (Visa)
- WMT (Walmart)

**Popular Stocks** (Good News Coverage)
- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)

---

## ⏱️ Time Estimates

| Task | Time Required |
|------|--------------|
| Installation | 2-3 minutes |
| First Run Setup | 1 minute |
| Fetch Data | 30 seconds |
| Train Models | 2-5 minutes |
| Make Prediction | 5 seconds |
| **Total First Use** | **6-10 minutes** |

---

## 🎯 Success Checklist

After following this guide, you should see:

- ✅ Application running on localhost:5000
- ✅ Home page with stock selection
- ✅ Dashboard with controls
- ✅ Charts showing data
- ✅ Model metrics displayed
- ✅ Prediction cards with signals
- ✅ News sentiment (if available)

---

## 📞 Need Help?

1. Check main README.md for detailed documentation
2. Review error messages in terminal
3. Verify all steps completed in order
4. Make sure internet connection is active
5. Confirm Python 3.10+ is installed

---

## 🎊 You're Ready!

Congratulations! You now have a working stock price forecasting system.

**Next Steps:**
1. Try different stocks
2. Compare model predictions
3. Analyze sentiment trends
4. Review model performance
5. Present for your viva! 🎓

---

**Remember: This is for academic purposes only. Do not use for real trading!**

---

Happy Forecasting! 📈🚀
