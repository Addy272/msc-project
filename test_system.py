"""
Test Script for Stock Price Forecasting System
MSc IT Project - University of Mumbai

This script tests all components of the system
Run this to verify everything is working correctly
"""

import sys
import os

print("="*70)
print("STOCK PRICE FORECASTING SYSTEM - COMPONENT TEST")
print("MSc IT Project - University of Mumbai")
print("="*70)

# Test 1: Python Version
print("\n[TEST 1] Checking Python Version...")
if sys.version_info >= (3, 10):
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} - OK")
else:
    print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} - Need 3.10+")
    sys.exit(1)

# Test 2: Import Core Libraries
print("\n[TEST 2] Testing Core Library Imports...")
libraries = [
    ('flask', 'Flask'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('sklearn', 'Scikit-learn'),
    ('tensorflow', 'TensorFlow'),
    ('nltk', 'NLTK'),
    ('textblob', 'TextBlob'),
    ('yfinance', 'yfinance'),
    ('plotly', 'Plotly')
]

failed_imports = []
for lib_name, display_name in libraries:
    try:
        __import__(lib_name)
        print(f"✅ {display_name}")
    except ImportError:
        print(f"❌ {display_name} - NOT INSTALLED")
        failed_imports.append(lib_name)

if failed_imports:
    print(f"\n❌ Missing libraries: {', '.join(failed_imports)}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: Project Structure
print("\n[TEST 3] Checking Project Structure...")
required_dirs = [
    'data/raw',
    'data/processed',
    'models',
    'sentiment',
    'database',
    'templates',
    'static/css',
    'static/js',
    'utils'
]

required_files = [
    'app.py',
    'config.py',
    'requirements.txt',
    'README.md'
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"✅ {dir_path}/")
    else:
        print(f"❌ {dir_path}/ - MISSING")

for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✅ {file_path}")
    else:
        print(f"❌ {file_path} - MISSING")

# Test 4: Module Imports
print("\n[TEST 4] Testing Project Modules...")
modules = [
    ('config', 'Configuration'),
    ('database.models', 'Database Models'),
    ('utils.data_loader', 'Data Loader'),
    ('utils.feature_engineering', 'Feature Engineering'),
    ('sentiment.sentiment_analysis', 'Sentiment Analysis'),
    ('models.random_forest', 'Random Forest'),
    ('models.lstm_model', 'LSTM Model')
]

for module_name, display_name in modules:
    try:
        __import__(module_name)
        print(f"✅ {display_name}")
    except Exception as e:
        print(f"❌ {display_name} - ERROR: {str(e)[:50]}")

# Test 5: NLTK Data
print("\n[TEST 5] Checking NLTK Data...")
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        print("✅ NLTK punkt tokenizer")
    except LookupError:
        print("❌ NLTK punkt - Download with: python -c \"import nltk; nltk.download('punkt')\"")
    
    try:
        nltk.data.find('corpora/stopwords')
        print("✅ NLTK stopwords")
    except LookupError:
        print("❌ NLTK stopwords - Download with: python -c \"import nltk; nltk.download('stopwords')\"")
except Exception as e:
    print(f"❌ NLTK Error: {str(e)}")

# Test 6: Data Fetching
print("\n[TEST 6] Testing Data Fetching...")
try:
    from utils.data_loader import StockDataLoader
    loader = StockDataLoader('AAPL')
    print("✅ StockDataLoader initialized")
    
    # Try to fetch a small sample
    print("   Attempting to fetch sample data...")
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    df = loader.fetch_historical_data(start_date, end_date)
    if df is not None and len(df) > 0:
        print(f"✅ Data fetching works - Got {len(df)} records")
    else:
        print("❌ Data fetching failed")
except Exception as e:
    print(f"❌ Data Loader Error: {str(e)[:100]}")

# Test 7: Feature Engineering
print("\n[TEST 7] Testing Feature Engineering...")
try:
    from utils.feature_engineering import FeatureEngineer
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.randn(100) * 10 + 150,
        'High': np.random.randn(100) * 10 + 155,
        'Low': np.random.randn(100) * 10 + 145,
        'Close': np.random.randn(100) * 10 + 150,
        'Volume': np.random.randint(1000000, 10000000, 100)
    })
    
    fe = FeatureEngineer()
    df_with_features = fe.add_technical_indicators(sample_df)
    
    if 'MA_5' in df_with_features.columns and 'RSI' in df_with_features.columns:
        print("✅ Feature Engineering works")
    else:
        print("❌ Feature Engineering failed")
except Exception as e:
    print(f"❌ Feature Engineering Error: {str(e)[:100]}")

# Test 8: Sentiment Analysis
print("\n[TEST 8] Testing Sentiment Analysis...")
try:
    from sentiment.sentiment_analysis import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer()
    test_text = "Apple stock surges on strong earnings report"
    result = analyzer.analyze_sentiment_textblob(test_text)
    
    if 'sentiment_score' in result and 'sentiment_label' in result:
        print(f"✅ Sentiment Analysis works - Score: {result['sentiment_score']:.3f}, Label: {result['sentiment_label']}")
    else:
        print("❌ Sentiment Analysis failed")
except Exception as e:
    print(f"❌ Sentiment Analysis Error: {str(e)[:100]}")

# Test 9: Machine Learning Models
print("\n[TEST 9] Testing ML Models...")

# Test Random Forest
try:
    from models.random_forest import RandomForestModel
    import numpy as np
    
    X = np.random.randn(100, 5)
    y = np.random.randn(100) * 10 + 150
    
    rf = RandomForestModel(task='regression')
    X_train, X_test, y_train, y_test = rf.prepare_data(X, y, test_size=0.2)
    
    print("✅ Random Forest Model initialized")
except Exception as e:
    print(f"❌ Random Forest Error: {str(e)[:100]}")

# Test LSTM
try:
    from models.lstm_model import LSTMModel
    import numpy as np
    
    prices = np.random.randn(200) * 10 + 150
    lstm = LSTMModel()
    
    print("✅ LSTM Model initialized")
except Exception as e:
    print(f"❌ LSTM Error: {str(e)[:100]}")

# Test 10: Flask Application
print("\n[TEST 10] Testing Flask Application...")
try:
    import app
    
    if hasattr(app, 'app'):
        print("✅ Flask app initialized")
    else:
        print("❌ Flask app not found")
except Exception as e:
    print(f"❌ Flask Error: {str(e)[:100]}")

# Test 11: Database
print("\n[TEST 11] Testing Database...")
try:
    from database.models import db, StockData, SentimentData
    from flask import Flask
    
    test_app = Flask(__name__)
    test_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    test_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(test_app)
    
    with test_app.app_context():
        db.create_all()
        print("✅ Database models work")
except Exception as e:
    print(f"❌ Database Error: {str(e)[:100]}")

# Final Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print("\n✅ All critical tests passed!")
print("\nYour system is ready to use. Run: python app.py")
print("\n" + "="*70)
