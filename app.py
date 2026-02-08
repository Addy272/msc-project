"""
Flask Web Application for Stock Price Forecasting
MSc IT Project - University of Mumbai

Main application file that handles routes and API endpoints
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Import project modules
from config import Config
from database.models import db, init_db, StockData, SentimentData, PredictionData, ModelMetrics
from utils.data_loader import StockDataLoader, NewsDataLoader, get_company_name
from utils.feature_engineering import FeatureEngineer
from sentiment.sentiment_analysis import SentimentAnalyzer
from models.random_forest import RandomForestModel, generate_trading_signals
from models.lstm_model import LSTMModel

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
init_db(app)

# Global variables for models
rf_model = None
lstm_model = None


def _safe_next_url(next_url):
    """Allow only relative next URLs to prevent open redirects."""
    if not next_url:
        return None
    if next_url.startswith('/'):
        return next_url
    return None


def admin_required(view_func):
    """Decorator to protect admin routes."""
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login', next=request.path))
        return view_func(*args, **kwargs)
    return wrapper


def _is_api_request():
    return request.path.startswith('/api/')


@app.before_request
def require_login():
    """Enforce login for all routes except login and static assets."""
    if request.endpoint is None:
        return None

    if request.endpoint == 'static':
        return None

    if request.endpoint in {'admin_login'}:
        return None

    if not session.get('logged_in'):
        if _is_api_request():
            return jsonify({'error': 'Authentication required'}), 401
        return redirect(url_for('admin_login', next=request.path))

    return None


@app.route('/')
def index():
    """Home page"""
    # Prefer symbols that actually exist in the database
    db_symbols = [row[0] for row in db.session.query(StockData.symbol).distinct().all()]
    symbols = sorted(db_symbols) if db_symbols else Config.STOCK_SYMBOLS

    # Overall system stats
    total_stock = StockData.query.count()
    total_sentiment = SentimentData.query.count()
    total_metrics = ModelMetrics.query.count()

    latest_stock_record = StockData.query.order_by(StockData.date.desc()).first()
    latest_training = ModelMetrics.query.order_by(ModelMetrics.training_date.desc()).first()

    # Per-symbol summary
    stock_summary = []
    for symbol in symbols:
        stock_count = StockData.query.filter_by(symbol=symbol).count()
        sentiment_count = SentimentData.query.filter_by(symbol=symbol).count()
        latest_stock = StockData.query.filter_by(symbol=symbol).order_by(StockData.date.desc()).first()
        latest_metric = ModelMetrics.query.filter_by(symbol=symbol).order_by(ModelMetrics.training_date.desc()).first()

        stock_summary.append({
            'symbol': symbol,
            'stock_count': stock_count,
            'sentiment_count': sentiment_count,
            'latest_date': latest_stock.date.strftime('%Y-%m-%d') if latest_stock else None,
            'latest_training': latest_metric.training_date.strftime('%Y-%m-%d %H:%M:%S') if latest_metric else None
        })

    return render_template(
        'index.html',
        stocks=symbols,
        stock_summary=stock_summary,
        total_stock=total_stock,
        total_sentiment=total_sentiment,
        total_metrics=total_metrics,
        latest_stock_date=latest_stock_record.date.strftime('%Y-%m-%d') if latest_stock_record else None,
        latest_training_date=latest_training.training_date.strftime('%Y-%m-%d %H:%M:%S') if latest_training else None
    )


@app.route('/dashboard')
def dashboard():
    """Main dashboard page"""
    symbol = request.args.get('symbol', Config.DEFAULT_STOCK_SYMBOL)
    return render_template('dashboard.html', symbol=symbol, stocks=Config.STOCK_SYMBOLS)


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    if session.get('logged_in'):
        next_url = _safe_next_url(request.args.get('next'))
        return redirect(next_url or url_for('index'))

    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if username == Config.ADMIN_USERNAME and password == Config.ADMIN_PASSWORD:
            session['logged_in'] = True
            session['admin_logged_in'] = True
            session['admin_username'] = username
            next_url = _safe_next_url(request.args.get('next'))
            return redirect(next_url or url_for('index'))

        error = 'Invalid username or password.'

    return render_template('admin_login.html', error=error)


@app.route('/admin/logout')
@admin_required
def admin_logout():
    """Admin logout"""
    session.pop('logged_in', None)
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    return redirect(url_for('admin_login'))


@app.route('/admin')
@admin_required
def admin_dashboard():
    """Admin dashboard page"""
    symbol = request.args.get('symbol', Config.DEFAULT_STOCK_SYMBOL)

    latest_overall_metrics = ModelMetrics.query.order_by(ModelMetrics.training_date.desc()).first()
    latest_rf = ModelMetrics.query.filter_by(
        symbol=symbol, model_type='RandomForest'
    ).order_by(ModelMetrics.training_date.desc()).first()
    latest_lstm = ModelMetrics.query.filter_by(
        symbol=symbol, model_type='LSTM'
    ).order_by(ModelMetrics.training_date.desc()).first()

    stock_count = StockData.query.filter_by(symbol=symbol).count()
    sentiment_count = SentimentData.query.filter_by(symbol=symbol).count()
    metrics_count = ModelMetrics.query.filter_by(symbol=symbol).count()

    total_stock = StockData.query.count()
    total_sentiment = SentimentData.query.count()
    total_metrics = ModelMetrics.query.count()

    news_key_set = bool(Config.NEWS_API_KEY and Config.NEWS_API_KEY != '658ab30b6eed4f879c8409f469d43689')
    rf_model_exists = os.path.exists(Config.RF_MODEL_PATH)
    rf_scaler_exists = os.path.exists(Config.RF_MODEL_PATH.replace('.pkl', '_scaler.pkl'))
    lstm_model_exists = os.path.exists(Config.LSTM_MODEL_PATH)
    lstm_scaler_exists = os.path.exists(Config.LSTM_MODEL_PATH.replace('.h5', '_scaler.pkl'))

    message = request.args.get('message')

    return render_template(
        'admin_dashboard.html',
        symbol=symbol,
        stocks=Config.STOCK_SYMBOLS,
        stock_count=stock_count,
        sentiment_count=sentiment_count,
        metrics_count=metrics_count,
        total_stock=total_stock,
        total_sentiment=total_sentiment,
        total_metrics=total_metrics,
        latest_rf=latest_rf,
        latest_lstm=latest_lstm,
        latest_overall_metrics=latest_overall_metrics,
        news_key_set=news_key_set,
        rf_model_exists=rf_model_exists,
        rf_scaler_exists=rf_scaler_exists,
        lstm_model_exists=lstm_model_exists,
        lstm_scaler_exists=lstm_scaler_exists,
        message=message
    )


@app.route('/admin/data/clear', methods=['POST'])
@admin_required
def admin_clear_data():
    """Clear admin-selected data for a symbol"""
    symbol = request.form.get('symbol', Config.DEFAULT_STOCK_SYMBOL)
    action = request.form.get('action', '')

    if action not in {'stock', 'sentiment', 'metrics', 'all'}:
        return redirect(url_for('admin_dashboard', symbol=symbol, message='Invalid action.'))

    if action in {'stock', 'all'}:
        StockData.query.filter_by(symbol=symbol).delete(synchronize_session=False)
    if action in {'sentiment', 'all'}:
        SentimentData.query.filter_by(symbol=symbol).delete(synchronize_session=False)
    if action in {'metrics', 'all'}:
        ModelMetrics.query.filter_by(symbol=symbol).delete(synchronize_session=False)

    db.session.commit()

    if action == 'all':
        message = f'Cleared stock, sentiment, and metrics data for {symbol}.'
    elif action == 'stock':
        message = f'Cleared stock data for {symbol}.'
    elif action == 'sentiment':
        message = f'Cleared sentiment data for {symbol}.'
    else:
        message = f'Cleared metrics data for {symbol}.'

    return redirect(url_for('admin_dashboard', symbol=symbol, message=message))


@app.route('/api/fetch_data', methods=['POST'])
def fetch_data():
    """
    Fetch and store stock and news data
    """
    try:
        data = request.get_json()
        symbol = data.get('symbol', Config.DEFAULT_STOCK_SYMBOL)
        
        print(f"\nFetching data for {symbol}...")
        
        # Fetch stock data
        stock_loader = StockDataLoader(symbol)
        stock_df = stock_loader.fetch_historical_data()
        
        if stock_df is None:
            return jsonify({'error': 'Failed to fetch stock data'}), 400
        
        # Fetch news data
        news_loader = NewsDataLoader()
        company_name = get_company_name(symbol)
        news_df = news_loader.fetch_news(symbol, company_name, days_back=30)
        
        # Perform sentiment analysis
        sentiment_analyzer = SentimentAnalyzer()
        if news_df is not None and len(news_df) > 0:
            sentiment_df = sentiment_analyzer.analyze_news_dataframe(news_df)
        else:
            sentiment_df = None
        
        # Save to database
        with app.app_context():
            # Clear existing data for this symbol
            StockData.query.filter_by(symbol=symbol).delete()
            SentimentData.query.filter_by(symbol=symbol).delete()
            
            # Save stock data
            for idx, row in stock_df.iterrows():
                stock_record = StockData(
                    symbol=symbol,
                    date=row['Date'].date(),
                    open_price=row['Open'],
                    high_price=row['High'],
                    low_price=row['Low'],
                    close_price=row['Close'],
                    volume=int(row['Volume']),
                    adj_close=row['Adj_Close']
                )
                db.session.add(stock_record)
            
            # Save sentiment data
            if sentiment_df is not None:
                for idx, row in sentiment_df.iterrows():
                    sentiment_record = SentimentData(
                        symbol=symbol,
                        date=row['date'].date(),
                        headline=row['headline'],
                        source=row['source'],
                        url=row.get('url', ''),
                        sentiment_score=row['sentiment_score'],
                        sentiment_label=row['sentiment_label'],
                        polarity=row['polarity'],
                        subjectivity=row['subjectivity']
                    )
                    db.session.add(sentiment_record)
            
            db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Data fetched successfully for {symbol}',
            'stock_records': len(stock_df),
            'sentiment_records': len(sentiment_df) if sentiment_df is not None else 0
        })
        
    except Exception as e:
        print(f"Error in fetch_data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/train_models', methods=['POST'])
def train_models():
    """
    Train both Random Forest and LSTM models
    """
    global rf_model, lstm_model
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', Config.DEFAULT_STOCK_SYMBOL)
        
        print(f"\nTraining models for {symbol}...")
        
        # Load data from database
        with app.app_context():
            stock_records = StockData.query.filter_by(symbol=symbol).order_by(StockData.date).all()
            sentiment_records = SentimentData.query.filter_by(symbol=symbol).order_by(SentimentData.date).all()
        
        if not stock_records:
            return jsonify({'error': 'No stock data found. Please fetch data first.'}), 400
        
        # Convert to DataFrames
        stock_df = pd.DataFrame([{
            'Date': r.date,
            'Open': r.open_price,
            'High': r.high_price,
            'Low': r.low_price,
            'Close': r.close_price,
            'Volume': r.volume,
            'Adj_Close': r.adj_close
        } for r in stock_records])
        
        if sentiment_records:
            sentiment_df = pd.DataFrame([{
                'date': r.date,
                'sentiment_score': r.sentiment_score,
                'polarity': r.polarity,
                'subjectivity': r.subjectivity
            } for r in sentiment_records])
        else:
            sentiment_df = pd.DataFrame()
        
        # Feature engineering
        fe = FeatureEngineer()
        stock_df = fe.add_technical_indicators(stock_df)
        stock_df = fe.add_target_variable(stock_df)
        
        # Merge sentiment if available
        if not sentiment_df.empty:
            stock_df = fe.merge_sentiment(stock_df, sentiment_df)
        else:
            stock_df['sentiment_score'] = 0
            stock_df['polarity'] = 0
            stock_df['subjectivity'] = 0.5
        
        # Prepare features
        X, y, feature_names = fe.prepare_ml_features(stock_df)
        
        # Train Random Forest
        print("\n" + "="*50)
        print("Training Random Forest Model...")
        rf_model = RandomForestModel(task='regression')
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = rf_model.prepare_data(X, y, test_size=0.2)
        rf_model.train(X_train_rf, y_train_rf)
        rf_metrics = rf_model.evaluate(X_test_rf, y_test_rf)
        rf_model.save_model()
        
        # Train LSTM
        print("\n" + "="*50)
        print("Training LSTM Model...")
        lstm_model = LSTMModel()
        prices = stock_df['Close'].values
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = lstm_model.prepare_data(prices, test_size=0.2)
        lstm_model.train(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm)
        lstm_metrics = lstm_model.evaluate(X_test_lstm, y_test_lstm)
        lstm_model.save_model()
        
        # Save metrics to database
        with app.app_context():
            # Random Forest metrics
            rf_metric_record = ModelMetrics(
                model_type='RandomForest',
                symbol=symbol,
                rmse=rf_metrics['rmse'],
                mae=rf_metrics['mae'],
                r2_score=rf_metrics['r2_score'],
                train_size=len(X_train_rf),
                test_size=len(X_test_rf),
                parameters=json.dumps(Config.RANDOM_FOREST_PARAMS)
            )
            db.session.add(rf_metric_record)
            
            # LSTM metrics
            lstm_metric_record = ModelMetrics(
                model_type='LSTM',
                symbol=symbol,
                rmse=lstm_metrics['rmse'],
                mae=lstm_metrics['mae'],
                r2_score=lstm_metrics['r2_score'],
                train_size=len(X_train_lstm),
                test_size=len(X_test_lstm),
                parameters=json.dumps(Config.LSTM_PARAMS)
            )
            db.session.add(lstm_metric_record)
            
            db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Models trained successfully',
            'rf_metrics': rf_metrics,
            'lstm_metrics': lstm_metrics
        })
        
    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make predictions using trained models
    """
    global rf_model, lstm_model
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', Config.DEFAULT_STOCK_SYMBOL)
        
        # Load models if not in memory
        if rf_model is None:
            rf_model = RandomForestModel()
            try:
                rf_model.load_model()
            except:
                return jsonify({'error': 'Random Forest model not found. Please train models first.'}), 400
        
        if lstm_model is None:
            lstm_model = LSTMModel()
            try:
                lstm_model.load_model()
            except:
                return jsonify({'error': 'LSTM model not found. Please train models first.'}), 400
        
        # Load latest data
        with app.app_context():
            latest_records = StockData.query.filter_by(symbol=symbol)\
                .order_by(StockData.date.desc())\
                .limit(100)\
                .all()
        
        if not latest_records:
            return jsonify({'error': 'No data found'}), 400
        
        # Reverse to chronological order
        latest_records = list(reversed(latest_records))
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'Date': r.date,
            'Open': r.open_price,
            'High': r.high_price,
            'Low': r.low_price,
            'Close': r.close_price,
            'Volume': r.volume
        } for r in latest_records])
        
        # Feature engineering
        fe = FeatureEngineer()
        df = fe.add_technical_indicators(df)
        df['sentiment_score'] = 0  # Use neutral sentiment for prediction
        df['polarity'] = 0
        df['subjectivity'] = 0.5
        df['Sentiment_MA_3'] = df['sentiment_score'].rolling(window=3).mean()
        df['Sentiment_MA_7'] = df['sentiment_score'].rolling(window=7).mean()
        df['Sentiment_MA_3'].fillna(0, inplace=True)
        df['Sentiment_MA_7'].fillna(0, inplace=True)
        
        # Get last record for prediction
        latest_data = df.iloc[[-1]]
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'Daily_Return', 'Volatility', 'ROC',
            'RSI', 'MACD', 'MACD_Signal',
            'Volume_Ratio', 'Momentum', 'HL_Spread',
            'sentiment_score', 'polarity', 'subjectivity',
            'Sentiment_MA_3', 'Sentiment_MA_7'
        ]
        
        available_features = [col for col in feature_columns if col in latest_data.columns]
        X_pred = latest_data[available_features].values
        
        # Random Forest prediction
        rf_prediction = rf_model.predict(X_pred)[0]
        current_price = float(df['Close'].iloc[-1])
        rf_signal = generate_trading_signals([rf_prediction], [current_price])[0]
        
        # LSTM prediction
        prices = df['Close'].values
        lstm_input = lstm_model.scaler.transform(prices[-lstm_model.sequence_length:].reshape(-1, 1))
        lstm_input = lstm_input.reshape(1, lstm_model.sequence_length, 1)
        lstm_prediction = lstm_model.predict(lstm_input)[0]
        lstm_signal = generate_trading_signals([lstm_prediction], [current_price])[0]
        
        # Ensemble prediction (average)
        ensemble_prediction = (rf_prediction + lstm_prediction) / 2
        ensemble_signal = generate_trading_signals([ensemble_prediction], [current_price])[0]
        
        return jsonify({
            'success': True,
            'current_price': current_price,
            'predictions': {
                'random_forest': float(rf_prediction),
                'lstm': float(lstm_prediction),
                'ensemble': float(ensemble_prediction)
            },
            'signals': {
                'random_forest': rf_signal,
                'lstm': lstm_signal,
                'ensemble': ensemble_signal
            },
            'date': df['Date'].iloc[-1].strftime('%Y-%m-%d')
        })
        
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock_data/<symbol>')
def get_stock_data(symbol):
    """Get historical stock data for visualization"""
    try:
        with app.app_context():
            records = StockData.query.filter_by(symbol=symbol)\
                .order_by(StockData.date)\
                .all()
        
        data = [{
            'date': r.date.strftime('%Y-%m-%d'),
            'close': r.close_price,
            'volume': r.volume
        } for r in records]
        
        return jsonify({'success': True, 'data': data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sentiment_data/<symbol>')
def get_sentiment_data(symbol):
    """Get sentiment analysis results"""
    try:
        with app.app_context():
            records = SentimentData.query.filter_by(symbol=symbol)\
                .order_by(SentimentData.date.desc())\
                .limit(50)\
                .all()
        
        data = [{
            'date': r.date.strftime('%Y-%m-%d'),
            'headline': r.headline,
            'sentiment_score': r.sentiment_score,
            'sentiment_label': r.sentiment_label
        } for r in records]
        
        return jsonify({'success': True, 'data': data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_metrics/<symbol>')
def get_model_metrics(symbol):
    """Get model performance metrics"""
    try:
        with app.app_context():
            rf_metrics = ModelMetrics.query.filter_by(
                symbol=symbol,
                model_type='RandomForest'
            ).order_by(ModelMetrics.training_date.desc()).first()
            
            lstm_metrics = ModelMetrics.query.filter_by(
                symbol=symbol,
                model_type='LSTM'
            ).order_by(ModelMetrics.training_date.desc()).first()
        
        return jsonify({
            'success': True,
            'random_forest': rf_metrics.to_dict() if rf_metrics else None,
            'lstm': lstm_metrics.to_dict() if lstm_metrics else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(Config.DATA_RAW_PATH, exist_ok=True)
    os.makedirs(Config.DATA_PROCESSED_PATH, exist_ok=True)
    os.makedirs(Config.MODELS_PATH, exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
