"""
Configuration file for Stock Price Forecasting System
MSc IT Project - University of Mumbai

This module contains all configuration parameters for the application
"""

import os
from datetime import datetime, timedelta

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'msc-stock-prediction-2024-secure-key'
    FLASK_APP = 'app.py'
    FLASK_ENV = 'development'
    
    # Database Configuration
    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASEDIR, 'database', 'db.sqlite')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Data Collection Parameters
    DEFAULT_STOCK_SYMBOL = 'AAPL'  # Default stock for demonstration
    STOCK_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']
    
    # Historical data period
    START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # News API Configuration
    #NEWS_API_KEY = os.environ.get('') or '658ab30b6eed4f879c8409f469d43689'
    NEWS_API_KEY = os.environ.get('658ab30b6eed4f879c8409f469d43689') or '658ab30b6eed4f879c8409f469d43689'
    NEWS_API_ENDPOINT = 'https://newsapi.org/v2/everything'
    MAX_NEWS_ARTICLES = 100

    # Admin Panel Credentials
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME') or 'admin'
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD') or 'admin@1234'
    
    # Model Parameters
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    }
    
    LSTM_PARAMS = {
        'sequence_length': 60,  # Use 60 days of data to predict next day
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'units': [50, 50],  # Two LSTM layers
        'dropout': 0.2
    }
    
    # Feature Engineering
    MOVING_AVERAGES = [5, 10, 20, 50]  # Days for moving averages
    
    # Train-Test Split
    TRAIN_TEST_SPLIT_RATIO = 0.8
    
    # Prediction Thresholds
    BUY_THRESHOLD = 0.02  # 2% increase
    SELL_THRESHOLD = -0.02  # 2% decrease
    
    # Sentiment Analysis
    SENTIMENT_WEIGHT = 0.3  # Weight of sentiment in final prediction
    
    # File Paths
    DATA_RAW_PATH = os.path.join(BASEDIR, 'data', 'raw')
    DATA_PROCESSED_PATH = os.path.join(BASEDIR, 'data', 'processed')
    MODELS_PATH = os.path.join(BASEDIR, 'models')
    
    # Model Save Paths
    RF_MODEL_PATH = os.path.join(MODELS_PATH, 'random_forest_model.pkl')
    LSTM_MODEL_PATH = os.path.join(MODELS_PATH, 'lstm_model.h5')
    SCALER_PATH = os.path.join(MODELS_PATH, 'scaler.pkl')
    
    # Visualization
    PLOT_STYLE = 'seaborn-v0_8'
    FIGURE_SIZE = (12, 6)
    
    # Logging
    LOG_LEVEL = 'INFO'
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        pass

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
