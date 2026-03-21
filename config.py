"""
Configuration file for Stock Price Forecasting System
MSc IT Project - University of Mumbai

This module contains all configuration parameters for the application
"""

import os
import shutil
from datetime import datetime, timedelta


def _as_bool(value):
    """Convert common environment variable strings to booleans."""
    return str(value or '').strip().lower() in {'1', 'true', 'yes', 'on'}

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'msc-stock-prediction-2024-secure-key'
    FLASK_APP = 'app.py'
    FLASK_ENV = 'development'
    
    # Database Configuration
    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    BUNDLED_STORAGE_ROOT = BASEDIR
    STORAGE_ROOT = os.path.abspath(os.environ.get('APP_STORAGE_ROOT') or BASEDIR)
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get('DATABASE_URL') or
        'sqlite:///' + os.path.join(STORAGE_ROOT, 'database', 'db.sqlite')
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Data Collection Parameters
    DEFAULT_STOCK_SYMBOL = ''
    STOCK_SYMBOLS = []
    
    # Historical data period
    START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # News API Configuration
    #NEWS_API_KEY = os.environ.get('') or '658ab30b6eed4f879c8409f469d43689'
    NEWS_API_KEY = os.environ.get('658ab30b6eed4f879c8409f469d43689') or '658ab30b6eed4f879c8409f469d43689'
    NEWS_API_ENDPOINT = 'https://newsapi.org/v2/everything'
    MAX_NEWS_ARTICLES = 100

    # Optional bootstrap credentials for creating the first admin user
    BOOTSTRAP_ADMIN_USERNAME = (
        os.environ.get('BOOTSTRAP_ADMIN_USERNAME') or
        os.environ.get('ADMIN_USERNAME')
    )
    BOOTSTRAP_ADMIN_PASSWORD = (
        os.environ.get('BOOTSTRAP_ADMIN_PASSWORD') or
        os.environ.get('ADMIN_PASSWORD')
    )
    BOOTSTRAP_ADMIN_SYNC = _as_bool(os.environ.get('BOOTSTRAP_ADMIN_SYNC'))
    ADMIN_USERNAME = BOOTSTRAP_ADMIN_USERNAME or 'admin'
    ADMIN_PASSWORD = BOOTSTRAP_ADMIN_PASSWORD or 'admin12345'
    
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
    DATA_RAW_PATH = os.path.join(STORAGE_ROOT, 'data', 'raw')
    DATA_PROCESSED_PATH = os.path.join(STORAGE_ROOT, 'data', 'processed')
    DATA_CONTRACTS_PATH = os.path.join(STORAGE_ROOT, 'data', 'contracts')
    MODELS_PATH = os.path.join(STORAGE_ROOT, 'models')
    CONTRACT_SYMBOLS_PATH = os.path.join(DATA_CONTRACTS_PATH, 'nse_contracts_latest.csv')
    CONTRACT_METADATA_PATH = os.path.join(DATA_CONTRACTS_PATH, 'nse_contracts_latest.json')
    
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
    def _sqlite_path(database_uri):
        """Extract the filesystem path from a SQLite database URI."""
        if not database_uri or database_uri == 'sqlite:///:memory:':
            return None

        prefix = 'sqlite:///'
        if database_uri.startswith(prefix):
            return database_uri[len(prefix):]

        return None

    @staticmethod
    def _copy_file_if_missing(source_path, target_path):
        """Copy a file into the runtime storage area if it does not exist yet."""
        if os.path.exists(source_path) and not os.path.exists(target_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)

    @staticmethod
    def _copy_directory_files_if_missing(source_dir, target_dir):
        """Copy top-level files from one directory to another when bootstrapping storage."""
        if not os.path.isdir(source_dir):
            return

        os.makedirs(target_dir, exist_ok=True)
        for entry in os.listdir(source_dir):
            source_path = os.path.join(source_dir, entry)
            target_path = os.path.join(target_dir, entry)
            if os.path.isfile(source_path) and not os.path.exists(target_path):
                shutil.copy2(source_path, target_path)

    @staticmethod
    def init_app(app):
        """Initialize application with configuration and runtime storage."""
        storage_root = app.config.get('STORAGE_ROOT', Config.BASEDIR)
        bundled_root = app.config.get('BUNDLED_STORAGE_ROOT', Config.BASEDIR)

        for path in (
            storage_root,
            app.config.get('DATA_RAW_PATH'),
            app.config.get('DATA_PROCESSED_PATH'),
            app.config.get('DATA_CONTRACTS_PATH'),
            app.config.get('MODELS_PATH'),
        ):
            if path:
                os.makedirs(path, exist_ok=True)

        sqlite_path = Config._sqlite_path(app.config.get('SQLALCHEMY_DATABASE_URI'))
        if sqlite_path:
            os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)

        if os.path.abspath(storage_root) == os.path.abspath(bundled_root):
            return

        if sqlite_path:
            Config._copy_file_if_missing(
                os.path.join(bundled_root, 'database', 'db.sqlite'),
                sqlite_path,
            )

        Config._copy_directory_files_if_missing(
            os.path.join(bundled_root, 'models'),
            app.config.get('MODELS_PATH'),
        )
        Config._copy_directory_files_if_missing(
            os.path.join(bundled_root, 'data', 'raw'),
            app.config.get('DATA_RAW_PATH'),
        )
        Config._copy_directory_files_if_missing(
            os.path.join(bundled_root, 'data', 'processed'),
            app.config.get('DATA_PROCESSED_PATH'),
        )

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
