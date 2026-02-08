"""
Database Models for Stock Price Forecasting System
MSc IT Project

This module defines the database schema using SQLAlchemy ORM
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class StockData(db.Model):
    """Store historical stock price data"""
    __tablename__ = 'stock_data'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False, index=True)
    open_price = db.Column(db.Float, nullable=False)
    high_price = db.Column(db.Float, nullable=False)
    low_price = db.Column(db.Float, nullable=False)
    close_price = db.Column(db.Float, nullable=False)
    volume = db.Column(db.BigInteger, nullable=False)
    adj_close = db.Column(db.Float)
    
    # Technical Indicators
    ma_5 = db.Column(db.Float)
    ma_10 = db.Column(db.Float)
    ma_20 = db.Column(db.Float)
    ma_50 = db.Column(db.Float)
    daily_return = db.Column(db.Float)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint('symbol', 'date', name='unique_symbol_date'),
    )
    
    def __repr__(self):
        return f'<StockData {self.symbol} {self.date}>'
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'date': self.date.strftime('%Y-%m-%d'),
            'open': self.open_price,
            'high': self.high_price,
            'low': self.low_price,
            'close': self.close_price,
            'volume': self.volume,
            'adj_close': self.adj_close,
            'ma_5': self.ma_5,
            'ma_10': self.ma_10,
            'ma_20': self.ma_20,
            'ma_50': self.ma_50,
            'daily_return': self.daily_return
        }


class SentimentData(db.Model):
    """Store sentiment analysis results"""
    __tablename__ = 'sentiment_data'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False, index=True)
    
    # News article details
    headline = db.Column(db.Text)
    source = db.Column(db.String(100))
    url = db.Column(db.Text)
    
    # Sentiment scores
    sentiment_score = db.Column(db.Float)  # -1 to 1
    sentiment_label = db.Column(db.String(20))  # Positive/Negative/Neutral
    polarity = db.Column(db.Float)
    subjectivity = db.Column(db.Float)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SentimentData {self.symbol} {self.date} {self.sentiment_label}>'
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'date': self.date.strftime('%Y-%m-%d'),
            'headline': self.headline,
            'source': self.source,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'polarity': self.polarity,
            'subjectivity': self.subjectivity
        }


class PredictionData(db.Model):
    """Store model predictions"""
    __tablename__ = 'prediction_data'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False, index=True)
    prediction_date = db.Column(db.Date, nullable=False, index=True)
    target_date = db.Column(db.Date, nullable=False)  # Date being predicted
    
    # Model predictions
    model_type = db.Column(db.String(20))  # 'RandomForest' or 'LSTM'
    predicted_price = db.Column(db.Float, nullable=False)
    actual_price = db.Column(db.Float)
    
    # Trading signal
    signal = db.Column(db.String(10))  # 'BUY', 'SELL', 'HOLD'
    confidence = db.Column(db.Float)
    
    # Performance metrics
    prediction_error = db.Column(db.Float)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<PredictionData {self.symbol} {self.model_type} {self.signal}>'
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'prediction_date': self.prediction_date.strftime('%Y-%m-%d'),
            'target_date': self.target_date.strftime('%Y-%m-%d'),
            'model_type': self.model_type,
            'predicted_price': self.predicted_price,
            'actual_price': self.actual_price,
            'signal': self.signal,
            'confidence': self.confidence,
            'prediction_error': self.prediction_error
        }


class ModelMetrics(db.Model):
    """Store model performance metrics"""
    __tablename__ = 'model_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(20), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    
    # Performance metrics
    accuracy = db.Column(db.Float)
    rmse = db.Column(db.Float)
    mae = db.Column(db.Float)
    r2_score = db.Column(db.Float)
    
    # Training details
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    train_size = db.Column(db.Integer)
    test_size = db.Column(db.Integer)
    
    # Model parameters (stored as JSON string)
    parameters = db.Column(db.Text)
    
    def __repr__(self):
        return f'<ModelMetrics {self.model_type} {self.symbol} RMSE:{self.rmse}>'
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'model_type': self.model_type,
            'symbol': self.symbol,
            'accuracy': self.accuracy,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2_score': self.r2_score,
            'training_date': self.training_date.strftime('%Y-%m-%d %H:%M:%S'),
            'train_size': self.train_size,
            'test_size': self.test_size
        }


def init_db(app):
    """Initialize database with Flask app"""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("Database initialized successfully!")
