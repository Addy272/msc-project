"""
Feature Engineering Module
MSc IT Project - Stock Price Forecasting

This module creates technical indicators and prepares features
for machine learning models
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class FeatureEngineer:
    """Create features for machine learning models"""
    
    def __init__(self):
        """Initialize Feature Engineer"""
        self.moving_averages = Config.MOVING_AVERAGES
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to dataframe
        
        Args:
            df (pd.DataFrame): Stock price dataframe
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Moving Averages
        for ma_period in self.moving_averages:
            df[f'MA_{ma_period}'] = df['Close'].rolling(window=ma_period).mean()
        
        # Daily Returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Volatility (20-day rolling standard deviation)
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        # Price Rate of Change
        df['ROC'] = df['Close'].pct_change(periods=10)
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Relative Strength Index (RSI)
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volume indicators
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # Price momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(4)
        
        # High-Low spread
        df['HL_Spread'] = df['High'] - df['Low']
        df['HL_Spread_MA'] = df['HL_Spread'].rolling(window=10).mean()
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index
        
        Args:
            prices (pd.Series): Price series
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def add_lag_features(self, df, lag_periods=[1, 2, 3, 5, 10]):
        """
        Add lagged price features
        
        Args:
            df (pd.DataFrame): Stock price dataframe
            lag_periods (list): List of lag periods
            
        Returns:
            pd.DataFrame: DataFrame with lag features
        """
        df = df.copy()
        
        for lag in lag_periods:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        return df
    
    def add_target_variable(self, df, horizon=1):
        """
        Add target variable for prediction
        
        Args:
            df (pd.DataFrame): Stock price dataframe
            horizon (int): Prediction horizon (days ahead)
            
        Returns:
            pd.DataFrame: DataFrame with target variable
        """
        df = df.copy()
        
        # Next day's closing price
        df['Target_Price'] = df['Close'].shift(-horizon)
        
        # Price direction (1 for up, 0 for down)
        df['Target_Direction'] = (df['Target_Price'] > df['Close']).astype(int)
        
        # Percentage change
        df['Target_Change'] = ((df['Target_Price'] - df['Close']) / df['Close']) * 100
        
        return df
    
    def merge_sentiment(self, stock_df, sentiment_df):
        """
        Merge stock data with sentiment scores
        
        Args:
            stock_df (pd.DataFrame): Stock price dataframe
            sentiment_df (pd.DataFrame): Sentiment scores dataframe
            
        Returns:
            pd.DataFrame: Merged dataframe
        """
        # Ensure date columns are datetime
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Aggregate sentiment by date (average of all news articles per day)
        sentiment_agg = sentiment_df.groupby('date').agg({
            'sentiment_score': 'mean',
            'polarity': 'mean',
            'subjectivity': 'mean'
        }).reset_index()
        
        sentiment_agg.rename(columns={'date': 'Date'}, inplace=True)
        
        # Merge with stock data
        merged_df = pd.merge(
            stock_df,
            sentiment_agg,
            on='Date',
            how='left'
        )
        
        # Fill missing sentiment values with 0 (neutral)
        merged_df['sentiment_score'].fillna(0, inplace=True)
        merged_df['polarity'].fillna(0, inplace=True)
        merged_df['subjectivity'].fillna(0.5, inplace=True)
        
        # Create sentiment momentum features
        merged_df['Sentiment_MA_3'] = merged_df['sentiment_score'].rolling(window=3).mean()
        merged_df['Sentiment_MA_7'] = merged_df['sentiment_score'].rolling(window=7).mean()
        
        return merged_df
    
    def prepare_ml_features(self, df):
        """
        Prepare final feature set for ML models
        
        Args:
            df (pd.DataFrame): Dataframe with all features
            
        Returns:
            tuple: (features_df, target_series, feature_names)
        """
        # Drop rows with NaN values
        df_clean = df.dropna()
        
        # Select feature columns
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'Daily_Return', 'Volatility', 'ROC',
            'RSI', 'MACD', 'MACD_Signal',
            'Volume_Ratio', 'Momentum', 'HL_Spread',
            'sentiment_score', 'polarity', 'subjectivity',
            'Sentiment_MA_3', 'Sentiment_MA_7'
        ]
        
        # Check which features exist
        available_features = [col for col in feature_columns if col in df_clean.columns]
        
        # Extract features and target
        X = df_clean[available_features]
        
        # Choose target based on what's available
        if 'Target_Price' in df_clean.columns:
            y = df_clean['Target_Price']
        else:
            # If no target, create it
            y = df_clean['Close'].shift(-1)
            df_clean = df_clean[:-1]  # Remove last row
            X = X[:-1]
        
        return X, y, available_features
    
    def create_lstm_sequences(self, data, sequence_length=60):
        """
        Create sequences for LSTM model
        
        Args:
            data (np.array): Normalized price data
            sequence_length (int): Length of input sequences
            
        Returns:
            tuple: (X_sequences, y_targets)
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def save_processed_data(self, df, filename):
        """
        Save processed data to CSV
        
        Args:
            df (pd.DataFrame): Processed dataframe
            filename (str): Output filename
        """
        filepath = os.path.join(Config.DATA_PROCESSED_PATH, filename)
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filename):
        """
        Load processed data from CSV
        
        Args:
            filename (str): Input filename
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        filepath = os.path.join(Config.DATA_PROCESSED_PATH, filename)
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Processed data loaded from {filepath}")
        return df


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import StockDataLoader
    
    print("Testing Feature Engineering...")
    
    # Load sample data
    loader = StockDataLoader('AAPL')
    df = loader.fetch_historical_data()
    
    if df is not None:
        # Initialize feature engineer
        fe = FeatureEngineer()
        
        # Add technical indicators
        df = fe.add_technical_indicators(df)
        print("\nDataFrame with technical indicators:")
        print(df.columns.tolist())
        print(f"Shape: {df.shape}")
        
        # Add target variable
        df = fe.add_target_variable(df)
        print("\nTarget variables added")
        
        # Prepare ML features
        X, y, features = fe.prepare_ml_features(df)
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature names: {features}")
