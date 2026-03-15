"""
Random Forest Model for Stock Price Prediction
MSc IT Project - Stock Price Forecasting

This module implements Random Forest classifier and regressor
for stock price prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class RandomForestModel:
    """Random Forest model for stock price prediction"""
    
    def __init__(self, task='regression', params=None):
        """
        Initialize Random Forest Model
        
        Args:
            task (str): 'regression' for price prediction or 'classification' for direction
            params (dict): Model hyperparameters
        """
        self.task = task
        self.params = params or Config.RANDOM_FOREST_PARAMS
        self.scaler = StandardScaler()
        
        # Initialize model based on task
        if task == 'regression':
            self.model = RandomForestRegressor(**self.params)
        else:
            self.model = RandomForestClassifier(**self.params)
        
        self.feature_names = None
        self.is_trained = False
    
    def prepare_data(self, X, y, test_size=0.2):
        """
        Prepare and split data for training
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Test set size ratio
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            shuffle=False  # Important for time series
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
        """
        print(f"Training Random Forest {self.task} model...")
        print(f"Training samples: {len(X_train)}")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Get feature importance
        if self.feature_names:
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10))
        
        print("Training completed!")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (np.array): Feature matrix
            
        Returns:
            np.array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        if self.task == 'regression':
            # Regression metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'mape': mape
            }
            
            print("\nModel Evaluation (Regression):")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
            print(f"MAPE: {mape:.2f}%")
            
        else:
            # Classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            metrics = {
                'accuracy': accuracy
            }
            
            print("\nModel Evaluation (Classification):")
            print(f"Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target variable
            cv (int): Number of folds
            
        Returns:
            dict: Cross-validation scores
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.task == 'regression':
            # Negative MSE for regression
            scores = cross_val_score(
                self.model, X_scaled, y,
                cv=cv,
                scoring='neg_mean_squared_error'
            )
            rmse_scores = np.sqrt(-scores)
            
            print(f"Cross-validation RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
            
            return {
                'cv_rmse_mean': rmse_scores.mean(),
                'cv_rmse_std': rmse_scores.std(),
                'cv_scores': rmse_scores
            }
        else:
            # Accuracy for classification
            scores = cross_val_score(
                self.model, X_scaled, y,
                cv=cv,
                scoring='accuracy'
            )
            
            print(f"Cross-validation Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
            
            return {
                'cv_accuracy_mean': scores.mean(),
                'cv_accuracy_std': scores.std(),
                'cv_scores': scores
            }
    
    def get_feature_importance(self):
        """
        Get feature importance dataframe
        
        Returns:
            pd.DataFrame: Feature importance
        """
        if not self.is_trained or not self.feature_names:
            return None
        
        return self.feature_importance
    
    def save_model(self, filepath=None):
        """
        Save model to disk
        
        Args:
            filepath (str): Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            filepath = Config.RF_MODEL_PATH
        
        # Save model and scaler
        joblib.dump(self.model, filepath)
        scaler_path = filepath.replace('.pkl', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {filepath}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, filepath=None):
        """
        Load model from disk
        
        Args:
            filepath (str): Path to load model from
        """
        if filepath is None:
            filepath = Config.RF_MODEL_PATH
        
        # Load model and scaler
        self.model = joblib.load(filepath)
        scaler_path = filepath.replace('.pkl', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def generate_trading_signals(predictions, current_prices, threshold=0.02):
    """
    Generate Buy/Sell/Hold signals based on predictions
    
    Args:
        predictions (np.array): Predicted prices
        current_prices (np.array): Current prices
        threshold (float): Threshold for signal generation
        
    Returns:
        list: Trading signals
    """
    signals = []
    
    for pred, current in zip(predictions, current_prices):
        change = (pred - current) / current
        
        if change > threshold:
            signals.append('BUY')
        elif change < -threshold:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    return signals


if __name__ == "__main__":
    # Test Random Forest model
    print("Testing Random Forest Model...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 100 + 150  # Price around 150
    
    # Initialize and train model
    rf_model = RandomForestModel(task='regression')
    
    # Prepare data
    X_train, X_test, y_train, y_test = rf_model.prepare_data(X, y, test_size=0.2)
    
    # Train
    rf_model.train(X_train, y_train)
    
    # Evaluate
    metrics = rf_model.evaluate(X_test, y_test)
    
    # Make predictions
    predictions = rf_model.predict(X_test[:10])
    print(f"\nSample Predictions: {predictions[:5]}")
    print(f"Actual Values: {y_test[:5].values}")
    
    # Generate signals
    signals = generate_trading_signals(predictions[:5], y_test[:5].values)
    print(f"Trading Signals: {signals}")
