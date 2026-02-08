"""
LSTM Model for Stock Price Forecasting
MSc IT Project - Stock Price Forecasting

This module implements Long Short-Term Memory (LSTM) neural network
for time series prediction of stock prices
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class LSTMModel:
    """LSTM model for stock price prediction"""
    
    def __init__(self, params=None):
        """
        Initialize LSTM Model
        
        Args:
            params (dict): Model hyperparameters
        """
        self.params = params or Config.LSTM_PARAMS
        self.sequence_length = self.params['sequence_length']
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        self.is_trained = False
    
    def prepare_data(self, data, test_size=0.2):
        """
        Prepare data for LSTM training
        
        Args:
            data (np.array or pd.Series): Price data
            test_size (float): Test set size ratio
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler)
        """
        # Convert to numpy array if pandas Series
        if isinstance(data, pd.Series):
            data = data.values.reshape(-1, 1)
        elif len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Calculate split index
        split_idx = int(len(scaled_data) * (1 - test_size))
        
        # Split into train and test
        train_data = scaled_data[:split_idx]
        test_data = scaled_data[split_idx - self.sequence_length:]
        
        # Create sequences
        X_train, y_train = self._create_sequences(train_data)
        X_test, y_test = self._create_sequences(test_data)
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Testing sequences: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def _create_sequences(self, data):
        """
        Create sequences for LSTM
        
        Args:
            data (np.array): Scaled data
            
        Returns:
            tuple: (X, y) sequences
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i, 0])
            y.append(data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        
        Args:
            input_shape (tuple): Input shape (sequence_length, n_features)
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.params['units'][0],
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(self.params['dropout']))
        
        # Second LSTM layer
        if len(self.params['units']) > 1:
            model.add(LSTM(
                units=self.params['units'][1],
                return_sequences=False
            ))
            model.add(Dropout(self.params['dropout']))
        
        # Dense output layer
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        self.model = model
        
        print("\nLSTM Model Architecture:")
        model.summary()
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the LSTM model
        
        Args:
            X_train (np.array): Training sequences
            y_train (np.array): Training targets
            X_val (np.array): Validation sequences (optional)
            y_val (np.array): Validation targets (optional)
        """
        if self.model is None:
            # Build model if not already built
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        print(f"\nTraining LSTM model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Epochs: {self.params['epochs']}")
        print(f"Batch size: {self.params['batch_size']}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            print(f"Validation samples: {len(X_val)}")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        print("\nTraining completed!")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (np.array): Input sequences
            
        Returns:
            np.array: Predictions (inverse scaled)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Make predictions
        predictions_scaled = self.model.predict(X, verbose=0)
        
        # Inverse scale predictions
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (np.array): Test sequences
            y_test (np.array): Test targets (scaled)
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        predictions_scaled = self.model.predict(X_test, verbose=0)
        
        # Inverse scale
        predictions = self.scaler.inverse_transform(predictions_scaled).flatten()
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        mae = mean_absolute_error(y_test_actual, predictions)
        r2 = r2_score(y_test_actual, predictions)
        mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape
        }
        
        print("\nModel Evaluation (LSTM):")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path (str): Path to save plot
        """
        if self.history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.close()
    
    def plot_predictions(self, y_true, y_pred, dates=None, save_path=None):
        """
        Plot actual vs predicted values
        
        Args:
            y_true (np.array): Actual values
            y_pred (np.array): Predicted values
            dates (pd.DatetimeIndex): Dates for x-axis
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(14, 6))
        
        if dates is not None:
            plt.plot(dates, y_true, label='Actual Price', color='blue', linewidth=2)
            plt.plot(dates, y_pred, label='Predicted Price', color='red', linewidth=2, alpha=0.7)
        else:
            plt.plot(y_true, label='Actual Price', color='blue', linewidth=2)
            plt.plot(y_pred, label='Predicted Price', color='red', linewidth=2, alpha=0.7)
        
        plt.title('LSTM Stock Price Prediction', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions plot saved to {save_path}")
        
        plt.close()
    
    def save_model(self, filepath=None):
        """
        Save model to disk
        
        Args:
            filepath (str): Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            filepath = Config.LSTM_MODEL_PATH
        
        # Save model
        self.model.save(filepath)
        
        # Save scaler
        import joblib
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
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
            filepath = Config.LSTM_MODEL_PATH
        
        # Load model
        self.model = keras.models.load_model(filepath)
        
        # Load scaler
        import joblib
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test LSTM model
    print("Testing LSTM Model...")
    
    # Create sample time series data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic stock price data
    t = np.linspace(0, 100, n_samples)
    trend = t * 0.5
    seasonality = 20 * np.sin(t * 0.2)
    noise = np.random.randn(n_samples) * 5
    prices = 100 + trend + seasonality + noise
    
    # Initialize and train model
    lstm_model = LSTMModel()
    
    # Prepare data
    X_train, X_test, y_train, y_test = lstm_model.prepare_data(prices, test_size=0.2)
    
    # Build and train
    lstm_model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    lstm_model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    metrics = lstm_model.evaluate(X_test, y_test)
    
    # Make predictions
    predictions = lstm_model.predict(X_test)
    y_test_actual = lstm_model.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"\nSample Predictions: {predictions[:5]}")
    print(f"Actual Values: {y_test_actual[:5]}")
