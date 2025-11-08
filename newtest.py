import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
import joblib
import json
from datetime import datetime, timedelta
import ta  # Technical Analysis library

warnings.filterwarnings('ignore')

class StockMarketPredictor:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Linear Regression': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = []
        self.results = {}
        
    def load_data(self, csv_file):
        """Load stock data from CSV file"""
        try:
            df = pd.IBM_data.csv(csv_file)
            # Ensure required columns exist
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df = df.reset_index(drop=True)
            
            print(f"Data loaded successfully: {len(df)} records")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_technical_indicators(self, df):
        """Create technical indicators as features"""
        df = df.copy()
        
        # Basic price features
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
        
        # Exponential Moving Averages
        for window in [12, 26]:
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # Price momentum
        for period in [1, 3, 5, 10]:
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
            df[f'Return_{period}'] = (df['Close'] / df['Close'].shift(period) - 1) * 100
        
        # Stochastic Oscillator
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * (high_max - df['Close']) / (high_max - low_min)
        
        # Average True Range (ATR)
        df['TR1'] = df['High'] - df['Low']
        df['TR2'] = abs(df['High'] - df['Close'].shift(1))
        df['TR3'] = abs(df['Low'] - df['Close'].shift(1))
        df['True_Range'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        
        # Clean up temporary columns
        df = df.drop(['TR1', 'TR2', 'TR3', 'True_Range'], axis=1, errors='ignore')
        
        return df
    
    def prepare_features(self, df, target_days=1):
        """Prepare features and target variable"""
        # Create target variable (next day's closing price)
        df['Target'] = df['Close'].shift(-target_days)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Select feature columns (exclude Date, Target, and basic OHLCV for features)
        feature_columns = [col for col in df.columns if col not in ['Date', 'Target', 'Close']]
        
        X = df[feature_columns]
        y = df['Target']
        
        self.feature_names = feature_columns
        
        print(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        return X, y, df
    
    def train_models(self, X, y):
        """Train multiple models and find the best one"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        best_score = float('-inf')
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            if name == 'Random Forest':
                # Hyperparameter tuning for Random Forest
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
                grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
                
            elif name == 'Gradient Boosting':
                # Hyperparameter tuning for Gradient Boosting
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5]
                }
                grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
            else:
                model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"{name} - R² Score: {r2:.4f}, RMSE: {rmse:.4f}")
            
            # Update best model
            if r2 > best_score:
                best_score = r2
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest Model: {self.best_model_name} (R² Score: {best_score:.4f})")
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance(self):
        """Get feature importance from the best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            return importance_df
        return None
    
    def predict_future(self, df, days=5):
        """Predict future stock prices"""
        predictions = []
        last_data = df.iloc[-1:].copy()
        
        for i in range(days):
            # Prepare features for prediction
            feature_data = last_data[self.feature_names].values
            feature_data_scaled = self.scaler.transform(feature_data)
            
            # Make prediction
            pred = self.best_model.predict(feature_data_scaled)[0]
            predictions.append(pred)
            
            # Update last_data for next prediction (simplified approach)
            # In a real scenario, you'd need to update all technical indicators
            last_data.loc[:, 'Close'] = pred
        
        return predictions
    
    def save_model(self, filepath='stock_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_model_name': self.best_model_name,
            'results': {name: {k: v for k, v in result.items() if k not in ['model', 'predictions', 'actual']} 
                       for name, result in self.results.items()}
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='stock_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.best_model_name = model_data['best_model_name']
        print(f"Model loaded from {filepath}")
    
    def generate_report(self):
        """Generate a comprehensive report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'model_performance': {},
            'feature_importance': None
        }
        
        # Add model performance
        for name, result in self.results.items():
            report['model_performance'][name] = {
                'r2_score': float(result['r2']),
                'rmse': float(result['rmse']),
                'mae': float(result['mae']),
                'mse': float(result['mse'])
            }
        
        # Add feature importance
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            report['feature_importance'] = importance_df.head(10).to_dict('records')
        
        # Save report
        with open('model_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    # Example usage
    predictor = StockMarketPredictor()
    
    # Load data (replace 'stock_data.csv' with your CSV file path)
    print("Loading stock data...")
    df = predictor.load_data('stock_data.csv')
    
    if df is not None:
        # Create technical indicators
        print("Creating technical indicators...")
        df = predictor.create_technical_indicators(df)
        
        # Prepare features
        print("Preparing features...")
        X, y, df_processed = predictor.prepare_features(df)
        
        # Train models
        print("Training models...")
        X_train, X_test, y_train, y_test = predictor.train_models(X, y)
        
        # Generate predictions for next 5 days
        print("Generating future predictions...")
        future_predictions = predictor.predict_future(df_processed, days=5)
        print(f"Next 5 days predictions: {future_predictions}")
        
        # Get feature importance
        importance_df = predictor.get_feature_importance()
        if importance_df is not None:
            print("\nTop 10 Important Features:")
            print(importance_df.head(10))
        
        # Save model
        predictor.save_model()
        
        # Generate report
        report = predictor.generate_report()
        print("\nModel training completed successfully!")
        print(f"Best model: {report['best_model']}")
        print(f"R² Score: {report['model_performance'][report['best_model']]['r2_score']:.4f}")

if __name__ == "__main__":
    main()