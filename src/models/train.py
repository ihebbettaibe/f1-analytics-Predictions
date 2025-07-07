import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1ModelTrainer:
    def __init__(self):
        """Initialize the F1 model trainer."""
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for model training.
        
        Args:
            X (pd.DataFrame): Raw features
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        # Convert categorical variables
        X_processed = pd.get_dummies(X, columns=['driver', 'gp'])
        
        # Scale numerical features
        numerical_cols = ['avg_speed', 'avg_lap_time', 'best_lap_time', 'tire_usage']
        X_processed[numerical_cols] = self.scaler.fit_transform(X_processed[numerical_cols])
        
        return X_processed
        
    def train_model(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Train the F1 prediction model.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): Targets
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocess features
        X_train_processed = self.preprocess_features(X_train)
        X_test_processed = self.preprocess_features(X_test)
        
        # Define model and hyperparameters
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Train model for position prediction
        position_model = RandomForestRegressor(random_state=42)
        position_grid = GridSearchCV(
            position_model, param_grid, cv=5, scoring='neg_mean_squared_error'
        )
        position_grid.fit(X_train_processed, y_train['position'])
        
        # Train model for points prediction
        points_model = RandomForestRegressor(random_state=42)
        points_grid = GridSearchCV(
            points_model, param_grid, cv=5, scoring='neg_mean_squared_error'
        )
        points_grid.fit(X_train_processed, y_train['points'])
        
        # Evaluate models
        position_pred = position_grid.predict(X_test_processed)
        points_pred = points_grid.predict(X_test_processed)
        
        metrics = {
            'position': {
                'mse': mean_squared_error(y_test['position'], position_pred),
                'r2': r2_score(y_test['position'], position_pred)
            },
            'points': {
                'mse': mean_squared_error(y_test['points'], points_pred),
                'r2': r2_score(y_test['points'], points_pred)
            }
        }
        
        # Save models and scaler
        self.model = {
            'position': position_grid.best_estimator_,
            'points': points_grid.best_estimator_
        }
        
        joblib.dump(self.model, 'models/f1_prediction_model.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        
        return metrics
        
    def save_model(self, path: str = 'models/f1_prediction_model.joblib') -> None:
        """Save the trained model.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model trained yet")
            
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

if __name__ == "__main__":
    # Example usage
    try:
        # Load processed data
        X = pd.read_csv('data/processed/features.csv')
        y = pd.read_csv('data/processed/targets.csv')
        
        # Train model
        trainer = F1ModelTrainer()
        metrics = trainer.train_model(X, y)
        
        # Print metrics
        print("\nModel Performance Metrics:")
        print("Position Prediction:")
        print(f"MSE: {metrics['position']['mse']:.4f}")
        print(f"R2 Score: {metrics['position']['r2']:.4f}")
        print("\nPoints Prediction:")
        print(f"MSE: {metrics['points']['mse']:.4f}")
        print(f"R2 Score: {metrics['points']['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}") 