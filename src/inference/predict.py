import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Any
from ..data_pipeline.data_processing import F1DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1Predictor:
    def __init__(self, model_path: str = 'models/f1_prediction_model.joblib',
                 scaler_path: str = 'models/scaler.joblib'):
        """Initialize the F1 predictor.
        
        Args:
            model_path (str): Path to the trained model
            scaler_path (str): Path to the scaler
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.data_processor = F1DataProcessor()
            logger.info("Successfully loaded model and scaler")
        except Exception as e:
            logger.error(f"Error loading model or scaler: {str(e)}")
            raise
            
    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for prediction.
        
        Args:
            X (pd.DataFrame): Raw features
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        # Convert categorical variables
        X_processed = pd.get_dummies(X, columns=['driver', 'gp'])
        
        # Scale numerical features
        numerical_cols = ['avg_speed', 'avg_lap_time', 'best_lap_time', 'tire_usage']
        X_processed[numerical_cols] = self.scaler.transform(X_processed[numerical_cols])
        
        return X_processed
        
    def predict_race(self, year: int, gp: str) -> Dict[str, Any]:
        """Predict race results for a specific Grand Prix.
        
        Args:
            year (int): Year of the race
            gp (str): Grand Prix name
            
        Returns:
            Dict[str, Any]: Predictions for position and points
        """
        try:
            # Load and process race data
            self.data_processor.load_race_session(year, gp)
            features = self.data_processor.extract_features()
            
            # Preprocess features
            X_processed = self.preprocess_features(features)
            
            # Make predictions
            position_pred = self.model['position'].predict(X_processed)
            points_pred = self.model['points'].predict(X_processed)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'driver': features['driver'],
                'predicted_position': position_pred.round().astype(int),
                'predicted_points': points_pred.round().astype(int)
            })
            
            # Sort by predicted position
            results = results.sort_values('predicted_position')
            
            return {
                'predictions': results.to_dict('records'),
                'race_info': {
                    'year': year,
                    'grand_prix': gp
                }
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
            
    def evaluate_predictions(self, year: int, gp: str) -> Dict[str, float]:
        """Evaluate prediction accuracy against actual results.
        
        Args:
            year (int): Year of the race
            gp (str): Grand Prix name
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        try:
            # Get predictions
            predictions = self.predict_race(year, gp)
            pred_df = pd.DataFrame(predictions['predictions'])
            
            # Get actual results
            self.data_processor.load_race_session(year, gp)
            actual_results = self.data_processor.extract_features()
            
            # Calculate metrics
            position_error = np.mean(np.abs(
                pred_df['predicted_position'] - actual_results['position']
            ))
            points_error = np.mean(np.abs(
                pred_df['predicted_points'] - actual_results['points']
            ))
            
            return {
                'mean_position_error': position_error,
                'mean_points_error': points_error
            }
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    try:
        predictor = F1Predictor()
        
        # Make predictions for a specific race
        year = 2023
        gp = 'Monaco'
        
        predictions = predictor.predict_race(year, gp)
        print(f"\nPredictions for {year} {gp}:")
        for pred in predictions['predictions']:
            print(f"Driver: {pred['driver']}")
            print(f"Predicted Position: {pred['predicted_position']}")
            print(f"Predicted Points: {pred['predicted_points']}\n")
            
        # Evaluate predictions
        metrics = predictor.evaluate_predictions(year, gp)
        print("\nPrediction Accuracy:")
        print(f"Mean Position Error: {metrics['mean_position_error']:.2f}")
        print(f"Mean Points Error: {metrics['mean_points_error']:.2f}")
        
    except Exception as e:
        logger.error(f"Error in prediction example: {str(e)}") 