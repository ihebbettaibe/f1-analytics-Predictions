import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_input_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate input data has required columns.
    
    Args:
        data (pd.DataFrame): Input data to validate
        required_columns (List[str]): List of required column names
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    return True

def format_predictions(predictions: Dict[str, Any]) -> pd.DataFrame:
    """Format predictions into a readable DataFrame.
    
    Args:
        predictions (Dict[str, Any]): Raw predictions dictionary
        
    Returns:
        pd.DataFrame: Formatted predictions
    """
    try:
        df = pd.DataFrame(predictions['predictions'])
        df['race_year'] = predictions['race_info']['year']
        df['grand_prix'] = predictions['race_info']['grand_prix']
        return df
    except Exception as e:
        logger.error(f"Error formatting predictions: {str(e)}")
        raise

def calculate_prediction_metrics(actual: pd.DataFrame, predicted: pd.DataFrame) -> Dict[str, float]:
    """Calculate prediction accuracy metrics.
    
    Args:
        actual (pd.DataFrame): Actual race results
        predicted (pd.DataFrame): Predicted race results
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    try:
        metrics = {
            'position_accuracy': np.mean(
                actual['position'] == predicted['predicted_position']
            ),
            'points_accuracy': np.mean(
                actual['points'] == predicted['predicted_points']
            ),
            'mean_position_error': np.mean(
                np.abs(actual['position'] - predicted['predicted_position'])
            ),
            'mean_points_error': np.mean(
                np.abs(actual['points'] - predicted['predicted_points'])
            )
        }
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

def save_predictions(predictions: pd.DataFrame, filename: str = None) -> str:
    """Save predictions to CSV file.
    
    Args:
        predictions (pd.DataFrame): Predictions to save
        filename (str, optional): Custom filename. Defaults to timestamp-based name.
        
    Returns:
        str: Path to saved file
    """
    try:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'predictions_{timestamp}.csv'
            
        predictions.to_csv(f'data/predictions/{filename}', index=False)
        logger.info(f"Predictions saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        raise

def load_historical_data(years: List[int], gps: List[str]) -> pd.DataFrame:
    """Load and combine historical F1 data.
    
    Args:
        years (List[int]): List of years to include
        gps (List[str]): List of Grand Prix names to include
        
    Returns:
        pd.DataFrame: Combined historical data
    """
    try:
        all_data = []
        for year in years:
            for gp in gps:
                try:
                    data = pd.read_csv(f'data/processed/{year}_{gp}.csv')
                    data['year'] = year
                    data['grand_prix'] = gp
                    all_data.append(data)
                except FileNotFoundError:
                    logger.warning(f"No data found for {year} {gp}")
                    continue
                    
        if not all_data:
            raise ValueError("No historical data found")
            
        return pd.concat(all_data, ignore_index=True)
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        # Test data validation
        test_data = pd.DataFrame({
            'driver': ['HAM', 'VER', 'LEC'],
            'position': [1, 2, 3],
            'points': [25, 18, 15]
        })
        
        required_cols = ['driver', 'position', 'points']
        is_valid = validate_input_data(test_data, required_cols)
        print(f"Data validation test: {'Passed' if is_valid else 'Failed'}")
        
    except Exception as e:
        logger.error(f"Error in helper functions example: {str(e)}") 