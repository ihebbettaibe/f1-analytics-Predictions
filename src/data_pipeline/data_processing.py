import fastf1
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1DataProcessor:
    def __init__(self, cache_dir: str = 'cache'):
        """Initialize the F1 data processor.
        
        Args:
            cache_dir (str): Directory to store cached F1 data
        """
        fastf1.Cache.enable_cache(cache_dir)
        self.session = None
        
    def load_race_session(self, year: int, gp: str, session: str = 'R') -> None:
        """Load a specific race session.
        
        Args:
            year (int): Year of the race
            gp (str): Grand Prix name
            session (str): Session type (R for race, Q for qualifying)
        """
        try:
            self.session = fastf1.get_session(year, gp, session)
            self.session.load()
            logger.info(f"Successfully loaded {year} {gp} {session} session")
        except Exception as e:
            logger.error(f"Error loading session: {str(e)}")
            raise

    def extract_features(self) -> pd.DataFrame:
        """Extract relevant features from the race session.
        
        Returns:
            pd.DataFrame: DataFrame containing processed features
        """
        if self.session is None:
            raise ValueError("No session loaded. Call load_race_session first.")
            
        # Get lap data
        laps = self.session.laps
        
        # Extract relevant features
        features = []
        for driver in self.session.drivers:
            driver_laps = laps.pick_driver(driver)
            
            if len(driver_laps) == 0:
                continue
                
            driver_data = {
                'driver': driver,
                'avg_speed': driver_laps['Speed'].mean(),
                'avg_lap_time': driver_laps['LapTime'].mean().total_seconds(),
                'best_lap_time': driver_laps['LapTime'].min().total_seconds(),
                'tire_usage': len(driver_laps['Compound'].unique()),
                'position': driver_laps['Position'].iloc[-1],
                'points': self.session.get_driver(driver).points
            }
            features.append(driver_data)
            
        return pd.DataFrame(features)

    def prepare_training_data(self, years: List[int], gps: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training data from multiple races.
        
        Args:
            years (List[int]): List of years to include
            gps (List[str]): List of Grand Prix names to include
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: X (features) and y (target) DataFrames
        """
        all_features = []
        
        for year in years:
            for gp in gps:
                try:
                    self.load_race_session(year, gp)
                    features = self.extract_features()
                    features['year'] = year
                    features['gp'] = gp
                    all_features.append(features)
                except Exception as e:
                    logger.warning(f"Could not process {year} {gp}: {str(e)}")
                    continue
        
        if not all_features:
            raise ValueError("No data could be processed")
            
        combined_data = pd.concat(all_features, ignore_index=True)
        
        # Prepare features and target
        X = combined_data.drop(['position', 'points'], axis=1)
        y = combined_data[['position', 'points']]
        
        return X, y

if __name__ == "__main__":
    # Example usage
    processor = F1DataProcessor()
    
    # Process data for 2023 season
    years = [2023]
    gps = ['Monaco', 'Silverstone', 'Spa']
    
    try:
        X, y = processor.prepare_training_data(years, gps)
        print("Features shape:", X.shape)
        print("Target shape:", y.shape)
        
        # Save processed data
        X.to_csv('data/processed/features.csv', index=False)
        y.to_csv('data/processed/targets.csv', index=False)
        
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}") 