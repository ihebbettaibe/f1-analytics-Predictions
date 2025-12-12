from src.data_pipeline.data_processing import F1DataProcessor
from src.models.train import F1ModelTrainer
import pandas as pd
import joblib
import os

def main():
    # Example: Train on multiple races (add more as needed)
    races = [
        {'year': 2023, 'gp': 'Monaco'},
        {'year': 2023, 'gp': 'Spanish'},
        {'year': 2023, 'gp': 'British'},
    ]
    all_features = []
    all_targets = []
    for race in races:
        processor = F1DataProcessor()
        processor.load_race_session(race['year'], race['gp'])
        features = processor.extract_features()
        # Target: You may want to adjust this depending on your prediction goal
        # Here, we assume 'positionOrder' is the target (add to extract_features if missing)
        if 'positionOrder' not in features.columns:
            print(f"positionOrder not found in features for {race['gp']}")
            continue
        targets = features[['positionOrder']]
        all_features.append(features)
        all_targets.append(targets)
    if not all_features:
        print("No data to train on.")
        return
    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_targets, ignore_index=True)
    trainer = F1ModelTrainer()
    results = trainer.train_model(X, y)
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(trainer.model, 'models/f1_prediction_model.joblib')
    joblib.dump(trainer.scaler, 'models/scaler.joblib')
    print("Model and scaler saved to 'models/' directory.")
    print("Training results:", results)

if __name__ == "__main__":
    main()
