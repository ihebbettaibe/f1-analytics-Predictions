from src.inference.predict import F1Predictor
from src.utils.helpers import format_predictions, save_predictions

def main():
    # Initialize the predictor
    predictor = F1Predictor()
    
    # Make predictions for a specific race
    year = 2023
    gp = 'Monaco'  # You can change this to any Grand Prix
    
    try:
        # Get predictions
        predictions = predictor.predict_race(year, gp)
        
        # Format predictions into a DataFrame
        pred_df = format_predictions(predictions)
        
        # Print predictions
        print(f"\nPredictions for {year} {gp}:")
        print("=" * 50)
        for _, row in pred_df.iterrows():
            print(f"Driver: {row['driver']}")
            print(f"Predicted Position: {row['predicted_position']}")
            print(f"Predicted Points: {row['predicted_points']}")
            print("-" * 30)
        
        # Save predictions to CSV
        filename = save_predictions(pred_df)
        print(f"\nPredictions saved to: {filename}")
        
        # Evaluate prediction accuracy
        metrics = predictor.evaluate_predictions(year, gp)
        print("\nPrediction Accuracy:")
        print(f"Mean Position Error: {metrics['mean_position_error']:.2f}")
        print(f"Mean Points Error: {metrics['mean_points_error']:.2f}")
        
    except Exception as e:
        print(f"Error making predictions: {str(e)}")

if __name__ == "__main__":
    main() 