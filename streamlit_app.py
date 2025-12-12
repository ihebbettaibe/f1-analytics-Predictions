import streamlit as st
import pandas as pd
import os
from src.data_pipeline.data_processing import F1DataProcessor
from src.models.train import F1ModelTrainer
from src.inference.predict import F1Predictor
from src.utils.helpers import format_predictions, save_predictions
import joblib

st.set_page_config(page_title="F1 Race Prediction", layout="wide")
st.title("🏁 F1 Race Prediction Pipeline")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Train Model", "Predict Race"])

# Helper: List available races
def get_available_races():
    return [
        {"year": 2023, "gp": "Monaco"},
        {"year": 2023, "gp": "Spanish"},
        {"year": 2023, "gp": "British"},
    ]

if page == "Train Model":
    st.header("Train Model on Historical Data")
    races = get_available_races()
    selected = st.multiselect(
        "Select races to train on:",
        options=[f"{r['year']} {r['gp']}" for r in races],
        default=[f"{r['year']} {r['gp']}" for r in races]
    )
    if st.button("Train Model"):
        all_features = []
        all_targets = []
        for race in races:
            label = f"{race['year']} {race['gp']}"
            if label not in selected:
                continue
            processor = F1DataProcessor()
            processor.load_race_session(race['year'], race['gp'])
            features = processor.extract_features()
            if 'positionOrder' not in features.columns:
                st.warning(f"positionOrder not found in features for {label}")
                continue
            targets = features[['positionOrder']]
            all_features.append(features)
            all_targets.append(targets)
        if not all_features:
            st.error("No data to train on.")
        else:
            X = pd.concat(all_features, ignore_index=True)
            y = pd.concat(all_targets, ignore_index=True)
            trainer = F1ModelTrainer()
            results = trainer.train_model(X, y)
            os.makedirs('models', exist_ok=True)
            joblib.dump(trainer.model, 'models/f1_prediction_model.joblib')
            joblib.dump(trainer.scaler, 'models/scaler.joblib')
            st.success("Model and scaler saved to 'models/' directory.")
            st.write("Training results:", results)

elif page == "Predict Race":
    st.header("Predict Race Results")
    races = get_available_races()
    race_options = [f"{r['year']} {r['gp']}" for r in races]
    selected_race = st.selectbox("Select race to predict:", race_options)
    if st.button("Predict"):
        year, gp = selected_race.split()
        predictor = F1Predictor()
        predictions = predictor.predict_race(int(year), gp)
        pred_df = format_predictions(predictions)
        st.subheader(f"Predictions for {year} {gp}")
        st.dataframe(pred_df)
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f'predictions_{year}_{gp}.csv',
            mime='text/csv',
        )
        # Optionally, show evaluation if available
        if hasattr(predictor, 'evaluate_predictions'):
            try:
                metrics = predictor.evaluate_predictions(int(year), gp)
                st.write("\nPrediction Accuracy:")
                st.write(metrics)
            except Exception:
                st.info("No ground truth available for evaluation.")
