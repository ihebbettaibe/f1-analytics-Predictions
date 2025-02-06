# F1 Race Prediction

## Overview
This project predicts F1 race results using machine learning models trained on historical race data.

## Project Structure
```
.
├── src/
│   ├── data_pipeline/
│   │   ├── data_processing.py  # Data preprocessing and feature engineering
│   ├── models/
│   │   ├── main.py  # Model training and evaluation
│   │   ├── train.py  # Hyperparameter tuning and model training
│   ├── inference/
│   │   ├── predict.py  # Model inference and predictions
│   ├── utils/
│   │   ├── helpers.py  # Utility functions
├── notebooks/
│   ├── exploratory_data_analysis.ipynb  # Initial EDA and data visualization
├── requirements.txt  # Dependencies
├── README.md  # Project documentation
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/f1-race-prediction.git
   cd f1-race-prediction
   ```
2. Create a virtual environment and install dependencies:
   ```sh
   python -m venv new_env
   source new_env/bin/activate  # On Windows use: new_env\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage
### 1. Data Preprocessing
Run the data processing pipeline:
```sh
python -m src.data_pipeline.data_processing
```

### 2. Train the Model
```sh
python -m src.models.main
```

### 3. Make Predictions
```sh
python -m src.inference.predict
```

## Debugging & Troubleshooting
- If you encounter `ModuleNotFoundError`, ensure the script is run from the project root with `python -m`
- Check that the test dataset has the correct features using:
  ```python
  print(X_test.columns)
  ```
- If the model expects different features, check `missing_cols` and `extra_cols` in `predict.py`

## Model Evaluation
- Predictions are saved as a DataFrame with `position` and `result_driver_standing`
- Compare predicted vs actual standings for evaluation

## Contributing
Feel free to submit issues or pull requests to improve this project.

## License
This project is licensed under the MIT License.
