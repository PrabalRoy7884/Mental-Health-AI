# Mental Health Treatment Predictor (Streamlit) 🧠

Predict whether an individual is likely to seek mental health treatment using a trained ML pipeline. The app in `app.py` provides an interactive form, model information, and batch CSV predictions.

## What’s in the app (app.py)
- Home: project overview and quick stats
- Model Information: shows model type, top feature importance, and a comparison chart
  - Metrics are read from `data/processed/model_comparison_results.csv`
- Make Prediction: interactive form with loading spinner, confidence scores, probability bar chart, key factors, and recommendations
- Batch Prediction: upload a CSV, get predictions for all rows, summary stats, charts, and download the results

## Setup
1) Create and activate a virtual environment (Windows PowerShell)
```bash
python -m venv venv
./venv/Scripts/Activate.ps1
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Run the Streamlit app
```bash
streamlit run app.py
```

## Expected files and folders
```
.
├─ app.py
├─ src/
│  ├─ data/
│  ├─ models/
│  └─ utils/
├─ data/
│  ├─ raw/
│  └─ processed/
│     └─ model_comparison_results.csv     # Shown in Model Information
├─ models/                                 # Trained artifacts used by predictor
│  ├─ best_model.pkl
│  ├─ feature_scaler.pkl
│  ├─ feature_selector.pkl
│  ├─ label_encoders.pkl
│  ├─ target_encoder.pkl
│  └─ selected_features.json
├─ prediction.csv                           # Example input for Batch Prediction
├─ requirements.txt
```

## Using Batch Prediction
- Prepare a CSV with the same columns as the prediction form (see app UI)
- Upload on the “Batch Prediction” page
- View results, summary metrics, charts, and download `mental_health_predictions.csv`

## Model Performance and Comparison
- The app reads `data/processed/model_comparison_results.csv` and displays:
  - Best model summary (accuracy, precision, recall, F1)
  - A comparison bar chart across models
  - A full results table

Actual model performance results (from your Model Performance page):

| Model                 | Test Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-----------------------|---------------|-----------|--------|----------|---------|
| Naive Bayes           | 0.7421        | 0.7692    | 0.7031 | 0.7347   | 0.7983  |
| SVM                   | 0.7381        | 0.7627    | 0.7031 | 0.7317   | 0.7851  |
| Logistic Regression   | 0.7302        | 0.7679    | 0.6719 | 0.7167   | 0.8218  |
| Random Forest         | 0.7222        | 0.7500    | 0.6797 | 0.7131   | 0.7762  |
| CatBoost              | 0.7222        | 0.7589    | 0.6641 | 0.7083   | 0.7888  |
| XGBoost               | 0.7183        | 0.7395    | 0.6875 | 0.7126   | 0.7539  |
| Gradient Boosting     | 0.7103        | 0.7350    | 0.6719 | 0.7020   | 0.8022  |
| LightGBM              | 0.7103        | 0.7311    | 0.6797 | 0.7045   | 0.7864  |
| Decision Tree         | 0.6548        | 0.6752    | 0.6172 | 0.6449   | 0.6515  |

**Best Model**: Naive Bayes (74.21% accuracy, 79.83% ROC AUC)
**Best ROC AUC**: Logistic Regression (82.20% ROC AUC)
**Ensemble Performance**: Voting Classifier (72.62% accuracy, 80.89% ROC AUC)

How to update the comparison for your results:
- Add or replace rows in `data/processed/model_comparison_results.csv` with columns (header order matters):
  `F1 Score, Precision, ROC AUC, Recall, Test Accuracy`
- The “Model Information” page in `app.py` will automatically refresh the chart and table with your values.

## Notes
- Python 3.10+
- The app uses Plotly for charts and shows a loading spinner while computing
- If any of the model artifacts are missing, the app will show a helpful error message

## Troubleshooting
- Model metrics not shown: make sure `data/processed/model_comparison_results.csv` exists
- Predictions fail to run: check that files in `models/` listed above are present
- For clean installs, you can pin your venv packages: `pip freeze > requirements.txt`

## License
Educational use only. Add your preferred license (e.g., MIT) before publishing.
