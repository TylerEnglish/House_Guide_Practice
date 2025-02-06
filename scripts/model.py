import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
if __name__ == "__main__":
    from data_engineering import build_preprocessor
else:
    from scripts.data_engineering import build_preprocessor

# File paths
DERIVED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "derived_data")
ENGINEERED_DATA_FILE = os.path.join(DERIVED_DATA_DIR, "engineered_boston_housing.csv")

# Model folder
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

MODEL_PIPELINE_FILE = os.path.join(MODELS_DIR, "model_pipeline.pkl")
MODEL_FILES = {
    "voting": os.path.join(MODELS_DIR, "voting_model.pkl"),
    "rf": os.path.join(MODELS_DIR, "rf_model.pkl"),
    "gbr": os.path.join(MODELS_DIR, "gbr_model.pkl"),
    "lr": os.path.join(MODELS_DIR, "lr_model.pkl"),
}

EVAL_METRICS_FILE = os.path.join(MODELS_DIR, "evaluation_metrics.txt")

def load_data():
    df = pd.read_csv(ENGINEERED_DATA_FILE)
    X = df.drop("price", axis=1)
    y = df["price"]
    return X, y

def train_models():
    X, y = load_data()
    # Split data into training and test sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build preprocessor using same logic as in data_engineering.py
    preprocessor = build_preprocessor(X_train)
    
    # Define individual models
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
    lr = LinearRegression()
    
    # Create pipelines for individual models
    pipeline_rf = Pipeline(steps=[("preprocessor", preprocessor), ("model", rf)])
    pipeline_gbr = Pipeline(steps=[("preprocessor", preprocessor), ("model", gbr)])
    pipeline_lr = Pipeline(steps=[("preprocessor", preprocessor), ("model", lr)])
    
    # Ensemble model using VotingRegressor
    voting = VotingRegressor(estimators=[
        ("rf", rf),
        ("gbr", gbr),
        ("lr", lr)
    ])
    pipeline_voting = Pipeline(steps=[("preprocessor", preprocessor), ("model", voting)])
    
    # Train and evaluate each pipeline
    models = {
        "rf": pipeline_rf,
        "gbr": pipeline_gbr,
        "lr": pipeline_lr,
        "voting": pipeline_voting
    }
    
    metrics = {}
    
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        metrics[name] = {"MAE": mae, "R2": r2}
        print(f"{name} model MAE: {mae:.2f}, R2: {r2:.2f}")
        # Save the model pipeline individually
        joblib.dump(pipe, MODEL_FILES[name])
        print(f"Saved {name} model to {MODEL_FILES[name]}")
    
    # Save the voting ensemble
    joblib.dump(pipeline_voting, MODEL_PIPELINE_FILE)
    print(f"Saved voting ensemble pipeline to {MODEL_PIPELINE_FILE}")
    
    # Save evaluation metrics
    with open(EVAL_METRICS_FILE, "w") as f:
        for name, met in metrics.items():
            f.write(f"Model: {name}\n")
            f.write(f"  MAE: {met['MAE']:.2f}\n")
            f.write(f"  R2: {met['R2']:.2f}\n")
            f.write("\n")
    print(f"Saved evaluation metrics to {EVAL_METRICS_FILE}")

if __name__ == "__main__":
    train_models()
