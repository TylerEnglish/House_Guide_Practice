import os
import pandas as pd
import numpy as np
import joblib
import shap

# Updated model paths using the new 'models' folder
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PIPELINE_FILE = os.path.join(MODELS_DIR, "model_pipeline.pkl")
MODEL_FILES = {
    "voting": os.path.join(MODELS_DIR, "voting_model.pkl"),
    "rf": os.path.join(MODELS_DIR, "rf_model.pkl"),
    "gbr": os.path.join(MODELS_DIR, "gbr_model.pkl"),
    "lr": os.path.join(MODELS_DIR, "lr_model.pkl"),
}

def load_model(model_name="voting"):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Model {model_name} not recognized. Choose from {list(MODEL_FILES.keys())}")
    return joblib.load(MODEL_FILES[model_name])

def predict_price_from_features(features, model_name="voting"):
    # Convert features dict to a DataFrame (one row)
    input_df = pd.DataFrame([features])
    model = load_model(model_name)
    prediction = model.predict(input_df)
    return prediction[0]

def recommend_house(budget, model_name="voting"):
    # Rule-based recommendations
    recommended_profile = {
        "estimated_area": np.clip(budget / 300, 500, 3000), 
        "rooms": int(np.clip(budget / 100000, 2, 6)),
        "bedrooms": int(np.clip(budget / 150000, 1, 5)),
        "bathrooms": int(np.clip(budget / 200000, 1, 4))
    }
    
    # Prepare a baseline feature set
    features_for_prediction = {
        "CRIM": 0.1,
        "ZN": 18.0,
        "INDUS": 2.5,
        "CHAS": 0,
        "NOX": 0.5,
        "RM": recommended_profile["rooms"] * 5,
        "AGE": 45,
        "DIS": 4.0,
        "RAD": 4,
        "TAX": 300,
        "PTRATIO": 15,
        "B": 390,
        "LSTAT": 4
    }
    
    # Refine the recommendation.
    predicted_price = predict_price_from_features(features_for_prediction, model_name=model_name)
    recommended_profile["predicted_price"] = predicted_price
    return recommended_profile

def get_similar_houses(budget, num=5):
    np.random.seed(42)
    data = {
        "area": np.random.randint(800, 3000, size=num),
        "price": np.random.randint(int(budget - 50000), int(budget + 50000), size=num),
        "bedrooms": np.random.randint(1, 6, size=num),
        "location": np.random.choice(["Downtown", "Suburb", "Countryside"], size=num)
    }
    return pd.DataFrame(data)

def explain_prediction(features, model_name="voting"):
    """
    Generate SHAP values using the raw input data (the original 13 features) only.
    This avoids the enormous feature expansion from the preprocessor.
    """
    # Create a DataFrame from the raw features.
    input_df = pd.DataFrame([features])
    
    # Load the model (a pipeline that includes a preprocessor and model)
    model = load_model(model_name)
    
    # Define a custom prediction function that accepts a 2D NumPy array,
    # converts it back into a DataFrame with the original feature names,
    # and then calls the pipeline's predict method.
    predict_func = lambda X: model.predict(pd.DataFrame(X, columns=input_df.columns))
    
    # Build the SHAP explainer on the raw input.
    explainer = shap.Explainer(predict_func, input_df)
    shap_values = explainer(input_df)
    
    # We limit the explanation to the original features.
    # (Assuming the number of raw features is the same as the length of `features`.)
    explanation = shap.Explanation(
        values = shap_values.values[0][:len(features)],  # keep only as many values as raw features
        base_values = (shap_values.base_values[0]
                       if isinstance(shap_values.base_values, (list, np.ndarray))
                       else shap_values.base_values),
        data = input_df.iloc[0].values,
        feature_names = list(input_df.columns)
    )
    
    return explanation, explanation.base_values

if __name__ == "__main__":
    # Define example input features (base features only)
    example_features = {
        "CRIM": 0.1,
        "ZN": 18.0,
        "INDUS": 2.5,
        "CHAS": 0,
        "NOX": 0.5,
        "RM": 6.0,
        "AGE": 45,
        "DIS": 4.0,
        "RAD": 4,
        "TAX": 300,
        "PTRATIO": 15,
        "B": 390,
        "LSTAT": 4
    }
    
    # Price prediction 
    price = predict_price_from_features(example_features, model_name="voting")
    print(f"Predicted price: {price:.2f}")
    
    # House recommendation 
    budget = 300000
    recommendation = recommend_house(budget, model_name="voting")
    print("House Recommendation:")
    for key, value in recommendation.items():
        print(f"  {key}: {value}")
    
    # Simulated similar houses
    similar_df = get_similar_houses(budget)
    print("Similar Houses:")
    print(similar_df)
    
    # SHAP explanation 
    explanation, base_val = explain_prediction(example_features, model_name="voting")
    print("SHAP Explanation:")
    print("Base Value:", base_val)
    print("SHAP Values:", explanation.values)
