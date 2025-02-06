import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

DERIVED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "derived_data")
PROCESSED_DATA_FILE = os.path.join(DERIVED_DATA_DIR, "processed_boston_housing.csv")
ENGINEERED_DATA_FILE = os.path.join(DERIVED_DATA_DIR, "engineered_boston_housing.csv")
PREPROCESSOR_FILE = os.path.join(DERIVED_DATA_DIR, "preprocessor.pkl")

def build_preprocessor(df):
    # Designate categorical features
    categorical_features = ['CHAS', 'RAD']
    # Numeric features are all other columns except the target 'price'
    numeric_features = [col for col in df.columns if col not in categorical_features + ['price']]
    
    # Pipeline for numeric features
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Pipeline for categorical features
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    
    return preprocessor

def feature_engineering(df):
    # Handle missing values for original features only
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df
def process_and_save():
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df = feature_engineering(df)
    
    preprocessor = build_preprocessor(df)
    X = df.drop("price", axis=1)
    preprocessor.fit(X)
    
    import joblib
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    print(f"Preprocessor saved to {PREPROCESSOR_FILE}")
    
    df.to_csv(ENGINEERED_DATA_FILE, index=False)
    print(f"Engineered data saved to {ENGINEERED_DATA_FILE}")

if __name__ == "__main__":
    process_and_save()
