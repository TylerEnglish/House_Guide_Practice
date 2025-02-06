import os
import pandas as pd
if __name__ == "__main__":
    from fetch_data import create_column_names
else:
    from scripts.fetch_data import create_column_names

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_data")
DERIVED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "derived_data")
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "boston_housing.data")
PROCESSED_DATA_FILE = os.path.join(DERIVED_DATA_DIR, "processed_boston_housing.csv")

def load_and_clean_data():
    # Load data with predefined column names
    columns = create_column_names()
    df = pd.read_csv(RAW_DATA_FILE, delim_whitespace=True, header=None, names=columns)
    
    # Rename target column from "MEDV" to "price"
    df.rename(columns={"MEDV": "price"}, inplace=True)
    
    # Validate each column
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Impute missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
            else:
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
    
    # As a last resort
    if df.isnull().sum().sum() > 0:
        df.dropna(inplace=True)
    
    return df

def save_processed_data(df):
    if not os.path.exists(DERIVED_DATA_DIR):
        os.makedirs(DERIVED_DATA_DIR)
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_FILE}")

if __name__ == "__main__":
    df = load_and_clean_data()
    save_processed_data(df)
