import os
import pandas as pd
import urllib.request

# Define URL for the Boston Housing dataset
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

# Define local paths
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_data")
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "boston_housing.data")

def fetch_data():
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
    
    print("Downloading Boston Housing data...")
    urllib.request.urlretrieve(DATA_URL, RAW_DATA_FILE)
    print(f"Data saved to {RAW_DATA_FILE}")

def create_column_names():
    # The dataset does not include headers. Define them according to the dataset description.
    columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
        "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B",
        "LSTAT", "MEDV"  # target column; will later be renamed to "price"
    ]
    return columns

if __name__ == "__main__":
    fetch_data()
