import os
import pandas as pd
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

def download_dataset():
    """Downloads the Telco Customer Churn dataset if not already present."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")
        
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_PATH)
            print(f"Dataset successfully downloaded and saved to: {DATA_PATH}")
        except Exception as e:
            print(f"Error downloading the dataset: {e}")
            raise e
    else:
        print(f"Dataset already exists at: {DATA_PATH}")

def load_data():
    """Loads the dataset into a pandas DataFrame."""
    download_dataset()
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def clean_data(df):
    """
    Cleans the Telco Customer Churn DataFrame:
    - Handles missing/empty values in 'TotalCharges'.
    - Converts 'TotalCharges' to float.
    - Removes duplicate rows.
    - Standardizes target column 'Churn' to binary (0/1).
    """
    df_clean = df.copy()
    
    # 1. Clean TotalCharges: convert empty spaces to NaN and then to 0.0
    # Empty spaces occur where tenure is 0 (new customers)
    total_charges_before = df_clean['TotalCharges'].isnull().sum()
    empty_strings_count = (df_clean['TotalCharges'].str.strip() == '').sum()
    print(f"Found {empty_strings_count} rows with blank/empty 'TotalCharges'.")
    
    # Convert empty spaces to NaN
    df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(r'^\s*$', None, regex=True)
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Fill NaN values with 0.0 for tenure == 0 (since they have not been billed yet)
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)
    print(f"Converted 'TotalCharges' to numeric and imputed empty/missing values with 0.0.")
    
    # 2. Check and remove duplicates
    duplicate_count = df_clean.duplicated().sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate rows. Removing duplicates...")
        df_clean = df_clean.drop_duplicates()
    else:
        print("No duplicate rows found.")
        
    # 3. Clean Churn column: map Yes/No to 1/0
    if 'Churn' in df_clean.columns:
        # Check if Churn is non-numeric (Yes/No string)
        if not pd.api.types.is_numeric_dtype(df_clean['Churn']):
            df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
            print("Mapped target column 'Churn' to binary: Yes -> 1, No -> 0.")
            
    # SeniorCitizen is 0/1, convert it to category for EDA, but keep as int for training
    # No changes needed here, we'll keep it as 0/1.
    
    return df_clean

if __name__ == "__main__":
    # Test loading and cleaning
    df = load_data()
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    df_clean = clean_data(df)
    print("\nCleaned Dataset Info:")
    print(df_clean.info())
    print(f"\nMissing values after cleaning in TotalCharges: {df_clean['TotalCharges'].isnull().sum()}")
