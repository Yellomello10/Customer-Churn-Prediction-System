import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_tenure_groups(df):
    """
    Creates a new feature 'TenureGroup' by binning the 'tenure' column.
    Bins:
    - 0-12 months
    - 12-24 months
    - 24-48 months
    - 48-60 months
    - 60+ months
    """
    df_engineered = df.copy()
    
    labels = ["0-12 Months", "12-24 Months", "24-48 Months", "48-60 Months", "60+ Months"]
    bins = [-1, 12, 24, 48, 60, 100]
    
    df_engineered['TenureGroup'] = pd.cut(df_engineered['tenure'], bins=bins, labels=labels)
    print("Created feature 'TenureGroup' successfully.")
    return df_engineered

def get_preprocessor(categorical_features, numerical_features):
    """
    Creates a scikit-learn ColumnTransformer for scaling and encoding.
    Uses:
    - StandardScaler for numerical columns.
    - OneHotEncoder (with drop='first') for categorical columns.
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

def preprocess_data(df):
    """
    Prepares features (X) and target (y) for machine learning.
    Steps:
    - Drops customerID.
    - Creates tenure groups.
    - Separates independent variables (X) and dependent variable (y).
    """
    # Create tenure groups
    df_proc = create_tenure_groups(df)
    
    # Drop customerID if exists
    if 'customerID' in df_proc.columns:
        df_proc = df_proc.drop(columns=['customerID'])
        print("Dropped 'customerID' column.")
        
    # Separate target
    if 'Churn' in df_proc.columns:
        X = df_proc.drop(columns=['Churn'])
        y = df_proc['Churn']
    else:
        X = df_proc
        y = None
        
    # Define feature lists
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'TenureGroup'
    ]
    
    # Ensure all specified categorical features are present
    categorical_features = [col for col in categorical_features if col in X.columns]
    
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    numerical_features = [col for col in numerical_features if col in X.columns]
    
    print(f"Numerical features to scale: {numerical_features}")
    print(f"Categorical features to encode: {categorical_features}")
    
    return X, y, categorical_features, numerical_features

if __name__ == "__main__":
    from data_cleaning import load_data, clean_data
    
    df = load_data()
    df_clean = clean_data(df)
    
    X, y, cat_cols, num_cols = preprocess_data(df_clean)
    preprocessor = get_preprocessor(cat_cols, num_cols)
    
    # Test transformation
    X_trans = preprocessor.fit_transform(X)
    print(f"\nFeature dimensions after preprocessing: {X_trans.shape}")
