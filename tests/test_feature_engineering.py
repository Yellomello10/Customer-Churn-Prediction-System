import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import create_tenure_groups, preprocess_data, get_preprocessor

def test_create_tenure_groups():
    # Sample dataframe with various tenures
    data = {
        'tenure': [0, 12, 13, 24, 25, 48, 50, 60, 61, 72]
    }
    df = pd.DataFrame(data)
    
    df_grouped = create_tenure_groups(df)
    
    # Assert correct groups
    expected = [
        "0-12 Months", "0-12 Months", 
        "12-24 Months", "12-24 Months", 
        "24-48 Months", "24-48 Months", 
        "48-60 Months", "48-60 Months", 
        "60+ Months", "60+ Months"
    ]
    assert list(df_grouped['TenureGroup']) == expected

def test_preprocess_data_splitting():
    data = {
        'customerID': ['123-ABC', '456-DEF'],
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'tenure': [5, 24],
        'MonthlyCharges': [29.85, 56.95],
        'TotalCharges': [29.85, 1889.50],
        'Churn': [0, 1]
    }
    df = pd.DataFrame(data)
    
    X, y, cat_cols, num_cols = preprocess_data(df)
    
    # CustomerID should be dropped
    assert 'customerID' not in X.columns
    # Target Churn should be separated into y
    assert 'Churn' not in X.columns
    assert list(y) == [0, 1]
    # TenureGroup should have been created
    assert 'TenureGroup' in X.columns
    
    # Check feature classification
    assert 'gender' in cat_cols
    assert 'tenure' in num_cols
    assert len(num_cols) == 3 # tenure, MonthlyCharges, TotalCharges

def test_preprocessor_pipeline():
    # Create simple preprocessed data
    X = pd.DataFrame({
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'tenure': [10.0, 20.0],
        'MonthlyCharges': [50.0, 100.0]
    })
    
    cat_cols = ['gender', 'SeniorCitizen']
    num_cols = ['tenure', 'MonthlyCharges']
    
    preprocessor = get_preprocessor(cat_cols, num_cols)
    
    # Fit and transform
    X_trans = preprocessor.fit_transform(X)
    
    # Check shape:
    # 2 numerical features -> 2 scaled features
    # gender (Male/Female) -> OneHotEncoded (drop='first') -> 1 feature
    # SeniorCitizen (0/1) -> OneHotEncoded (drop='first') -> 1 feature
    # Total features: 4 columns
    assert X_trans.shape == (2, 4)
