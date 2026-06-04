import os
import pandas as pd
import numpy as np
import pytest
from src.data_cleaning import clean_data

def test_clean_data_charges_imputation():
    # Create sample dataframe with empty space in TotalCharges
    data = {
        'customerID': ['1', '2', '3'],
        'tenure': [0, 10, 0],
        'TotalCharges': [' ', '120.5', '  '],
        'Churn': ['No', 'Yes', 'No']
    }
    df = pd.DataFrame(data)
    
    # Run clean data
    cleaned_df = clean_data(df)
    
    # Assert values
    assert cleaned_df.shape[0] == 3
    # Empty charges should be imputed to 0.0
    assert cleaned_df.loc[0, 'TotalCharges'] == 0.0
    assert cleaned_df.loc[2, 'TotalCharges'] == 0.0
    assert cleaned_df.loc[1, 'TotalCharges'] == 120.5

def test_clean_data_churn_mapping():
    data = {
        'customerID': ['1', '2'],
        'tenure': [5, 12],
        'TotalCharges': ['50.0', '150.0'],
        'Churn': ['No', 'Yes']
    }
    df = pd.DataFrame(data)
    
    cleaned_df = clean_data(df)
    
    # Check mapping
    assert cleaned_df.loc[0, 'Churn'] == 0
    assert cleaned_df.loc[1, 'Churn'] == 1

def test_clean_data_duplicates():
    # Data with a duplicate row
    data = {
        'customerID': ['1', '2', '1'],
        'tenure': [5, 12, 5],
        'TotalCharges': ['50.0', '150.0', '50.0'],
        'Churn': ['No', 'Yes', 'No']
    }
    df = pd.DataFrame(data)
    
    cleaned_df = clean_data(df)
    
    # Duplicate row should be removed (shape should be 2, not 3)
    assert cleaned_df.shape[0] == 2
