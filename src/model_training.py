import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Dataset split: Train shape = {X_train.shape}, Test shape = {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train, preprocessor):
    """
    Constructs and trains three pipeline models:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. Random Forest Classifier
    
    Each pipeline couples preprocessing with the estimator to avoid data leakage.
    """
    print("Training models...")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000, 
            solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, 
            max_depth=6, 
            min_samples_leaf=4
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42, 
            n_estimators=100, 
            max_depth=10, 
            min_samples_leaf=4, 
            n_jobs=-1
        )
    }
    
    trained_pipelines = {}
    
    # Construct and train pipeline for each model
    for name, clf in models.items():
        print(f"Training {name} pipeline...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Fit the entire pipeline
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        print(f"{name} trained successfully.")
        
    return trained_pipelines

if __name__ == "__main__":
    from data_cleaning import load_data, clean_data
    from feature_engineering import preprocess_data, get_preprocessor
    
    # Test training
    df = load_data()
    df_clean = clean_data(df)
    X, y, cat_cols, num_cols = preprocess_data(df_clean)
    preprocessor = get_preprocessor(cat_cols, num_cols)
    
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    pipelines = train_models(X_train, y_train, preprocessor)
    
    for name, pipe in pipelines.items():
        train_score = pipe.score(X_train, y_train)
        test_score = pipe.score(X_test, y_test)
        print(f"{name}: Train Acc = {train_score:.4f}, Test Acc = {test_score:.4f}")
