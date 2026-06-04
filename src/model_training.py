import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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
    Constructs and trains three pipeline models using GridSearchCV:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. Random Forest Classifier
    
    Each pipeline couples preprocessing with the estimator to avoid data leakage.
    Optimizes models based on F1 Score.
    """
    print("Training models with GridSearchCV optimization...")
    
    # Define baseline classifiers
    base_models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000, 
            solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42, 
            n_jobs=-1
        )
    }
    
    # Define hyperparameter search spaces (with classifier__ prefix for pipeline compatibility)
    param_grids = {
        'Logistic Regression': {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__class_weight': ['balanced', None]
        },
        'Decision Tree': {
            'classifier__max_depth': [4, 6, 8, 10],
            'classifier__min_samples_leaf': [2, 4, 8]
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [6, 8, 10],
            'classifier__min_samples_leaf': [2, 4, 6],
            'classifier__class_weight': ['balanced', None]
        }
    }
    
    trained_pipelines = {}
    
    # Construct, tune, and train pipeline for each model
    for name, clf in base_models.items():
        print(f"\nGrid Search Tuning for {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Grid Search with F1-Score optimization
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
        
        # Store the best estimator pipeline
        trained_pipelines[name] = grid_search.best_estimator_
        print(f"{name} pipeline optimized and trained successfully.")
        
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
