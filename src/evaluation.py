import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
try:
    from src.visualization import plot_confusion_matrix
except ModuleNotFoundError:
    from visualization import plot_confusion_matrix

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates a single model on test data.
    Computes: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    Saves the confusion matrix visualization.
    """
    print(f"\nEvaluating Model: {model_name}")
    print("=" * 40)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print metrics
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Active', 'Churned'])
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Save Confusion Matrix plot
    plot_confusion_matrix(cm, model_name)
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def compare_models(eval_results):
    """
    Compares all trained models in a pandas DataFrame and selects the best model.
    Selection is based on F1-Score (balances Precision and Recall for churn).
    """
    comparison_df = pd.DataFrame(eval_results)
    
    print("\nModel Comparison Table:")
    print("=" * 60)
    # Format for display
    print(comparison_df.to_string(index=False, formatters={
        'Accuracy': '{:,.4%}'.format,
        'Precision': '{:,.4%}'.format,
        'Recall': '{:,.4%}'.format,
        'F1 Score': '{:,.4%}'.format
    }))
    print("=" * 60)
    
    # Find best model based on F1 Score
    best_row = comparison_df.loc[comparison_df['F1 Score'].idxmax()]
    best_model_name = best_row['Model']
    best_f1 = best_row['F1 Score']
    
    print(f"\nBest Model Selected: {best_model_name} with F1-Score: {best_f1:.4%}")
    return comparison_df, best_model_name
