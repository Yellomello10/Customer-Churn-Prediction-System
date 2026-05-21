import os
import joblib
import pandas as pd
import numpy as np

# Import modules from src
from src.data_cleaning import load_data, clean_data
from src.feature_engineering import preprocess_data, get_preprocessor
from src.visualization import generate_all_eda_plots
from src.model_training import split_dataset, train_models
from src.evaluation import evaluate_model, compare_models

def setup_directories():
    """Create outputs directories if they do not exist."""
    directories = [
        "outputs/charts",
        "outputs/reports",
        "outputs/models"
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def save_business_insights(df, comparison_df, best_model_name):
    """
    Computes key data aggregates and writes a professional, 
    actionable Business Insights report to outputs/reports/business_insights.txt.
    """
    report_path = "outputs/reports/business_insights.txt"
    print(f"Generating business insights report at: {report_path}")
    
    # 1. Overall Churn Rate
    churn_rate = df['Churn'].mean() * 100
    total_customers = len(df)
    churn_count = df['Churn'].sum()
    
    # 2. Contract Type Impact
    contract_churn = df.groupby('Contract')['Churn'].agg(['count', 'mean'])
    contract_churn['mean'] = contract_churn['mean'] * 100
    
    # 3. Monthly Charges Impact
    avg_monthly_active = df[df['Churn'] == 0]['MonthlyCharges'].mean()
    avg_monthly_churned = df[df['Churn'] == 1]['MonthlyCharges'].mean()
    
    # 4. Tenure Impact (First year vs later years)
    first_year_churn = df[df['tenure'] <= 12]['Churn'].mean() * 100
    long_term_churn = df[df['tenure'] > 24]['Churn'].mean() * 100
    
    # 5. Internet Service Type Impact
    internet_churn = df.groupby('InternetService')['Churn'].agg(['count', 'mean'])
    internet_churn['mean'] = internet_churn['mean'] * 100
    
    # 6. Payment Method Impact
    payment_churn = df.groupby('PaymentMethod')['Churn'].agg(['count', 'mean'])
    payment_churn['mean'] = payment_churn['mean'] * 100

    report_content = f"""================================================================================
CUSTOMER CHURN prediction & BUSINESS INSIGHTS REPORT
================================================================================
Prepared by: Senior Data Scientist / ML Engineer
Status: Completed
Selected Prediction Model: {best_model_name}
================================================================================

1. EXECUTIVE SUMMARY
-------------------
An analysis of {total_customers:,} customers reveals an overall churn rate of {churn_rate:.2f}%. Out of the 
analyzed base, {churn_count:,} customers left the company. The predictive system has selected 
the {best_model_name} pipeline as the most robust model for early detection, enabling targeting 
of high-risk customers with proactive retention campaigns.

2. KEY CHURN DRIVERS & DESCRIPTIVE ANALYSIS
-------------------------------------------

A. Impact of Contract Type (High Significance)
   - Month-to-month contract customers show an extremely high churn rate of {contract_churn.loc['Month-to-month', 'mean']:.2f}% (Count: {contract_churn.loc['Month-to-month', 'count']:,}).
   - One-year contract customers churn rate: {contract_churn.loc['One year', 'mean']:.2f}% (Count: {contract_churn.loc['One year', 'count']:,}).
   - Two-year contract customers show a very low churn rate of {contract_churn.loc['Two year', 'mean']:.2f}% (Count: {contract_churn.loc['Two year', 'count']:,}).
   
   Insight: Contract rigidity is the strongest predictor of customer loyalty. Customers on short-term 
   flexible contracts have very low switching costs and high churn propensity.

B. Financial / Billing Impact
   - Active (No Churn) Customers Average Monthly Bill : ${avg_monthly_active:.2f}
   - Churned Customers Average Monthly Bill            : ${avg_monthly_churned:.2f}
   
   Insight: Customers who churned were paying, on average, ${avg_monthly_churned - avg_monthly_active:.2f} more per month 
   than active customers. High monthly charges strongly correlate with increased churn, suggesting price 
   sensitivity or dissatisfaction with perceived value.

C. Tenure & Relationship Life Cycle
   - Churn rate in the first 12 months (Tenure <= 1 year): {first_year_churn:.2f}%
   - Churn rate for long-term customers (Tenure > 24 months): {long_term_churn:.2f}%
   
   Insight: The first year of the customer journey is the most critical window. Customers who pass 
   the 2-year mark exhibit significantly higher retention, reflecting long-term brand lock-in.

D. Internet Service Risk Profiles
"""
    
    # Add Internet Service Churn details
    for idx, row in internet_churn.iterrows():
        report_content += f"   - {idx} Service: Churn rate is {row['mean']:.2f}% (Total Customers: {row['count']:,})\n"
        
    report_content += """
   Insight: Fiber optic customers churn at a significantly higher rate than DSL or analog customers, 
   despite fiber optic providing faster speeds. This suggests issues with connection stability, customer 
   onboarding, or pricing dissatisfaction specific to the fiber service tier.

E. Payment Method & Friction
"""
    
    # Add Payment Method Churn details
    for idx, row in payment_churn.iterrows():
        report_content += f"   - {idx}: Churn rate is {row['mean']:.2f}% (Total Customers: {row['count']:,})\n"
        
    report_content += f"""
   Insight: Customers paying via Electronic Check churn at an alarmingly high rate compared to auto-pay 
   methods (Credit Card or Bank Transfer). Electronic billing methods require monthly manual action, 
   creating recurring decision points where a customer might choose to cancel their subscription.

3. ACTIONABLE STRATEGIC RECOMMENDATIONS
---------------------------------------
1. CONTRACT MIGRATION CAMPAIGN: Launch targeted marketing campaigns offering incentives (e.g., $5 off 
   per month or 1 month free) to Month-to-month customers if they switch to 1-year or 2-year contracts.
   
2. AUTOPAY SIGN-UP INCENTIVE: Focus on converting Electronic Check users to Credit Card or Bank Transfer 
   autopay methods. Offer a one-time bill credit (e.g., $10) for set-and-forget billing enrollment.
   
3. EARLY-TENURE RETENTION PROGRAM: Develop an aggressive onboarding and check-in system for new customers 
   during their first 6-12 months. Early tech support interventions can prevent churn before patterns set in.
   
4. FIBER OPTIC tier AUDIT: Investigate why Fiber Optic customers have high churn. Ensure support desks are 
   adequately troubleshooting Fiber-specific latency, and check if pricing models are aligned with competitor rates.

5. HIGH-RISK PRICING ADJUSTMENT: Leverage the {best_model_name} to flag customers whose monthly charges 
   cross high thresholds, and offer them loyalty loyalty discounts or add-ons to enhance perceived value.

================================================================================
END OF REPORT
================================================================================
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Report written successfully to {report_path}.")

def main():
    print("================================================================================")
    print("                STARTING CUSTOMER CHURN PREDICTION PIPELINE                     ")
    print("================================================================================")
    
    # 1. Setup folders
    setup_directories()
    
    # 2. Load and Clean data
    raw_df = load_data()
    
    # Display basic descriptive info
    print("\nDataset Basic Info:")
    print("-" * 30)
    print(f"Number of rows: {raw_df.shape[0]}")
    print(f"Number of columns: {raw_df.shape[1]}")
    print(f"Number of duplicate rows: {raw_df.duplicated().sum()}")
    print(f"Missing values per column:\n{raw_df.isnull().sum()[raw_df.isnull().sum() > 0]}")
    
    cleaned_df = clean_data(raw_df)
    
    # 3. Exploratory Data Analysis (Save charts)
    generate_all_eda_plots(cleaned_df)
    
    # 4. Feature Preprocessing
    X, y, cat_cols, num_cols = preprocess_data(cleaned_df)
    preprocessor = get_preprocessor(cat_cols, num_cols)
    
    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # 6. Train Models
    pipelines = train_models(X_train, y_train, preprocessor)
    
    # 7. Evaluate and Compare Models
    eval_results = []
    for name, model_pipeline in pipelines.items():
        metrics = evaluate_model(model_pipeline, X_test, y_test, name)
        eval_results.append(metrics)
        
    comparison_df, best_model_name = compare_models(eval_results)
    
    # Save the comparison dataframe to outputs/reports
    comparison_df.to_csv("outputs/reports/model_comparison.csv", index=False)
    print("Saved model comparison table to 'outputs/reports/model_comparison.csv'")
    
    # 8. Save Best Model
    best_pipeline = pipelines[best_model_name]
    model_save_path = f"outputs/models/best_model.joblib"
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved the best trained model pipeline to: {model_save_path}")
    
    # 9. Generate Business Insights Report
    save_business_insights(cleaned_df, comparison_df, best_model_name)
    
    print("\n================================================================================")
    print("            CUSTOMER CHURN PREDICTION PIPELINE COMPLETED SUCCESSFULLY!           ")
    print("================================================================================")

if __name__ == "__main__":
    main()
