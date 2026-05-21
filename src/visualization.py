import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set professional visualization styles
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Custom Color Palette
# Churn: Yes -> Coral (#E63946), No -> Steel Blue (#457B9D)
CHURN_PALETTE = {1: '#E63946', 0: '#457B9D', 'Yes': '#E63946', 'No': '#457B9D', '1': '#E63946', '0': '#457B9D'}
NEUTRAL_COLOR = '#2A9D8F'

CHARTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "charts"))

def ensure_charts_dir():
    """Ensures that the output charts directory exists."""
    if not os.path.exists(CHARTS_DIR):
        os.makedirs(CHARTS_DIR)
        print(f"Created charts directory: {CHARTS_DIR}")

def save_and_close(filename):
    """Saves the current plot and closes the figure to free memory."""
    ensure_charts_dir()
    filepath = os.path.join(CHARTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved chart: {filepath}")

def plot_churn_distribution(df):
    """Plots and saves the Churn distribution (Countplot)."""
    plt.figure(figsize=(6, 5))
    
    # Calculate percentage
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    
    ax = sns.countplot(x='Churn', hue='Churn', data=df, palette=CHURN_PALETTE, legend=False)
    plt.title('Customer Churn Distribution', pad=15, weight='bold')
    plt.xlabel('Churn Status', labelpad=10)
    plt.ylabel('Number of Customers', labelpad=10)
    plt.xticks([0, 1], ['No Churn (Active)', 'Churned (Left)'])
    
    # Annotate with values and percentages
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        pct = churn_pct.iloc[i]
        ax.annotate(f'{height:,}\n({pct:.1f}%)', 
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', color='white', 
                    weight='bold', fontsize=11)
        
    save_and_close('churn_distribution.png')

def plot_contract_types(df):
    """Plots and saves Contract Types vs Churn."""
    plt.figure(figsize=(8, 5))
    
    # We want to see Churn percentage by contract type
    ax = sns.countplot(x='Contract', hue='Churn', data=df, palette=CHURN_PALETTE)
    plt.title('Churn Rate by Contract Type', pad=15, weight='bold')
    plt.xlabel('Contract Type', labelpad=10)
    plt.ylabel('Customer Count', labelpad=10)
    
    legend_labels = ['Active (No)', 'Churned (Yes)']
    plt.legend(title='Status', labels=legend_labels, loc='upper right')
    
    # Annotate counts
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points')
            
    save_and_close('contract_vs_churn.png')

def plot_monthly_charges(df):
    """Plots and saves Monthly Charges distribution by Churn status (Histogram/KDE)."""
    plt.figure(figsize=(10, 5))
    
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, 
                 element='step', stat='density', common_norm=False, 
                 palette=CHURN_PALETTE, alpha=0.5)
    
    plt.title('Monthly Charges Distribution by Churn Status', pad=15, weight='bold')
    plt.xlabel('Monthly Charges ($)', labelpad=10)
    plt.ylabel('Density', labelpad=10)
    
    legend_labels = ['Active (No)', 'Churned (Yes)']
    plt.legend(title='Status', labels=legend_labels, loc='upper right')
    
    save_and_close('monthly_charges_distribution.png')

def plot_tenure_vs_churn(df):
    """Plots and saves Tenure vs Churn (Boxplot and KDE)."""
    plt.figure(figsize=(10, 5))
    
    sns.boxplot(x='Churn', y='tenure', hue='Churn', data=df, palette=CHURN_PALETTE, legend=False)
    plt.title('Customer Tenure vs Churn Status', pad=15, weight='bold')
    plt.xlabel('Churn Status', labelpad=10)
    plt.ylabel('Tenure (Months)', labelpad=10)
    plt.xticks([0, 1], ['Active', 'Churned'])
    
    save_and_close('tenure_boxplot.png')

def plot_internet_service(df):
    """Plots and saves Internet Service Type vs Churn."""
    plt.figure(figsize=(8, 5))
    
    # Grouped countplot
    ax = sns.countplot(x='InternetService', hue='Churn', data=df, palette=CHURN_PALETTE)
    plt.title('Churn by Internet Service Type', pad=15, weight='bold')
    plt.xlabel('Internet Service Provider', labelpad=10)
    plt.ylabel('Customer Count', labelpad=10)
    
    legend_labels = ['Active (No)', 'Churned (Yes)']
    plt.legend(title='Status', labels=legend_labels, loc='upper right')
    
    # Annotate counts
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points')
            
    save_and_close('internet_service_vs_churn.png')

def plot_payment_methods(df):
    """Plots and saves Payment Method vs Churn."""
    plt.figure(figsize=(10, 6))
    
    ax = sns.countplot(y='PaymentMethod', hue='Churn', data=df, palette=CHURN_PALETTE)
    plt.title('Churn by Payment Method', pad=15, weight='bold')
    plt.xlabel('Customer Count', labelpad=10)
    plt.ylabel('Payment Method', labelpad=10)
    
    legend_labels = ['Active (No)', 'Churned (Yes)']
    plt.legend(title='Status', labels=legend_labels, loc='lower right')
    
    # Annotate counts
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(f'{int(width)}', 
                        (width, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=9, xytext=(3, 0),
                        textcoords='offset points')
            
    save_and_close('payment_method_vs_churn.png')

def plot_correlation_matrix(df):
    """Plots and saves Correlation Matrix Heatmap for numeric columns."""
    plt.figure(figsize=(8, 6))
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Compute correlation
    corr = numeric_df.corr()
    
    # Generate a mask for the upper triangle (optional, but looks cleaner)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=-1, vmax=1, square=True, linewidths=.5, 
                cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix of Numerical Features', pad=15, weight='bold')
    
    save_and_close('correlation_matrix.png')

def plot_confusion_matrix(cm, model_name):
    """Plots and saves the Confusion Matrix for a specific model."""
    plt.figure(figsize=(6, 5))
    
    # Standardize label ordering
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Active (0)', 'Churned (1)'],
                yticklabels=['Active (0)', 'Churned (1)'])
    
    plt.title(f'Confusion Matrix: {model_name}', pad=15, weight='bold')
    plt.xlabel('Predicted Label', labelpad=10)
    plt.ylabel('True Label', labelpad=10)
    
    # Clean model name for file system
    clean_name = model_name.lower().replace(' ', '_')
    save_and_close(f'confusion_matrix_{clean_name}.png')

def generate_all_eda_plots(df):
    """Convenience function to run and save all EDA visualizations."""
    print("Generating EDA Plots...")
    plot_churn_distribution(df)
    plot_contract_types(df)
    plot_monthly_charges(df)
    plot_tenure_vs_churn(df)
    plot_internet_service(df)
    plot_payment_methods(df)
    plot_correlation_matrix(df)
    print("All EDA Plots generated successfully.")
