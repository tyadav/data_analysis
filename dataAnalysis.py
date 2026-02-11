"""
Data Analysis & Predictive Modeling Pipeline

This script handles the complete workflow: data ingestion, exploratory analysis, 
predictive modeling, and output generation.
"""

# ============================================================================
# 1. Import Required Libraries
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error
from scipy import stats
import warnings
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ============================================================================
# 2. Load Data from Local File
# ============================================================================

# Configuration - Update these paths
data_file_path = r"C:/Users/TejYadav/OneDrive - kyndryl/Data/Sales by Store.csv"  # Change this to your file path
output_folder = "analysis_output"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load data
try:
    df = pd.read_csv(data_file_path)
    print(f"✓ Data loaded successfully from: {data_file_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic info:")
    print(df.info())
except FileNotFoundError:
    print(f"✗ Error: File '{data_file_path}' not found. Please check the path.")
except Exception as e:
    print(f"✗ Error loading file: {e}")


# ============================================================================
# 3. Exploratory Data Analysis (EDA)
# ============================================================================

# Statistical Summary
print("=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(df.describe())

# Missing Values Analysis
print("\n" + "=" * 50)
print("MISSING VALUES ANALYSIS")
print("=" * 50)
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_values,
    'Percentage': missing_percent
})
print(missing_df[missing_df['Missing_Count'] > 0])

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

# Correlation Matrix
if len(numeric_cols) > 1:
    print("\n" + "=" * 50)
    print("CORRELATION MATRIX")
    print("=" * 50)
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix)
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/01_correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Correlation matrix saved")

# Distribution plots for numeric columns
for col in numeric_cols[:6]:  # Limit to first 6 columns
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(df[col].dropna(), bins=30, edgecolor='black')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df[col].dropna())
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/02_distribution_{col}.png", dpi=300, bbox_inches='tight')
    plt.show()

print("✓ Distribution plots saved")

# ============================================================================
# 3.1 Target Variable & Categorical Features Analysis
# ============================================================================

# Target variable analysis (if numeric)
if numeric_cols:
    target_col_temp = numeric_cols[-1]
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(df[target_col_temp], bins=50, edgecolor='black', color='steelblue')
    plt.xlabel(target_col_temp)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Target Variable: {target_col_temp}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df[target_col_temp])
    plt.ylabel(target_col_temp)
    plt.title(f'Boxplot of Target Variable')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/03_target_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Target variable distribution saved")

# Categorical features analysis
if categorical_cols:
    print("\n" + "=" * 50)
    print("CATEGORICAL FEATURES DISTRIBUTION")
    print("=" * 50)
    
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        plt.figure(figsize=(12, 4))
        
        # Count plot
        value_counts = df[col].value_counts().head(10)
        plt.subplot(1, 2, 1)
        value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f'Top 10 Values in {col}')
        plt.ylabel('Count')
        plt.xlabel(col)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        top_5 = df[col].value_counts().head(5)
        plt.pie(top_5.values, labels=top_5.index, autopct='%1.1f%%', startangle=90)
        plt.title(f'Top 5 Categories in {col}')
        
        plt.tight_layout()
        plt.savefig(f"{output_folder}/03_categorical_{col}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"✓ Categorical features analysis saved")

# ============================================================================
# 4. Data Preprocessing and Cleaning
# ============================================================================

# Create a copy for processing
df_processed = df.copy()

print("=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)

# 1. Handle missing values
print("\n1. Handling missing values...")
for col in df_processed.columns:
    if df_processed[col].isnull().sum() > 0:
        if df_processed[col].dtype in [np.float64, np.int64]:
            df_processed[col].fillna(df_processed[col].mean(), inplace=True)
            print(f"   - Filled {col} with mean value")
        else:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            print(f"   - Filled {col} with mode value")

# 2. Remove duplicates
initial_rows = len(df_processed)
df_processed.drop_duplicates(inplace=True)
print(f"\n2. Removed {initial_rows - len(df_processed)} duplicate rows")

# 3. Handle outliers using IQR method
print("\n3. Handling outliers...")
for col in numeric_cols:
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
    if len(outliers) > 0:
        print(f"   - Found {len(outliers)} outliers in {col}")

print(f"\n✓ Preprocessing complete. Final shape: {df_processed.shape}")

# ============================================================================
# 5. Feature Engineering
# ============================================================================

print("=" * 50)
print("FEATURE ENGINEERING")
print("=" * 50)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f"✓ Encoded {col}")

# Separate features and target
# IMPORTANT: Update this based on your target variable
target_col = numeric_cols[-1] if numeric_cols else None  # Assumes last numeric column is target

if target_col is None:
    print("\n⚠ Warning: No numeric target column found. Please specify target_col manually.")
else:
    print(f"\nTarget variable: {target_col}")
    
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("\n✓ Features scaled using StandardScaler")


# ============================================================================
# 6. Build Predictive Models
# ============================================================================

if target_col is None:
    print("Skipping model building - target variable not defined")
else:
    print("=" * 50)
    print("PREDICTIVE MODELING - REGRESSION")
    print("=" * 50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Dictionary to store models and metrics
    models = {}
    results = {}
    
    # 1. Linear Regression
    print("\n" + "-" * 50)
    print("1. Linear Regression")
    print("-" * 50)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    models['Linear Regression'] = lr_model
    results['Linear Regression'] = {
        'MAE': mean_absolute_error(y_test, y_pred_lr),
        'MSE': mean_squared_error(y_test, y_pred_lr),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'R2': r2_score(y_test, y_pred_lr)
    }
    
    print(f"MAE:  {results['Linear Regression']['MAE']:.4f}")
    print(f"RMSE: {results['Linear Regression']['RMSE']:.4f}")
    print(f"R²:   {results['Linear Regression']['R2']:.4f}")
    
    # 2. Random Forest Regressor
    print("\n" + "-" * 50)
    print("2. Random Forest Regressor")
    print("-" * 50)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'MSE': mean_squared_error(y_test, y_pred_rf),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'R2': r2_score(y_test, y_pred_rf)
    }
    
    print(f"MAE:  {results['Random Forest']['MAE']:.4f}")
    print(f"RMSE: {results['Random Forest']['RMSE']:.4f}")
    print(f"R²:   {results['Random Forest']['R2']:.4f}")
    
    # Summary of all models
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    # Select best model
    best_model_name = results_df['R2'].idxmax()
    print(f"\n✓ Best Model: {best_model_name}")
    
    best_model = models[best_model_name]
    best_predictions = models[best_model_name].predict(X_test) if best_model_name == 'Random Forest' else y_pred_lr


# ============================================================================
# 7. Model Evaluation and Validation
# ============================================================================

if target_col is None:
    print("Skipping evaluation - target variable not defined")
else:
    print("=" * 50)
    print("DETAILED MODEL EVALUATION")
    print("=" * 50)
    
    # Cross-validation scores
    print("\nCross-Validation Scores (5-fold):")
    for model_name, model in models.items():
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        print(f"\n{model_name}:")
        print(f"  CV R² Scores: {cv_scores}")
        print(f"  Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Feature importance for Random Forest
    if 'Random Forest' in models:
        print("\n" + "-" * 50)
        print("Random Forest - Feature Importance")
        print("-" * 50)
        feature_importance = pd.DataFrame({
            'Feature': X_scaled.columns,
            'Importance': models['Random Forest'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(feature_importance.head(10))
        
        # Plot top features
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importances (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{output_folder}/03_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Feature importance plot saved")


# ============================================================================
# 8. Generate Insights and Visualizations
# ============================================================================

if target_col is None:
    print("Skipping insights - target variable not defined")
else:
    print("=" * 50)
    print("KEY INSIGHTS & VISUALIZATIONS")
    print("=" * 50)
    
    # Get predictions from best model
    if best_model_name == 'Linear Regression':
        y_pred_best = lr_model.predict(X_test)
    else:
        y_pred_best = rf_model.predict(X_test)
    
    # 1. Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(y_test, y_pred_best, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'Actual vs Predicted ({best_model_name})')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_test - y_pred_best
    axes[1].scatter(y_pred_best, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/04_predictions_residuals.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Predictions vs Residuals plot saved")
    
    # 2. Error distribution
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.axvline(x=0, color='r', linestyle='--', lw=2)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/05_residual_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Error distribution plot saved")
    
    # 3. Model comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.loc[['Linear Regression', 'Random Forest'], 'R2'].plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral'])
    ax.set_title('Model Performance Comparison (R² Score)')
    ax.set_ylabel('R² Score')
    ax.set_xlabel('Model')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/06_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Model comparison plot saved")


# ============================================================================
# 8.1 Advanced Insights & Analysis
# ============================================================================

if target_col is None:
    print("Skipping advanced insights - target variable not defined")
else:
    print("=" * 50)
    print("ADVANCED INSIGHTS & ANALYSIS")
    print("=" * 50)
    
    # 1. Feature Correlation with Target
    print("\n" + "-" * 50)
    print("Top 15 Features Correlated with Target")
    print("-" * 50)
    correlations = X_scaled.copy()
    correlations['Target'] = y.values
    target_corr = correlations.corr()['Target'].drop('Target').abs().sort_values(ascending=False)
    print(target_corr.head(15))
    
    plt.figure(figsize=(10, 8))
    target_corr.head(15).plot(kind='barh', color='steelblue', edgecolor='black')
    plt.xlabel('Absolute Correlation with Target')
    plt.title('Top 15 Features Correlated with Target Variable')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_folder}/07_target_correlations.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Target correlation plot saved")
    
    # 2. Prediction Error Analysis
    print("\n" + "-" * 50)
    print("Prediction Error Analysis")
    print("-" * 50)
    errors = np.abs(y_test.values - y_pred_best)
    print(f"Mean Error: {errors.mean():.4f}")
    print(f"Median Error: {np.median(errors):.4f}")
    print(f"Std Dev Error: {errors.std():.4f}")
    print(f"Min Error: {errors.min():.4f}")
    print(f"Max Error: {errors.max():.4f}")
    print(f"95th Percentile Error: {np.percentile(errors, 95):.4f}")
    
    # Error distribution with statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Error histogram
    axes[0, 0].hist(errors, bins=40, edgecolor='black', color='coral')
    axes[0, 0].axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f}')
    axes[0, 0].axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}')
    axes[0, 0].set_xlabel('Absolute Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Prediction Errors')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q plot for residuals
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot of Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Prediction accuracy by range
    y_range_10 = pd.cut(y_test, bins=10)
    error_by_range = []
    range_labels = []
    for range_val in y_range_10.unique():
        mask = y_range_10 == range_val
        if mask.sum() > 0:
            error_by_range.append(errors[mask].mean())
            range_labels.append(f'{range_val.left:.0f}-{range_val.right:.0f}')
    
    axes[1, 0].plot(range_labels, error_by_range, marker='o', color='purple', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Target Value Range')
    axes[1, 0].set_ylabel('Mean Error')
    axes[1, 0].set_title('Error by Target Value Range')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative error distribution
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    axes[1, 1].plot(sorted_errors, cumulative, color='darkgreen', linewidth=2)
    axes[1, 1].axvline(np.percentile(errors, 95), color='red', linestyle='--', label='95th percentile')
    axes[1, 1].set_xlabel('Error Value')
    axes[1, 1].set_ylabel('Cumulative Percentage')
    axes[1, 1].set_title('Cumulative Distribution of Errors')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/08_error_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Detailed error analysis plot saved")
    
    # 3. Model Performance Metrics Summary
    print("\n" + "-" * 50)
    print("Detailed Model Performance Summary")
    print("-" * 50)
    
    metrics_summary = pd.DataFrame({
        'Linear Regression': {
            'MAE': mean_absolute_error(y_test, y_pred_lr),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            'R2': r2_score(y_test, y_pred_lr),
            'Explained Variance': explained_variance_score(y_test, y_pred_lr),
            'Median AE': median_absolute_error(y_test, y_pred_lr)
        },
        'Random Forest': {
            'MAE': mean_absolute_error(y_test, y_pred_rf),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'R2': r2_score(y_test, y_pred_rf),
            'Explained Variance': explained_variance_score(y_test, y_pred_rf),
            'Median AE': median_absolute_error(y_test, y_pred_rf)
        }
    }).T
    
    print(metrics_summary)
    
    # Visualization of metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # MAE comparison
    metrics_summary['MAE'].plot(kind='bar', ax=axes[0], color=['skyblue', 'lightcoral'], edgecolor='black')
    axes[0].set_title('Mean Absolute Error (MAE)')
    axes[0].set_ylabel('Error Value')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # RMSE comparison
    metrics_summary['RMSE'].plot(kind='bar', ax=axes[1], color=['skyblue', 'lightcoral'], edgecolor='black')
    axes[1].set_title('Root Mean Squared Error (RMSE)')
    axes[1].set_ylabel('Error Value')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # R2 comparison
    metrics_summary['R2'].plot(kind='bar', ax=axes[2], color=['skyblue', 'lightcoral'], edgecolor='black')
    axes[2].set_title('R2 Score (Higher is Better)')
    axes[2].set_ylabel('R2 Value')
    axes[2].set_ylim([0, 1.1])
    axes[2].tick_params(axis='x', rotation=0)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/09_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Detailed metrics comparison plot saved")


# ============================================================================
# 9. Save Results to Output Folder (Enhanced Summary Section)
# ============================================================================

print("=" * 50)
print("SAVING RESULTS")
print("=" * 50)

# Print Executive Summary before saving
if target_col is not None:
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY & KEY INSIGHTS")
    print("=" * 70)
    
    # Top correlated features
    top_features = target_corr.head(3)
    
    insights = f"""
    
    📊 DATASET OVERVIEW
    {'—' * 65}
    • Total Records: {len(df):,}
    • Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical
    • Total Features: {len(numeric_cols) + len(categorical_cols)}
    • Data Quality: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}% complete
    
    📈 TARGET VARIABLE: {target_col}
    {'—' * 65}
    • Mean: {y.mean():.4f}
    • Median: {y.median():.4f}
    • Std Dev: {y.std():.4f}
    • Range: [{y.min():.4f}, {y.max():.4f}]
    • Skewness: {y.skew():.4f}
    
    🎯 MODEL PERFORMANCE
    {'—' * 65}
    • Best Model: {best_model_name}
    • R2 Score: {r2_score(y_test, y_pred_best):.4f}
    • MAE: {mean_absolute_error(y_test, y_pred_best):.4f}
    • RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}
    • Mean Prediction Error: {errors.mean():.4f}
    • Median Prediction Error: {np.median(errors):.4f}
    
    🔍 KEY INSIGHTS
    {'—' * 65}
    Top Correlated Features:
    """
    
    for i, (feat, corr) in enumerate(top_features.items(), 1):
        insights += f"    {i}. {feat}: {corr:.4f}\n"
    
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    lr_r2 = r2_score(y_test, y_pred_lr)
    rf_r2 = r2_score(y_test, y_pred_rf)
    
    insights += f"""
    Data Quality:
    • Missing Data: {missing_pct:.2f}%
    • Duplicate Rows: {len(df) - len(df.drop_duplicates())}
    
    Model Comparison:
    • Linear Regression R2: {lr_r2:.4f}
    • Random Forest R2: {rf_r2:.4f}
    • Performance Gain: {(rf_r2 - lr_r2) / lr_r2 * 100 if lr_r2 != 0 else 0:.2f}%
    
    Error Distribution:
    • 50% of errors ≤ {np.percentile(errors, 50):.4f}
    • 90% of errors ≤ {np.percentile(errors, 90):.4f}
    • 95% of errors ≤ {np.percentile(errors, 95):.4f}
    
    {'=' * 65}
    """
    
    print(insights)
    
    # Create summary visualization
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    summary_text = f"ANALYSIS SUMMARY\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns  |  Target: {target_col}  |  Model: {best_model_name}"
    ax1.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7), family='monospace')
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    numeric_stats = f"NUMERIC FEATURES\n{'-'*25}\n" + "\n".join([
        f"{col[:15]:<15} {df[col].mean():>10.2f}" for col in numeric_cols[:5]
    ])
    ax2.text(0.1, 0.5, numeric_stats, ha='left', va='center', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    model_stats = f"MODEL METRICS\n{'-'*25}\nR2: {r2_score(y_test, y_pred_best):.4f}\nMAE: {mean_absolute_error(y_test, y_pred_best):.4f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}"
    ax3.text(0.1, 0.5, model_stats, ha='left', va='center', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    error_stats = f"ERROR ANALYSIS\n{'-'*25}\nMean: {errors.mean():.4f}\n95th %ile: {np.percentile(errors, 95):.4f}\nMax: {errors.max():.4f}"
    ax4.text(0.1, 0.5, error_stats, ha='left', va='center', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.hist(y_test, bins=30, alpha=0.6, label='Actual', edgecolor='black')
    ax5.hist(y_pred_best, bins=30, alpha=0.6, label='Predicted', edgecolor='black')
    ax5.set_xlabel('Value')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Target Distribution: Actual vs Predicted')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[2, 2])
    r2_score_val = r2_score(y_test, y_pred_best)
    colors = ['red' if r2_score_val < 0.3 else 'orange' if r2_score_val < 0.6 else 'yellow' if r2_score_val < 0.8 else 'green']
    ax6.barh([0], [r2_score_val], color=colors[0], edgecolor='black', height=0.5)
    ax6.set_xlim([0, 1])
    ax6.set_ylim([-0.5, 0.5])
    ax6.set_xlabel('R2 Score')
    ax6.set_title('Model Performance')
    ax6.set_yticks([])
    ax6.text(r2_score_val/2, 0, f'{r2_score_val:.3f}', ha='center', va='center', 
             fontsize=14, fontweight='bold', color='white')
    ax6.grid(True, alpha=0.3, axis='x')
    
    plt.savefig(f"{output_folder}/10_summary_insights.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Summary insights visualization saved")

print("\n" + "=" * 50)
print("SAVING RESULTS")
print("=" * 50)

# 1. Save processed data
processed_data_path = f"{output_folder}/processed_data.csv"
df_processed.to_csv(processed_data_path, index=False)
print(f"✓ Processed data saved to: {processed_data_path}")

# 2. Save predictions
if target_col is not None:
    predictions_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred_best,
        'Residual': y_test.values - y_pred_best
    })
    predictions_path = f"{output_folder}/predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predictions saved to: {predictions_path}")

# 3. Save model metrics report
metrics_report = {
    'Analysis_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'Dataset_Shape': df.shape,
    'Processed_Shape': df_processed.shape,
    'Target_Variable': str(target_col),
    'Best_Model': best_model_name,
    'Model_Metrics': results
}

report_path = f"{output_folder}/model_metrics_report.json"
with open(report_path, 'w') as f:
    json.dump(metrics_report, f, indent=4, default=str)
print(f"✓ Model metrics report saved to: {report_path}")

# 4. Save feature list
feature_list_path = f"{output_folder}/features.txt"
with open(feature_list_path, 'w') as f:
    f.write("FEATURE LIST\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total Features: {len(X_scaled.columns)}\n\n")
    f.write("Features:\n")
    for i, feature in enumerate(X_scaled.columns, 1):
        f.write(f"{i}. {feature}\n")
    if target_col:
        f.write(f"\n\nTarget Variable: {target_col}\n")
print(f"✓ Feature list saved to: {feature_list_path}")

# 5. Save insights summary
summary_path = f"{output_folder}/analysis_summary.txt"
with open(summary_path, 'w') as f:
    f.write("DATA ANALYSIS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Original Data Shape: {df.shape}\n")
    f.write(f"Processed Data Shape: {df_processed.shape}\n")
    f.write(f"Target Variable: {target_col}\n\n")
    
    if target_col is not None:
        f.write("\nMODEL PERFORMANCE\n")
        f.write("-" * 50 + "\n")
        for model_name, metrics in results.items():
            f.write(f"\n{model_name}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
        
        f.write(f"\n\nBest Model: {best_model_name}\n")

print(f"✓ Analysis summary saved to: {summary_path}")

# 6. List all saved files
print("\n" + "=" * 50)
print("OUTPUT FILES GENERATED")
print("=" * 50)
saved_files = os.listdir(output_folder)
for i, file in enumerate(saved_files, 1):
    file_path = os.path.join(output_folder, file)
    file_size = os.path.getsize(file_path) / 1024  # Size in KB
    print(f"{i}. {file} ({file_size:.2f} KB)")

print(f"\n✓ All results saved to folder: {output_folder}/")
print("✓ Analysis complete!")

