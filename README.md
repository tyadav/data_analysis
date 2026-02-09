# Data Analysis & Predictive Modeling Pipeline

A comprehensive Python script for end-to-end data analysis, exploratory data analysis (EDA), and predictive modeling with performance evaluation and visualization.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setup & Activation](#setup--activation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Output Files](#output-files)
- [Script Workflow](#script-workflow)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Features

### Data Analysis & Exploration
- **Exploratory Data Analysis (EDA)**: Statistical summaries, distribution analysis, and missing value detection
- **Correlation Analysis**: Heatmap visualization of feature correlations
- **Distribution Plots**: Histograms and boxplots for numeric features
- **Categorical Analysis**: Value counts, bar charts, and pie charts for categorical data

### Data Preprocessing & Cleaning
- Automatic missing value imputation (mean for numeric, mode for categorical)
- Duplicate row removal
- Outlier detection using IQR (Interquartile Range) method
- Data validation and quality assessment

### Feature Engineering
- Categorical variable encoding (Label Encoding)
- Feature scaling using StandardScaler
- Automatic feature-target separation

### Predictive Modeling
- **Linear Regression**: Simple baseline model
- **Random Forest Regressor**: Ensemble-based prediction model
- Cross-validation scoring (5-fold)
- Comprehensive model comparison and evaluation

### Performance Evaluation
- Multiple evaluation metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
  - Explained Variance Score
  - Median Absolute Error
- Residual analysis and error distribution visualization
- Feature importance analysis (Random Forest)
- Prediction accuracy by value range

### Visualization & Reporting
- 10+ visualization outputs:
  - Correlation matrix heatmap
  - Distribution plots and boxplots
  - Feature importance charts
  - Actual vs Predicted scatter plots
  - Residual plots
  - Error distribution curves
  - Q-Q plots for normality testing
  - Summary dashboards
- JSON report generation with detailed metrics
- CSV exports of processed data and predictions

## Requirements

- Python 3.7 or higher
- Virtual environment (recommended)

### Python Dependencies
See `requirements.txt` for all required packages:
- `pandas>=1.3.0` - Data manipulation and analysis
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.4.0` - Plotting and visualization
- `seaborn>=0.11.0` - Statistical data visualization
- `scikit-learn>=0.24.0` - Machine learning library
- `scipy>=1.7.0` - Scientific computing

## Installation

### 1. Clone or Download the Project
```bash
# Navigate to your project directory
cd C:\Tej\Python26
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
& .\.venv\Scripts\Activate.ps1

# Or for Command Prompt (Command Prompt - not PowerShell)
.venv\Scripts\activate.bat

# For Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

## Setup & Activation

### Windows PowerShell (Recommended)
```powershell
# Navigate to project directory
cd C:\Tej\Python26

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# If you get an execution policy error, run PowerShell as Administrator and execute:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Windows Command Prompt
```bash
cd C:\Tej\Python26
.venv\Scripts\activate.bat
```

### Linux/macOS
```bash
cd ~/Python26
source .venv/bin/activate
```

## Usage

### Running the Script

```bash
# Make sure virtual environment is activated first
python dataAnalysis.py
```

### Expected Output
- Console output with analysis progress and metrics
- 10+ visualization images saved to `analysis_output/` folder
- CSV files with processed data and predictions
- JSON report with detailed metrics
- Text file with analysis summary

## Project Structure

```
Python26/
├── dataAnalysis.py                 # Main analysis script
├── data_analysis_pipeline.ipynb    # Jupyter notebook version
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── docs/
│   └── instructions.md             # Setup instructions
└── analysis_output/                # Output folder (auto-created)
    ├── processed_data.csv
    ├── predictions.csv
    ├── model_metrics_report.json
    ├── analysis_summary.txt
    ├── features.txt
    ├── 01_correlation_matrix.png
    ├── 02_distribution_*.png
    ├── 03_*.png
    ├── 04_predictions_residuals.png
    ├── 05_residual_distribution.png
    ├── 06_model_comparison.png
    ├── 07_target_correlations.png
    ├── 08_error_analysis.png
    ├── 09_metrics_comparison.png
    └── 10_summary_insights.png
```

## Output Files

### Data Files
- **processed_data.csv**: Cleaned and preprocessed dataset
- **predictions.csv**: Predictions with actual values and residuals
- **features.txt**: List of all features used in the model

### Reports
- **model_metrics_report.json**: Detailed metrics for all models
- **analysis_summary.txt**: Human-readable analysis summary

### Visualizations
1. **01_correlation_matrix.png** - Feature correlation heatmap
2. **02_distribution_*.png** - Distribution and boxplots for numeric features
3. **03_categorical_*.png** - Distribution of categorical features
4. **03_feature_importance.png** - Top 10 important features (Random Forest)
5. **03_target_distribution.png** - Target variable distribution
6. **04_predictions_residuals.png** - Actual vs Predicted and residual plots
7. **05_residual_distribution.png** - Error distribution histogram
8. **06_model_comparison.png** - R² comparison between models
9. **07_target_correlations.png** - Top features correlated with target
10. **08_error_analysis.png** - Detailed error analysis (4-panel visualization)
11. **09_metrics_comparison.png** - MAE, RMSE, R² comparison
12. **10_summary_insights.png** - Executive summary dashboard

## Script Workflow

The script follows this structured workflow:

1. **Library Imports** - Load all required packages
2. **Data Loading** - Read CSV file from specified path
3. **EDA** - Exploratory analysis with visualizations
4. **Data Preprocessing** - Clean, handle missing values, remove duplicates, detect outliers
5. **Feature Engineering** - Encode categorical variables, scale features
6. **Model Building** - Train Linear Regression and Random Forest models
7. **Model Evaluation** - Calculate metrics and cross-validation scores
8. **Analysis & Insights** - Generate visualizations and error analysis
9. **Results Export** - Save all outputs to analysis_output folder

## Configuration

### Important: Update Data File Path

Before running the script, modify the data file path in `dataAnalysis.py`:

```python
# Line 40 - Change this to your data file path
data_file_path = r"C:/Users/TejYadav/OneDrive - kyndryl/Data/Sales by Store.csv"

# Change to your path:
data_file_path = r"YOUR_PATH_TO_DATA_FILE.csv"
```

### Output Folder
The script automatically creates an `analysis_output` folder in the current directory. All results are saved here.

## Troubleshooting

### Virtual Environment Issues

**Problem**: Command not found: `python` or `pip`
```powershell
# Solution: Use full path or activate virtual environment
& .\.venv\Scripts\python.exe --version
& .\.venv\Scripts\pip.exe list
```

**Problem**: ExecutionPolicy error in PowerShell
```powershell
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Package Installation Issues

**Problem**: `pip install` fails
```bash
# Upgrade pip, setuptools, and wheel
python -m pip install --upgrade pip setuptools wheel

# Then install requirements
pip install -r requirements.txt
```

**Problem**: Missing module error when running script
```bash
# Verify packages are installed
pip list

# Reinstall specific package
pip install --force-reinstall pandas numpy matplotlib seaborn scikit-learn scipy
```

### Data File Issues

**Problem**: `FileNotFoundError: File not found`
- Check the `data_file_path` in the script
- Ensure the file path uses forward slashes `/` or raw strings `r"path"`
- Verify the file exists and you have read permissions

**Problem**: Encoding errors when reading CSV
```python
# Add encoding parameter to the read_csv call (line ~45):
df = pd.read_csv(data_file_path, encoding='utf-8')
# or
df = pd.read_csv(data_file_path, encoding='latin-1')
```

### Memory Issues with Large Datasets

**Problem**: Script runs out of memory
```python
# Read data in chunks:
chunksize = 10000
df = pd.concat(pd.read_csv(data_file_path, chunksize=chunksize))
```

## Model Selection

The script automatically selects the best performing model based on R² score:

- **Linear Regression**: Good for baseline and interpretability
- **Random Forest**: Better for complex, non-linear relationships

Both models use 80-20 train-test split with a random seed of 42 for reproducibility.

## Performance Metrics Explained

- **MAE** (Mean Absolute Error): Average magnitude of prediction errors (lower is better)
- **RMSE** (Root Mean Squared Error): Standard deviation of prediction errors (lower is better)
- **R² Score**: Proportion of variance explained by the model (higher is better, max 1.0)
- **Explained Variance Score**: Similar to R², shows how well the model captures variance
- **Median AE**: Median of absolute errors, robust to outliers

## Next Steps

1. Modify the data file path to point to your dataset
2. Run the script and review the outputs in `analysis_output/`
3. Adjust model parameters in the script if needed
4. Export results for further analysis or reporting

---

**Created**: February 2026
**Version**: 1.0
**Python Version**: 3.7+
