import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np

# Define paths (single-line assignments)
cleaned_data_path = '/home/ubuntu/heart_cleaned.csv'
dt_model_path = '/home/ubuntu/model_outputs/decision_tree_depth_3_model.joblib'
rf_model_path = '/home/ubuntu/model_outputs/random_forest_model.joblib'
output_dir = '/home/ubuntu/model_outputs'
cv_results_path = os.path.join(output_dir, 'cross_validation_results.txt')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the cleaned data
try:
    df = pd.read_csv(cleaned_data_path)
    print(f"Loaded cleaned data from {cleaned_data_path}")

    # Separate features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Load the trained models
    dt_classifier = joblib.load(dt_model_path)
    print(f"Loaded Decision Tree model (depth=3) from {dt_model_path}")
    rf_classifier = joblib.load(rf_model_path)
    print(f"Loaded Random Forest model from {rf_model_path}")

    # Define cross-validation strategy (e.g., 10-fold stratified)
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    print(f"Using {cv_strategy.get_n_splits()}-fold stratified cross-validation.")

    # Perform cross-validation for Decision Tree
    print("\nPerforming cross-validation for Decision Tree (depth=3)...")
    dt_cv_scores = cross_val_score(dt_classifier, X, y, cv=cv_strategy, scoring='accuracy')
    dt_cv_mean = np.mean(dt_cv_scores)
    dt_cv_std = np.std(dt_cv_scores)
    print(f"Decision Tree CV Accuracy Scores: {dt_cv_scores}")
    print(f"Decision Tree CV Mean Accuracy: {dt_cv_mean:.4f}")
    print(f"Decision Tree CV Std Dev:       {dt_cv_std:.4f}")

    # Perform cross-validation for Random Forest
    print("\nPerforming cross-validation for Random Forest...")
    rf_cv_scores = cross_val_score(rf_classifier, X, y, cv=cv_strategy, scoring='accuracy')
    rf_cv_mean = np.mean(rf_cv_scores)
    rf_cv_std = np.std(rf_cv_scores)
    print(f"Random Forest CV Accuracy Scores: {rf_cv_scores}")
    print(f"Random Forest CV Mean Accuracy: {rf_cv_mean:.4f}")
    print(f"Random Forest CV Std Dev:       {rf_cv_std:.4f}")

    # Save results to file
    cv_results_text = f"Cross-Validation Results (10-fold Stratified):\n\n"
    cv_results_text += f"Decision Tree (Depth=3):\n"
    cv_results_text += f"  Scores: {dt_cv_scores}\n"
    cv_results_text += f"  Mean Accuracy: {dt_cv_mean:.4f}\n"
    cv_results_text += f"  Standard Deviation: {dt_cv_std:.4f}\n\n"
    cv_results_text += f"Random Forest:\n"
    cv_results_text += f"  Scores: {rf_cv_scores}\n"
    cv_results_text += f"  Mean Accuracy: {rf_cv_mean:.4f}\n"
    cv_results_text += f"  Standard Deviation: {rf_cv_std:.4f}\n"

    with open(cv_results_path, 'w') as f:
        f.write(cv_results_text)
    print(f"\nCross-validation results saved to {cv_results_path}")

except FileNotFoundError as e:
    print(f"Error: Required file not found. {e}")
except Exception as e:
    print(f"An error occurred during cross-validation: {e}")

