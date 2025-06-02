import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Define paths (single-line assignments)
cleaned_data_path = '/home/ubuntu/heart_cleaned.csv'
rf_model_path = '/home/ubuntu/model_outputs/random_forest_model.joblib'
output_dir = '/home/ubuntu/model_outputs'
feature_importance_plot_path = os.path.join(output_dir, 'rf_feature_importance.png')
feature_importance_text_path = os.path.join(output_dir, 'rf_feature_importance.txt')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the trained Random Forest model and cleaned data
try:
    rf_classifier = joblib.load(rf_model_path)
    print(f"Loaded Random Forest model from {rf_model_path}")
    df = pd.read_csv(cleaned_data_path)
    print(f"Loaded cleaned data from {cleaned_data_path}")

    # Get feature names (excluding the target variable)
    feature_names = df.drop('target', axis=1).columns

    # Get feature importances
    importances = rf_classifier.feature_importances_

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("\n--- Feature Importances ---")
    print(feature_importance_df)

    # Save feature importances to text file
    feature_importance_df.to_csv(feature_importance_text_path, index=False, sep='\t')
    print(f"Feature importances saved to {feature_importance_text_path}")

    # Visualize feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(feature_importance_plot_path)
    print(f"Feature importance plot saved to {feature_importance_plot_path}")
    plt.close() # Close the plot to free memory

except FileNotFoundError as e:
    print(f"Error: Required file not found. {e}")
except Exception as e:
    print(f"An error occurred during feature importance interpretation: {e}")

