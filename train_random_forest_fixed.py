import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import joblib

# Define paths
cleaned_data_path = '/home/ubuntu/heart_cleaned.csv'
output_dir = '/home/ubuntu/model_outputs'
rf_model_save_path = os.path.join(output_dir, 'random_forest_model.joblib')
accuracy_comparison_path = os.path.join(output_dir, 'accuracy_comparison.txt')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the cleaned data
try:
    df = pd.read_csv(cleaned_data_path)
    print(f"Loaded cleaned data from {cleaned_data_path}")

    # Separate features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data into training and testing sets (same split as before)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Using training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets.")

    # 3. Train a Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    # Using common default parameters
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    print("Random Forest training complete.")

    # Evaluate on the test set
    y_pred_rf = rf_classifier.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy on the test set: {rf_accuracy:.4f}")

    # Save the trained Random Forest model
    joblib.dump(rf_classifier, rf_model_save_path)
    print(f"Trained Random Forest model saved to {rf_model_save_path}")

    # 4. Compare accuracy with Decision Tree
    # Load previous DT results if needed, or just use the known values
    # From previous step: Best DT (depth=3) Test Accuracy = 0.8033
    dt_best_accuracy = 0.8033 # From analyze_dt_overfitting_fixed.py output

    print("\n--- Accuracy Comparison ---")
    print(f"Decision Tree (Best Depth=3) Test Accuracy: {dt_best_accuracy:.4f}")
    print(f"Random Forest Test Accuracy:                 {rf_accuracy:.4f}")

    comparison_text = f"Accuracy Comparison:\n"
    comparison_text += f"- Decision Tree (Best Depth=3) Test Accuracy: {dt_best_accuracy:.4f}\n"
    comparison_text += f"- Random Forest Test Accuracy:                 {rf_accuracy:.4f}\n"

    # Save comparison to file
    with open(accuracy_comparison_path, 'w') as f:
        f.write(comparison_text)
    print(f"Accuracy comparison saved to {accuracy_comparison_path}")

except FileNotFoundError:
    print(f"Error: Cleaned data file not found at {cleaned_data_path}")
except Exception as e:
    print(f"An error occurred during Random Forest training/comparison: {e}")

