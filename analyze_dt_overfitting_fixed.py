import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os
import joblib

# Define paths
cleaned_data_path = '/home/ubuntu/heart_cleaned.csv'
output_dir = '/home/ubuntu/model_outputs'
overfitting_plot_path = os.path.join(output_dir, 'dt_overfitting_analysis.png')
overfitting_results_path = os.path.join(output_dir, 'dt_overfitting_results.csv')

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

    # Analyze overfitting by varying max_depth
    max_depths = range(1, 16) # Test depths from 1 to 15
    train_accuracies = []
    test_accuracies = []

    print("\nAnalyzing overfitting by varying max_depth...")
    for depth in max_depths:
        # Train Decision Tree with specific max_depth
        dt_depth = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt_depth.fit(X_train, y_train)

        # Evaluate on training set
        y_train_pred = dt_depth.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_acc)

        # Evaluate on test set
        y_test_pred = dt_depth.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_accuracies.append(test_acc)

        print(f"Max Depth: {depth}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save results to CSV
    results_df = pd.DataFrame({
        'Max Depth': max_depths,
        'Train Accuracy': train_accuracies,
        'Test Accuracy': test_accuracies
    })
    results_df.to_csv(overfitting_results_path, index=False)
    print(f"\nOverfitting analysis results saved to {overfitting_results_path}")

    # Plot the results
    print("\nPlotting training vs. test accuracy...")
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 7))
    plt.plot(max_depths, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(max_depths, test_accuracies, marker='s', label='Test Accuracy')
    plt.xlabel('Max Depth of Decision Tree', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Decision Tree Accuracy vs. Max Depth (Overfitting Analysis)', fontsize=14)
    plt.xticks(max_depths)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(overfitting_plot_path)
    plt.close()
    print(f"Overfitting analysis plot saved to {overfitting_plot_path}")

    # Identify best depth based on test accuracy
    best_depth_index = np.argmax(test_accuracies)
    best_depth = max_depths[best_depth_index]
    best_test_accuracy = test_accuracies[best_depth_index]
    corresponding_train_accuracy = train_accuracies[best_depth_index]
    print(f"\nBest performance on test set found at max_depth={best_depth}")
    print(f"  - Test Accuracy: {best_test_accuracy:.4f}")
    print(f"  - Training Accuracy: {corresponding_train_accuracy:.4f}")

    # Train and save the 'best' model based on this analysis
    print(f"\nTraining final Decision Tree with max_depth={best_depth}...")
    best_dt_classifier = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    best_dt_classifier.fit(X_train, y_train)
    # Corrected f-string for filename - ensure it's on one line
    best_model_save_path = os.path.join(output_dir, f'decision_tree_best_depth_{best_depth}.joblib')
    joblib.dump(best_dt_classifier, best_model_save_path)
    print(f"Best Decision Tree model (depth {best_depth}) saved to {best_model_save_path}")

except FileNotFoundError:
    print(f"Error: Cleaned data file not found at {cleaned_data_path}")
except Exception as e:
    print(f"An error occurred during overfitting analysis: {e}")

