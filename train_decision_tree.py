import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz
import os
import joblib # For saving the model

# Define paths
cleaned_data_path = '/home/ubuntu/heart_cleaned.csv'

# Create directory for model outputs if it doesn't exist
output_dir = '/home/ubuntu/model_outputs'
os.makedirs(output_dir, exist_ok=True)

model_save_path = os.path.join(output_dir, 'decision_tree_model.joblib')
tree_vis_dot_path = os.path.join(output_dir, 'decision_tree.dot')
tree_vis_png_path = os.path.join(output_dir, 'decision_tree.png')

# Load the cleaned data
try:
    df = pd.read_csv(cleaned_data_path)
    print(f"Loaded cleaned data from {cleaned_data_path}")

    # Separate features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = list(X.columns)
    class_names = [str(c) for c in sorted(y.unique())] # ['0', '1']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets.")

    # 1. Train a Decision Tree Classifier (unconstrained for now)
    print("\nTraining Decision Tree Classifier (unconstrained)...")
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    print("Decision Tree training complete.")

    # Evaluate on the test set
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set: {accuracy:.4f}")

    # Save the trained model
    joblib.dump(dt_classifier, model_save_path)
    print(f"Trained Decision Tree model saved to {model_save_path}")

    # 2. Visualize the tree
    print("\nVisualizing the Decision Tree...")
    # Export the tree to a DOT file
    # Limit depth for initial visualization clarity
    export_graphviz(dt_classifier,
                    out_file=tree_vis_dot_path,
                    feature_names=feature_names,
                    class_names=class_names,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    max_depth=3)
    print(f"Decision Tree DOT data saved to {tree_vis_dot_path}")

    # Convert DOT file to PNG using graphviz library
    try:
        # Create graph from dot file
        graph = graphviz.Source.from_file(tree_vis_dot_path)
        # Render to PNG - ensure filename doesn't contain .png before render adds it
        png_base_name = os.path.splitext(tree_vis_png_path)[0]
        graph.render(filename=png_base_name, format='png', cleanup=True)
        # Check if the file exists after rendering
        if os.path.exists(tree_vis_png_path):
             print(f"Decision Tree visualization saved to {tree_vis_png_path}")
        else:
             # Sometimes render adds .png automatically, check that too
             if os.path.exists(png_base_name + '.png'):
                  print(f"Decision Tree visualization saved to {png_base_name + '.png'}")
             else:
                  print(f"Error: PNG file not found after rendering at expected path: {tree_vis_png_path} or {png_base_name + '.png'}")

    except Exception as e:
        print(f"Error generating PNG from DOT file: {e}")
        print("DOT file saved, but PNG conversion failed. Ensure Graphviz is installed and in PATH.")

except FileNotFoundError:
    print(f"Error: Cleaned data file not found at {cleaned_data_path}")
except Exception as e:
    print(f"An error occurred during Decision Tree training/visualization: {e}")

