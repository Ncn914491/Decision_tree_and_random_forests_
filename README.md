# AI & ML Internship Task: Decision Trees and Random Forests

## Objective

The primary goal of this task is to understand and implement tree-based models (Decision Trees and Random Forests) for classification using the Heart Disease dataset. This involves training models, visualizing decision trees, analyzing overfitting, comparing model performance, interpreting feature importance, and evaluating models using cross-validation.

## Tools Used

*   **Python 3.11**
*   **Libraries:**
    *   `pandas`: For data manipulation and loading.
    *   `scikit-learn`: For machine learning tasks (train/test split, Decision Tree, Random Forest, accuracy metrics, cross-validation).
    *   `matplotlib` & `seaborn`: For data visualization (EDA plots, overfitting analysis, feature importance).
    *   `graphviz`: For visualizing the Decision Tree structure.
    *   `joblib`: For saving and loading trained models.
    *   `numpy`: For numerical operations (mean, std dev).
    *   `kagglehub`: For downloading the dataset.
*   **Graphviz (System Package):** Required by the `graphviz` Python library to render the decision tree visualization.

## Project Structure

```
.
├── README.md                 # This file
├── heart_cleaned.csv         # Cleaned dataset used for modeling
├── preprocess_data.py        # Script for initial data loading and cleaning
├── eda.py                    # Script for Exploratory Data Analysis
├── eda_plots/                # Directory containing EDA plots
│   ├── boxplots/             # Boxplots for numerical features
│   │   └── ...
│   └── histograms/           # Histograms for numerical features
│       └── ...
├── train_decision_tree.py    # Script to train and visualize the initial Decision Tree
├── analyze_dt_overfitting_fixed.py # Script to analyze DT overfitting and find best depth
├── train_random_forest_fixed.py # Script to train Random Forest and compare with DT
├── interpret_feature_importance.py # Script to calculate and plot RF feature importance
├── evaluate_cross_validation.py # Script to perform cross-validation on DT and RF
└── model_outputs/            # Directory containing model outputs
    ├── decision_tree.dot                 # Graphviz dot file for DT
    ├── decision_tree.png                 # Visualized Decision Tree
    ├── decision_tree_model.joblib        # Saved initial Decision Tree model
    ├── dt_overfitting_analysis.png       # Plot showing DT accuracy vs. depth
    ├── dt_overfitting_results.csv        # CSV with DT accuracy for different depths
    ├── decision_tree_best_depth_3.joblib # Saved DT model with best depth (depth=3)
    ├── decision_tree_depth_3_model.joblib # Copy of the best DT model for CV script
    ├── random_forest_model.joblib        # Saved Random Forest model
    ├── accuracy_comparison.txt           # Text file comparing DT and RF test accuracy
    ├── rf_feature_importance.png         # Plot showing RF feature importances
    ├── rf_feature_importance.txt         # Text file with RF feature importances
    └── cross_validation_results.txt      # Text file with cross-validation results
```

## How to Run

1.  **Prerequisites:** Ensure Python 3.11+, pip, and the Graphviz system package (`sudo apt-get install graphviz` on Debian/Ubuntu) are installed.
2.  **Install Libraries:** `pip install pandas scikit-learn matplotlib seaborn graphviz joblib numpy kagglehub`
3.  **Download Data:** The `preprocess_data.py` script uses `kagglehub` to download the dataset initially. If this fails (e.g., due to authentication), manually download `heart-disease-dataset.zip` from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset), extract `heart.csv`, and place it in the root directory.
4.  **Run Scripts:** Execute the Python scripts in the following order:
    *   `python3.11 preprocess_data.py`
    *   `python3.11 eda.py`
    *   `python3.11 train_decision_tree.py`
    *   `python3.11 analyze_dt_overfitting_fixed.py`
    *   `python3.11 train_random_forest_fixed.py`
    *   `python3.11 interpret_feature_importance.py`
    *   `python3.11 evaluate_cross_validation.py`

## Summary of Findings

1.  **Data Preprocessing:** The dataset was loaded, basic cleaning was performed (no significant missing values found in this dataset), and features were prepared for modeling.
2.  **EDA:** Exploratory Data Analysis revealed distributions and potential relationships between features and the target variable (presence of heart disease). Plots are available in the `eda_plots/` directory.
3.  **Decision Tree:** An initial Decision Tree was trained and visualized (`model_outputs/decision_tree.png`).
4.  **Overfitting Analysis:** By varying the `max_depth` parameter, we observed how Decision Tree performance changes on training and test sets. The analysis (`model_outputs/dt_overfitting_analysis.png`, `model_outputs/dt_overfitting_results.csv`) showed that a depth of 3 provided the best generalization on the test set (Accuracy: ~0.8033) while avoiding the overfitting seen at greater depths.
5.  **Random Forest:** A Random Forest model (100 trees) was trained. It achieved a test set accuracy of ~0.7541 (`model_outputs/accuracy_comparison.txt`). In this specific run, the pruned Decision Tree (depth=3) slightly outperformed the default Random Forest on the initial test split.
6.  **Feature Importance:** The Random Forest model identified `cp` (chest pain type), `thalach` (max heart rate achieved), `ca` (number of major vessels colored by fluoroscopy), `oldpeak` (ST depression induced by exercise), and `thal` (thalassemia type) as the most important features (`model_outputs/rf_feature_importance.png`, `model_outputs/rf_feature_importance.txt`).
7.  **Cross-Validation:** 10-fold stratified cross-validation provided a more robust evaluation (`model_outputs/cross_validation_results.txt`):
    *   Decision Tree (Depth=3): Mean Accuracy ~0.7884 (Std Dev ~0.0766)
    *   Random Forest: Mean Accuracy ~0.8246 (Std Dev ~0.0647)
    Cross-validation indicated that the Random Forest model generalizes better on average across different data subsets compared to the single pruned Decision Tree.

## Interview Questions & Answers

**1. How does a decision tree work?**

A decision tree is a supervised learning algorithm used for both classification and regression. It works by recursively partitioning the dataset into smaller subsets based on the values of input features. It starts with a root node representing the entire dataset. At each node, it selects the feature and a split point (or condition) that best separates the data according to a specific criterion (like Gini impurity or information gain for classification). This process creates branches leading to child nodes, each representing a subset of the data that satisfies the condition. The splitting continues until a stopping criterion is met, such as reaching a maximum depth, having a minimum number of samples in a node, or achieving pure nodes (nodes containing samples of only one class). The terminal nodes, called leaf nodes, contain the final prediction (the majority class for classification or the average value for regression) for samples reaching that leaf.

**2. What is entropy and information gain?**

*   **Entropy:** In the context of decision trees, entropy measures the impurity or randomness of a set of samples. It quantifies the uncertainty associated with the class labels in a subset of data. Entropy is 0 if all samples in a node belong to the same class (pure node) and reaches its maximum value when the classes are equally distributed. The formula for entropy for a set S with C classes is: `Entropy(S) = - Σ (p_i * log2(p_i))` where `p_i` is the proportion of samples belonging to class `i`.
*   **Information Gain:** Information Gain is the metric used by decision tree algorithms (like ID3) to select the best feature to split on at each node. It measures the reduction in entropy achieved by partitioning the data based on a particular feature. It is calculated as the difference between the entropy of the parent node and the weighted average entropy of the child nodes resulting from the split. `InformationGain(S, A) = Entropy(S) - Σ (|S_v| / |S|) * Entropy(S_v)`, where S is the set of samples, A is the feature to split on, v represents the possible values of feature A, S_v is the subset of S for which feature A has value v, and |S| is the number of samples in S. The feature that provides the highest information gain is chosen for the split, as it leads to the most significant reduction in impurity.

**3. How is random forest better than a single tree?**

Random Forest is an ensemble learning method that builds multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. It improves upon a single decision tree in several ways:

*   **Reduces Overfitting:** Single decision trees are prone to overfitting, meaning they learn the training data too well, including noise, and perform poorly on unseen data. Random Forests reduce overfitting by averaging the predictions of many trees, each trained on a different random subset of the data (bagging) and considering only a random subset of features at each split. This randomness helps to decorrelate the trees and makes the overall model more robust.
*   **Improves Accuracy:** By combining the predictions of multiple diverse trees, Random Forests often achieve higher accuracy and better generalization performance than a single optimized tree.
*   **Handles High Dimensionality:** They can handle datasets with a large number of features effectively due to the random feature subset selection at each split.
*   **Robust to Noise:** The averaging process makes the model less sensitive to noise in the data compared to a single tree.

**4. What is overfitting and how do you prevent it?**

*   **Overfitting:** Overfitting occurs when a machine learning model learns the training data too well, capturing not only the underlying patterns but also the noise and random fluctuations specific to that training set. As a result, the model performs exceptionally well on the training data but poorly on new, unseen data (like the test set or real-world data) because it fails to generalize.
*   **Prevention Techniques:** Several techniques can be used to prevent overfitting, especially in decision trees and random forests:
    *   **Pruning (for Decision Trees):** Limiting the growth of the tree by setting constraints like `max_depth` (maximum depth of the tree), `min_samples_split` (minimum samples required to split a node), `min_samples_leaf` (minimum samples required in a leaf node), or `max_leaf_nodes`. Post-pruning involves growing the full tree and then removing branches that provide little predictive power on a validation set.
    *   **Using Ensemble Methods (like Random Forest):** As discussed, Random Forests inherently reduce overfitting through bagging and random feature selection.
    *   **Cross-Validation:** Using techniques like k-fold cross-validation helps to get a more reliable estimate of the model's performance on unseen data and can be used to tune hyperparameters (like `max_depth`) effectively.
    *   **Regularization (Less common for trees, but applicable):** Adding penalties for model complexity.
    *   **Getting More Data:** A larger, representative training dataset can help the model learn more general patterns.

**5. What is bagging?**

Bagging, short for Bootstrap Aggregating, is an ensemble machine learning technique designed to improve the stability and accuracy of models, reduce variance, and help avoid overfitting. The process involves:

1.  **Bootstrap Sampling:** Creating multiple random subsets of the original training dataset by sampling *with replacement*. This means each subset (bootstrap sample) has the same size as the original dataset, but some data points may appear multiple times while others may be omitted.
2.  **Model Training:** Training a separate base model (e.g., a decision tree) independently on each bootstrap sample.
3.  **Aggregation:** Combining the predictions of all the individual models. For classification, this is typically done by majority voting (taking the most frequent prediction). For regression, it's usually done by averaging the predictions.

Random Forest is a specific implementation of bagging that uses decision trees as base learners and adds an extra layer of randomness by selecting a random subset of features at each split point in the trees.

**6. How do you visualize a decision tree?**

Decision trees can be visualized graphically, which helps in understanding how the model makes predictions. Common methods include:

*   **Using Graphviz:** Libraries like `scikit-learn` provide functions (`sklearn.tree.export_graphviz`) to export the trained tree structure into a `.dot` file format. This `.dot` file can then be processed by the Graphviz software (or the `graphviz` Python library) to generate visual representations (e.g., PNG, PDF, SVG images) of the tree. The visualization typically shows the nodes, the splitting conditions, the impurity measure (e.g., Gini), the number of samples at each node, and the class distribution or predicted value.
*   **Using `sklearn.tree.plot_tree`:** Scikit-learn also offers a built-in function (`plot_tree`) that uses `matplotlib` to directly plot the decision tree without needing the external Graphviz software installation. It provides similar information within the plot nodes.

**7. How do you interpret feature importance?**

Feature importance scores indicate the relative contribution of each input feature in making predictions by the model (particularly tree-based models like Random Forest). Interpretation involves:

*   **Calculation:** In tree-based models, importance is often calculated based on how much each feature contributes to reducing impurity (like Gini impurity or entropy) across all the splits in all the trees (for Random Forest) or in a single tree. Features used higher up in the trees or in more splits generally have higher importance scores.
*   **Ranking:** Scores are typically normalized (summing to 1 or scaled relative to the most important feature). Features are then ranked from most to least important.
*   **Visualization:** Bar plots are commonly used to visualize feature importances, making it easy to compare the relative contributions of different features.
*   **Meaning:** A higher score means the feature was more influential in the model's decision-making process. This helps understand which factors the model considers most critical for prediction, aids in feature selection (potentially removing low-importance features), and provides insights into the underlying data patterns.

**8. What are the pros/cons of random forests?**

**Pros:**

*   **High Accuracy:** Generally provide high predictive accuracy and perform well on a wide range of classification and regression tasks.
*   **Robust to Overfitting:** Significantly less prone to overfitting compared to individual decision trees due to bagging and feature randomness.
*   **Handles Non-linearities:** Can capture complex non-linear relationships between features and the target variable.
*   **Handles Missing Values:** Can handle missing data to some extent (though imputation is often still recommended).
*   **Feature Importance:** Provide useful estimates of feature importance.
*   **No Need for Feature Scaling:** Do not require feature scaling or normalization like distance-based algorithms (e.g., SVM, k-NN).
*   **Parallelizable:** Training individual trees can be parallelized, speeding up the training process on multi-core systems.

**Cons:**

*   **Less Interpretable:** Random Forests are often considered 
