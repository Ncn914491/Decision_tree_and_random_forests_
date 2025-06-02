import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
cleaned_data_path = '/home/ubuntu/heart_cleaned.csv'
eda_plots_dir = '/home/ubuntu/eda_plots'
hist_dir = os.path.join(eda_plots_dir, 'histograms')
box_dir = os.path.join(eda_plots_dir, 'boxplots')
summary_stats_path = '/home/ubuntu/summary_statistics.txt'

# Load the cleaned data
try:
    df = pd.read_csv(cleaned_data_path)
    print(f"Loaded cleaned data from {cleaned_data_path}")

    # 1. Generate summary statistics
    print("\n--- Summary Statistics ---")
    summary_stats = df.describe()
    print(summary_stats)
    # Save summary statistics to a file
    with open(summary_stats_path, 'w') as f:
        f.write("Summary Statistics:\n")
        f.write(summary_stats.to_string())
    print(f"Summary statistics saved to {summary_stats_path}")

    # Identify numerical columns (excluding binary/categorical encoded as int for specific plots)
    # For histograms and boxplots, we usually want continuous or discrete numerical variables.
    # Let's assume all columns except 'target' and potentially others like 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal' which are categorical/binary need careful consideration.
    # For simplicity in this step, we'll plot all.
    numerical_cols = df.columns

    # 2. Create histograms for numeric features
    print("\nGenerating histograms...")
    plt.style.use('seaborn-v0_8-darkgrid') # Use a visually appealing style
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        hist_path = os.path.join(hist_dir, f'{col}_histogram.png')
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved histogram for {col} to {hist_path}")

    # 3. Create boxplots for numeric features
    print("\nGenerating boxplots...")
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}', fontsize=14)
        plt.ylabel(col, fontsize=12)
        plt.tight_layout()
        box_path = os.path.join(box_dir, f'{col}_boxplot.png')
        plt.savefig(box_path)
        plt.close()
        print(f"Saved boxplot for {col} to {box_path}")

    # 4. Use correlation matrix for feature relationships
    print("\nGenerating correlation matrix heatmap...")
    plt.figure(figsize=(14, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Matrix of Features', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    corr_path = os.path.join(eda_plots_dir, 'correlation_matrix.png')
    plt.savefig(corr_path)
    plt.close()
    print(f"Saved correlation matrix heatmap to {corr_path}")

    print("\nEDA plots generated successfully.")

except FileNotFoundError:
    print(f"Error: Cleaned data file not found at {cleaned_data_path}")
except Exception as e:
    print(f"An error occurred during EDA: {e}")

