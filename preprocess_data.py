import pandas as pd

data_path = '/home/ubuntu/.cache/kagglehub/datasets/johnsmith88/heart-disease-dataset/versions/2/heart.csv'
cleaned_data_path = '/home/ubuntu/heart_cleaned.csv'

try:
    df = pd.read_csv(data_path)
    print(f"Original dataset shape: {df.shape}")

    # Check for duplicates
    duplicate_rows = df.duplicated().sum()
    print(f"Number of duplicate rows found: {duplicate_rows}")

    if duplicate_rows > 0:
        df_cleaned = df.drop_duplicates()
        print(f"Dataset shape after removing duplicates: {df_cleaned.shape}")
    else:
        df_cleaned = df
        print("No duplicate rows to remove.")

    # Verify no missing values (already checked, but good practice)
    missing_values = df_cleaned.isnull().sum().sum()
    if missing_values == 0:
        print("Confirmed: No missing values in the cleaned dataset.")
    else:
        print(f"Warning: Found {missing_values} missing values after cleaning.")

    # Display info of the cleaned data
    print("\nCleaned Dataset Info:")
    df_cleaned.info()

    # Save the cleaned data
    df_cleaned.to_csv(cleaned_data_path, index=False)
    print(f"\nCleaned dataset saved to {cleaned_data_path}")

except FileNotFoundError:
    print(f"Error: File not found at {data_path}")
except Exception as e:
    print(f"An error occurred during preprocessing: {e}")

