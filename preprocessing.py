# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats

# Step 2: Load the dataset with error handling
try:
    # Attempt to load the CSV file with error handling
    df = pd.read_csv('a.csv', on_bad_lines='skip', low_memory=False)
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Step 3: Removing Duplicates
initial_shape = df.shape
df = df.drop_duplicates()
print(f"Removed duplicates. Shape changed from {initial_shape} to {df.shape}")

# Step 4: Handling Missing Values
# For numerical columns, we fill missing values with the mean.
# For categorical columns, we drop rows with missing values for simplicity.

# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Handle missing values in numerical columns by replacing with the mean
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Handle missing values in categorical columns by dropping rows
df = df.dropna(subset=cat_cols)

# Step 5: Outlier Detection using Z-Score and handling skewness
# Z-score threshold of 3 for detecting outliers in numerical columns
z_scores = np.abs(stats.zscore(df[num_cols]))
df = df[(z_scores < 3).all(axis=1)]
print(f"Outliers removed. Shape after outlier removal: {df.shape}")

# Step 6: Data Type Conversion
# Convert columns to their appropriate types if needed
# For example, some numerical columns might be read as objects
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 7: Removing Irrelevant Columns
# Remove columns that are irrelevant based on domain knowledge or analysis
# Example: If we have a column with a single unique value
for col in df.columns:
    if df[col].nunique() == 1:
        df.drop(col, axis=1, inplace=True)
        print(f"Removed irrelevant column: {col}")

# Step 8: Fixing Structural Errors
# Check for any structural errors (e.g., typos, inconsistent formatting)
# This can involve inspecting categorical columns
for col in cat_cols:
    df[col] = df[col].str.strip().str.lower()

# Step 9: Standardizing and Normalizing Numerical Data
# Standardizing numerical data (z-score normalization)
df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()

# Step 10: Text Cleaning (for any textual columns)
# Removing unwanted characters, formatting from text columns
for col in cat_cols:
    df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)

# Step 11: Scaling and Normalizing the data
# Min-Max Scaling numerical data between 0 and 1
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Step 12: Final dataset summary
print(f"Final dataset shape: {df.shape}")
print(f"Null values in the final dataset:\n{df.isnull().sum()}")
