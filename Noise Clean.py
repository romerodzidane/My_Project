import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the dataset (assuming the data is in a CSV file named 'your_dataset.csv')
df = pd.read_csv('your_dataset.csv')

# Step 2: Handling Missing Values
# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# Fill missing values with mean or median
df.fillna(df.mean(), inplace=True)  # If any missing values

# Step 3: Outlier Detection and Treatment
# Visualize potential outliers (box plot, scatter plot, etc.)

# Identify the numerical columns in the DataFrame
numerical_columns = df.select_dtypes(include=[np.number]).columns

# Visualize Outliers using Box Plots
plt.figure(figsize=(10, 6))
df[numerical_columns].boxplot()
plt.title('Box Plot of Numerical Features')
plt.xticks(rotation=45)
plt.show()

# Visualize Outliers using Scatter Plots (for each pair of numerical features)
plt.figure(figsize=(12, 8))
for i in range(len(numerical_columns)):
    for j in range(i + 1, len(numerical_columns)):
        plt.scatter(df[numerical_columns[i]], df[numerical_columns[j]])
        plt.xlabel(numerical_columns[i])
        plt.ylabel(numerical_columns[j])
        plt.title(f'Scatter Plot of {numerical_columns[i]} vs {numerical_columns[j]}')
        plt.show()

# Remove outliers using the IQR method (example):
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Step 4: Data Standardization or Normalization (optional)
# Standardize or normalize numerical features (example):
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()  # or MinMaxScaler()
df[['numerical_feature1', 'numerical_feature2']] = scaler.fit_transform(df[['numerical_feature1', 'numerical_feature2']])

# Step 5: Feature Selection and Engineering (optional)
# Select relevant features for analysis and model training
# Engineer new features if necessary

# Step 6: Handling Categorical Variables (optional)
# Convert categorical variables to numerical representations
# using techniques like one-hot encoding or label encoding
# Example for one-hot encoding:
df = pd.get_dummies(df, columns=['categorical_column'])

# Step 7: Handling Duplicate Data
# Identify and remove duplicate rows if any
df = df.drop_duplicates()

# Step 8: Data Smoothing (optional)
# Apply data smoothing techniques to reduce noisy fluctuations
# Example of moving average smoothing (window size = 3):
df['smoothed_column'] = df['column'].rolling(window=3).mean()

# Step 9: Feature Scaling (optional)
# Scale numerical features to a similar range (if needed)
# Example of Min-Max scaling:
df['scaled_column'] = (df['column'] - df['column'].min()) / (df['column'].max() - df['column'].min())

# Step 10: Feature Transformation (optional)
# Apply feature transformations (log, square root, polynomial, etc.)
# Example of log transformation:
df['transformed_column'] = np.log(df['column'])

# Step 11: Data Visualization (optional)
# Visualize the data to identify patterns, trends, and noisy data points

# Step 12: Save the Cleaned Dataset (optional)
# Save the cleaned DataFrame to a new CSV file
df.to_csv('cleaned_data.csv', index=False)

# After cleaning the data, you can proceed with your analysis and modeling.

# Example of plotting a histogram for a numerical feature
plt.hist(df['numerical_feature'], bins=20)
plt.xlabel('Numerical Feature')
plt.ylabel('Frequency')
plt.title('Histogram of Numerical Feature')
plt.show()