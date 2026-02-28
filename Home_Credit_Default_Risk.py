"""
Project 3: Home Credit Default Risk

- Dataset: application_train.csv
        application_test.csv
        more to be added...

- Goal: Building perfect model using pipeline and feature engineering correctly. learning all possible outcomes and reasons to be better at competitions and such working environments
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_df = pd.read_csv("application_train.csv")
test_df = pd.read_csv("application_test.csv")

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)
print("Training data columns:", train_df.columns)
print("Testing data columns:", test_df.columns)
print(train_df.head())

# Check for missing values
print("Missing values in training data:\n", train_df.isnull().sum())
print("Missing values in testing data:\n", test_df.isnull().sum())

print("Categorical features in training data:\n", train_df.select_dtypes(include=["object"]).columns)
print("Categorical features in testing data:\n", test_df.select_dtypes(include=["object"]).columns)
print("Numerical features in training data:\n", train_df.select_dtypes(include=["number"]).columns)
print("Numerical features in testing data:\n", test_df.select_dtypes(include=["number"]).columns)

# Missing Value Intelligence (per column)
missing_values = train_df.isnull().sum()
missing_percentages = (missing_values / len(train_df)) * 100
missing_values_df = pd.DataFrame({"Missing Count": missing_values, "Percentage": missing_percentages})
missing_df = missing_values_df[missing_values_df["Missing Count"] > 0].sort_values(by="Percentage", ascending=False)
print(10*"-", "Missing values per column in training data (sorted by percentage)", 10*"-" + "\n", missing_df)

# For each column with >30% missing: we can consider dropping them or using advanced imputation techniques
for col in missing_df[missing_df["Percentage"] > 30].index:
    train_df[col + "_IS_MISSING"] = train_df[col].isnull().astype(int)

# credit - income - annuity ratios
train_df["CREDIT_INCOME_RATIO"] = train_df["AMT_CREDIT"] / train_df["AMT_INCOME_TOTAL"]
train_df["ANNUITY_INCOME_RATIO"] = train_df["AMT_ANNUITY"] / train_df["AMT_INCOME_TOTAL"]
train_df["CREDIT_ANNUITY_RATIO"] = train_df["AMT_CREDIT"] / train_df["AMT_ANNUITY"]

# income per child & employment ratio (with anomaly value)
train_df["INCOME_PER_CHILD"] = train_df["AMT_INCOME_TOTAL"] / (train_df["CNT_CHILDREN"] + 1)
train_df["EMPLOYED_ANOMALY"] = (train_df["DAYS_EMPLOYED"] == 365243).astype(int)
train_df["DAYS_EMPLOYED"] = train_df["DAYS_EMPLOYED"].replace(365243, np.nan)
train_df["DAYS_EMPLOYED_RATIO"] = (train_df["DAYS_EMPLOYED"]).abs() / (train_df["DAYS_BIRTH"]).abs()

if train_df.columns.str.startswith("FLAG_DOCUMENT_").any():
    columns = train_df.columns[train_df.columns.str.startswith("FLAG_DOCUMENT_")]
    train_df[columns] = train_df[columns].abs()
    train_df["TOTAL_DOCUMENT_FLAGS"] = train_df[columns].sum(axis=1)

# redefining target variable and features after doing it pre feature engineering for checking the target imbalance and such
x = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
y = train_df["TARGET"]

# target imbalance check
print("Target value counts:\n", y.value_counts())
print("Target value proportions:\n", y.value_counts(normalize=True))

# Handling missing values: Impute with median for numerical features and mode for categorical features if the percentage of missing values is less than 30%
columns = missing_df[missing_df["Percentage"] <= 30]
if columns.columns.dtypes == "object":
    train_df[columns] = train_df[columns].fillna(train_df[columns].mode().iloc[0])
elif columns.columns.dtypes == "number":
    train_df[columns] = train_df[columns].fillna(train_df[columns].median())

"""
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
"""