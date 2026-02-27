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

# defining target variable and features
x = train_df.drop(columns=["TARGET"])
y = train_df["TARGET"]

# target imbalance check
print("Target value counts:\n", y.value_counts())
print("Target value proportions:\n", y.value_counts(normalize=True))

# Missing Value Intelligence (per column) % missing per column (sorted descending)
missing_values = x.isnull().sum()
missing_percentages = (missing_values / len(x)) * 100
missing_values_df = pd.DataFrame({"Missing Count": missing_values, "Percentage": missing_percentages})
missing_df = missing_values_df[missing_values_df["Missing Count"] > 0].sort_values(by="Percentage", ascending=False)
print(10*"-", "Missing values per column in training data (sorted by percentage)", 10*"-" + "\n", missing_df)

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)