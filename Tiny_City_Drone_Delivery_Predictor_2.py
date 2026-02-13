"""
# Title: Project 2 - Tiny City Drone Delivery Predictor 2
# Purpose: Load Project 2 datasets, perform sanity checks, inspect data types, identify potential issues, prepare for feature engineering
"""
import pandas as pd
import numpy as np

train_df = pd.read_csv("project2_train.csv")
test_df = pd.read_csv("project2_test.csv")

# Quick look at the datasets
print(train_df.head())
print(train_df.info())
print(test_df.head())
print(test_df.info())

# Sanity checks: missing values & duplicates
print("Missing values in train: \n", train_df.isna().sum())
print("Missing values in test: \n", test_df.isna().sum())
print("Duplicate rows in train: ", train_df.duplicated().sum())
print("Duplicate rows in test: ", test_df.duplicated().sum())

# definining column types and checking for potential issues
numeric_cols = [col for col in train_df.select_dtypes(include=np.number).columns]
for col in numeric_cols:
    print(f"{col} stats: min={train_df[col].min()}, max={train_df[col].max()}")

categorical_cols = [col for col in train_df.select_dtypes(include="object").columns]
for col in categorical_cols:
    print(f"{col} unique values:", train_df[col].unique())

sequence_cols = [col for col in train_df.columns if train_df[col].dtype == object and train_df[col].str.contains(",").any()]
print("Columns with potential sequences:", sequence_cols)

constant_cols = [col for col in train_df.columns if train_df[col].nunique() == 1]
print("Constant columns:", constant_cols)

assert set(train_df.columns) - {"Survived"} == set(test_df.columns), "Train and test columns mismatch!"

print("Basic sanity checks DONE successfully!")