"""
# Title: Project 2 - Tiny City Drone Delivery Predictor 2
# Purpose: Load Project 2 datasets, perform sanity checks, inspect data types, identify potential issues, prepare for feature engineering
"""
import re
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

# Title Extraction with regex for potential feature engineering
import re
"""
for row in train_df["Name"]:
    title = re.search(" ([A-Za-z]+)\.", row)
    if title:
        print(f"Extracted title from '{row}': {title.group(1)}")
    else:
        print(f"No title found in '{row}'")
"""
train_df["Title"] = train_df["Name"].apply(lambda x: re.search(" ([A-Za-z]+)\.", x).group(1) if re.search(" ([A-Za-z]+)\.", x) else "Unknown")
test_df["Title"] = test_df["Name"].apply(lambda x: re.search(" ([A-Za-z]+)\.", x).group(1) if re.search(" ([A-Za-z]+)\.", x) else "Unknown")
print("Unique titles in train_df: ", train_df["Title"].value_counts())
print("Unique titles in test_df: ", test_df["Title"].value_counts())

count_rare_titles = train_df["Title"].value_counts()[train_df["Title"].value_counts() < 10].index
count_rare_titles = test_df["Title"].value_counts()[test_df["Title"].value_counts() < 10].index

for df in [train_df, test_df]:
    for row in df["Title"]:
        if row in count_rare_titles:
            print(f"Others: {row}")

# family size feature engineering
train_df["family_size"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["family_size"] = test_df["SibSp"] + test_df["Parch"] + 1

is_Alone = train_df["family_size"] == 1
train_df["isAlone"] = is_Alone.astype(int)
is_Alone_test = test_df["family_size"] == 1
test_df["isAlone"] = is_Alone_test.astype(int)

corr_family_train = train_df["family_size"].corr(train_df["Survived"])
print(f"Correlation of family_size with Survived: {corr_family_train}")
corr_isAlone_train = train_df["isAlone"].corr(train_df["Survived"])
print(f"Correlation of isAlone with Survived: {corr_isAlone_train}")

train_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
test_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)