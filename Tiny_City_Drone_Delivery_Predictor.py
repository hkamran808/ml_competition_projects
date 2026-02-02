import pandas as pd
import numpy as np

train_df = pd.read_csv("project1_train.csv")
test_df = pd.read_csv("project1_test.csv")

print(train_df.head())
print(train_df.info())

# Dataset size & sanity check
print("Missing values for train:\n", train_df.isna().sum())
print("Missing values for test:\n", test_df.isna().sum())
print("Duplicating values for train:\n", train_df.duplicated().sum())
print("Duplicating values for test:\n", test_df.duplicated().sum())
    
# Mismatch in route and altitude lengths in data
mismatch_idx_train = train_df[
    train_df["route"].str.split(",").apply(len) + 1
    != train_df["altitude"].str.split(",").apply(len)].index

mismatch_idx_test = test_df[
    test_df["route"].str.split(",").apply(len) + 1
    != test_df["altitude"].str.split(",").apply(len)].index

print(f"Rows with mismatched route & altitude lengths for train data {mismatch_idx_train}, and test data {mismatch_idx_test}")
# Removing rows with mismatched lengths
train_df = train_df.drop(index=mismatch_idx_train).reset_index(drop=True)
test_df = test_df.drop(index=mismatch_idx_test).reset_index(drop=True)

# Sanity checks after cleaning
assert (
    train_df["route"].str.split(",").apply(len) + 1
    == train_df["altitude"].str.split(",").apply(len)).all()

assert (
    test_df["route"].str.split(",").apply(len) + 1
    == test_df["altitude"].str.split(",").apply(len)).all()

assert (train_df["route"].str.len() > 0).all()
assert (test_df["route"].str.len() > 0).all()
assert (train_df["altitude"].str.len() > 0).all()
assert (test_df["altitude"].str.len() > 0).all()

# Searching error in wind token in data
assert train_df["wind"].str.split("_").apply(len).eq(3).all()
assert test_df["wind"].str.split("_").apply(len).eq(3).all()

assert train_df["wind"].dtype == object
assert test_df["wind"].dtype == object

# Negative or zero distances or masses
numeric_cols = ["distance", "mass"]
for col in numeric_cols:
    assert train_df[col].notna().all()
    assert test_df[col].notna().all()
    assert (train_df[col] > 0).all()
    assert (test_df[col] > 0).all()

constant_cols = [
    col for col in train_df.columns
    if train_df[col].nunique() == 1]
print("Constant columns:", constant_cols)

# Train-test columns sanity check
assert set(train_df.columns) - {"on_time"} == set(test_df.columns)
print("Shape of datasets: ", train_df.shape, test_df.shape)
print("All sanity checks passed!")