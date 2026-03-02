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
for col in missing_df[missing_df["Percentage"] > 70].index:
    train_df.drop(columns=[col], inplace=True)

for col in missing_df[missing_df["Percentage"] > 30].index:
    train_df[col + "_IS_MISSING"] = train_df[col].isnull().astype(int)

# credit - income - annuity ratios
train_df["AMT_ANNUITY"] = train_df["AMT_ANNUITY"].replace(np.nan, 0)
train_df["AMT_ANNUITY"] = train_df["AMT_ANNUITY"].replace([np.inf, -np.inf], 0)

train_df["CREDIT_INCOME_RATIO"] = train_df["AMT_CREDIT"] / train_df["AMT_INCOME_TOTAL"]
train_df["ANNUITY_INCOME_RATIO"] = train_df["AMT_ANNUITY"] / train_df["AMT_INCOME_TOTAL"]
train_df["CREDIT_ANNUITY_RATIO"] = np.where(train_df["AMT_ANNUITY"] == 0, 0, train_df["AMT_CREDIT"] / train_df["AMT_ANNUITY"])

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
x = train_df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
y = train_df["TARGET"]

# target imbalance check
print("Target value counts:\n", y.value_counts())
print("Target value proportions:\n", y.value_counts(normalize=True))

# minimal imputation for now
cat_cols = x.select_dtypes(include="object").columns
num_cols = x.select_dtypes(exclude="object").columns
for col in cat_cols:
    x[col] = x[col].fillna(x[col].mode().iloc[0])
for col in num_cols:
    x[col] = x[col].fillna(x[col].median())

print("Missing values after imputation:", x.isnull().sum().sum())

"""
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
"""

# we ll choose kfold instead of train-test, and start with random forest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# encoding categorical variables using one-hot encoding
x = pd.get_dummies(x, columns=cat_cols, drop_first=True)

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
oof_preds = np.zeros(len(x))
feature_importances = np.zeros(x.shape[1])

model = RandomForestClassifier(n_estimators=200, random_state=1)

rf_grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 8, 12],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]}

rf_grid_search = GridSearchCV(
    model,
    rf_grid,
    scoring="roc_auc",
    cv=skfold,
    n_jobs=-1,
    verbose=1)

# manual OOP
fold_scores = []
for fold, (train_idx, val_idx) in enumerate(skfold.split(x, y)):
    
    print(f"Training fold {fold+1}")
    
    x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(x_train, y_train)
    val_preds_proba = model.predict_proba(x_val)[:, 1]
    oof_preds[val_idx] = val_preds_proba
    
    fold_roc_auc = roc_auc_score(y_val, val_preds_proba)
    print(f"Fold {fold+1} ROC AUC: {fold_roc_auc}")
    
    # accumulate feature importance (average by dividing by number of folds)
    feature_importances += model.feature_importances_ / skfold.n_splits
    fold_scores.append(fold_roc_auc)

# Final OOF ROC AUC
final_auc = roc_auc_score(y, oof_preds)
# mean & std deviation auc
print("\nFinal OOF ROC AUC => ", final_auc)
mean_auc = np.mean(fold_scores)
print("Mean Fold ROC AUC: ", mean_auc)
std_auc = np.std(fold_scores)
print("Std Fold ROC AUC: ", std_auc)


# Feature Importance
importance_df = pd.DataFrame({
    "feature": x.columns,
    "importance": feature_importances}).sort_values(by="importance", ascending=False)

print(10*"*" + "Top 15 Important Features" + 10*"*")
print(importance_df.head(15))

# to be continued...