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

# Title Extraction with regex for feature engineering
import re
"""
# titles:
for row in train_df["Name"]:
    title = re.search(" ([A-Za-z]+)\.", row)
    if title:
        print(f"Extracted title from '{row}': {title.group(1)}")
    else:
        print(f"No title found in '{row}'")
"""
train_df["Title"] = train_df["Name"].apply(lambda x: re.search(" ([A-Za-z]+)\.", x).group(1) if re.search(" ([A-Za-z]+)\.", x) else "Unknown")
test_df["Title"] = test_df["Name"].apply(lambda x: re.search(" ([A-Za-z]+)\.", x).group(1) if re.search(" ([A-Za-z]+)\.", x) else "Unknown")

count_rare_titles = train_df["Title"].value_counts()[train_df["Title"].value_counts() < 10].index

print("Unique titles in train_df: ", train_df["Title"].value_counts())
print("Unique titles in test_df: ", test_df["Title"].value_counts())

train_df["Title"] = train_df["Title"].apply(lambda x: "Rare" if x in count_rare_titles else x)
test_df["Title"] = test_df["Title"].apply(lambda x: "Rare" if x in count_rare_titles else x)

"""
# encoding titles to numeric for modeling (this one is less ideal for rf)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df["Title_encoded"] = le.fit_transform(train_df["Title"])
test_df["Title_encoded"] = le.transform(test_df["Title"])
"""
# better for random forest - encoding titles with one-hot encoding
dummy_train_df = pd.get_dummies(train_df["Title"], prefix="Title", dtype=int)
dummy_test_df = pd.get_dummies(test_df["Title"], prefix="Title", dtype=int)
train_df = pd.concat([train_df, dummy_train_df], axis=1)
test_df = pd.concat([test_df, dummy_test_df], axis=1)
for df in [train_df, test_df]:
    df.drop(columns=["Title"], inplace=True)
    
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

# handling missing values in Age and Embarked for potential modeling
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
train_df["Age"] = imputer.fit_transform(train_df[["Age"]])
test_df["Age"] = imputer.transform(test_df[["Age"]])

imputer_embarked = SimpleImputer(strategy="most_frequent")
train_df["Embarked"] = imputer_embarked.fit_transform(train_df[["Embarked"]])[:,0]
test_df["Embarked"] = imputer_embarked.transform(test_df[["Embarked"]])[:,0]

# categoricals
train_df["Sex"] = train_df["Sex"].map({"male": 1, "female": 0})
test_df["Sex"] = test_df["Sex"].map({"male": 1, "female": 0})

# same for Embarked - one-hot encoding
dummy_train_df1 = pd.get_dummies(train_df["Embarked"], prefix="Embarked", dtype=int)
dummy_test_df1 = pd.get_dummies(test_df["Embarked"], prefix="Embarked", dtype=int)
train_df = pd.concat([train_df, dummy_train_df1], axis=1)
test_df = pd.concat([test_df, dummy_test_df1], axis=1)
for df in [train_df, test_df]:
    df.drop(columns=["Embarked"], inplace=True)
test_df = test_df.reindex(columns=train_df.columns.drop("Survived"), fill_value=0)

"""
# feature scaling for numeric features - not needed for tree-based models (since we now use rf)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_df[["Age", "Fare", "SibSp", "Parch", "family_size"]] = scaler.fit_transform(train_df[["Age", "Fare", "SibSp", "Parch", "family_size"]])
test_df[["Age", "Fare", "SibSp", "Parch", "family_size"]] = scaler.transform(test_df[["Age", "Fare", "SibSp", "Parch", "family_size"]])
"""
# train-test splitting for modeling
from sklearn.model_selection import train_test_split
x = train_df.drop(columns=["Survived"])
y = train_df["Survived"]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

# ... building the model (rf is chosen for now)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
"""
# MANUAL TUNING: with some for loops best n value is determined to be 20 for max_leaf_nodes, 10 for max depth, 200 for n_estimators
model = RandomForestClassifier(n_estimators=200, max_leaf_nodes=20, random_state=1, 
                                   n_jobs=-1, max_depth=10, class_weight="balanced")
cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring="roc_auc")
print(f"BEST mean ROC AUC CV score after tuning => {cv_scores.mean()}")

model.fit(X_train, Y_train)
#model.fit(x, y) fitting on the entire training set for final evaluation on test set
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:,1] #[:, 0] for class 0 probabilities (negatives)
"""
# AUTOMATIC HYPERPARAMETER TUNING with GridSearchCV
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    max_leaf_nodes=20,
    random_state=1,
    n_jobs=-1)

param_grid = {
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
    "class_weight": [None, "balanced"]}

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1)

grid.fit(X_train, Y_train)
print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

best_model = grid.best_estimator_

# Validation ROC AUC evaluation AND other metrics on the validation set
best_model.fit(X_train, Y_train)
val_predictions = best_model.predict(X_test)
val_probabilities = best_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Validation ROC AUC:", roc_auc_score(Y_test, val_probabilities))
print(10*"-", "METRICS of OUR MODEL", 10*"-")
print("Accuracy: ", accuracy_score(Y_test, val_predictions))
print("Classification Report: \n", classification_report(Y_test, val_predictions))
print("Confusion Matrix: \n", confusion_matrix(Y_test, val_predictions))

# feature_importances_
importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=x.columns)
feat_imp = feat_imp.sort_values(ascending=False)

# plotting feature importances
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
feat_imp.head(15).plot(kind="bar")
plt.title("Top Feature Importances - Random Forest")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.tight_layout()
plt.show()

# final evaluation on test set
final_model = best_model
final_model.fit(x, y)
predictions = final_model.predict(test_df)

# permutation importance for more robust feature importance evaluation
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(final_model, X_test, Y_test, 
                                         n_repeats=10, random_state=1, n_jobs=-1)

perm_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance": perm_importance.importances_mean}).sort_values("importance", ascending=False)

print(perm_df)

"""
# save predictions to csv for submission
passenger_ids = test_df["PassengerId"]
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": predictions
})
submission.to_csv("kamran's_titanic_rf_submission.csv", index=False)
print("Submission file created successfully!", 10*"-", "PROJECT 2 DONE!", sep="\n")
"""