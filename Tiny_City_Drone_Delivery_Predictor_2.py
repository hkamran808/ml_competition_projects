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

# boosting feature engineering with fare per person, fare&age bins and sex*class interaction
train_df["FarePerPerson"] = (train_df["Fare"] / train_df["family_size"]).fillna(0)
test_df["FarePerPerson"] = (test_df["Fare"] / test_df["family_size"]).fillna(0)

train_df["FPP_bin"] = pd.qcut(
    train_df["FarePerPerson"],
    4,
    labels=["Low", "Mid", "High", "VeryHigh"])

test_df["FPP_bin"] = pd.qcut(
    test_df["FarePerPerson"],
    4,
    labels=["Low", "Mid", "High", "VeryHigh"])

farepp_train = pd.get_dummies(train_df["FPP_bin"], prefix="FPPBin", dtype=int)
farepp_test = pd.get_dummies(test_df["FPP_bin"], prefix="FPPBin", dtype=int)
train_df = pd.concat([train_df, farepp_train], axis=1)
test_df = pd.concat([test_df, farepp_test], axis=1)

train_df.drop(columns=["FPP_bin"], inplace=True)
test_df.drop(columns=["FPP_bin"], inplace=True)


train_df["Age_bin"] = pd.cut(train_df["Age"], 
                             bins=[0, 12, 20, 40, 60, 80], 
                             labels=["Child", "Teen", "Adult", "MidAge", "Senior"])
test_df["Age_bin"] = pd.cut(test_df["Age"], 
                            bins=[0, 12, 20, 40, 60, 80], 
                            labels=["Child", "Teen", "Adult", "MidAge", "Senior"])

age_train_dummies = pd.get_dummies(train_df["Age_bin"], prefix="AgeBin", dtype=int)
age_test_dummies = pd.get_dummies(test_df["Age_bin"], prefix="AgeBin", dtype=int)
train_df = pd.concat([train_df, age_train_dummies], axis=1)
test_df = pd.concat([test_df, age_test_dummies], axis=1)

train_df.drop(columns=["Age_bin"], inplace=True)
test_df.drop(columns=["Age_bin"], inplace=True)


train_df["Fare_bin"] = pd.qcut(train_df["Fare"], 
                               4, 
                               labels=["Low", "Mid", "High", "VeryHigh"])
test_df["Fare_bin"] = pd.qcut(test_df["Fare"], 
                              4, 
                              labels=["Low", "Mid", "High", "VeryHigh"])

fare_train_dummies = pd.get_dummies(train_df["Fare_bin"], prefix="FareBin", dtype=int)
fare_test_dummies = pd.get_dummies(test_df["Fare_bin"], prefix="FareBin", dtype=int)
train_df = pd.concat([train_df, fare_train_dummies], axis=1)
test_df = pd.concat([test_df, fare_test_dummies], axis=1)

train_df.drop(columns=["Fare_bin"], inplace=True)
test_df.drop(columns=["Fare_bin"], inplace=True)


train_df["Sex"] = train_df["Sex"].map({"male": 1, "female": 0})
test_df["Sex"] = test_df["Sex"].map({"male": 1, "female": 0})
train_df["Sex_Pclass"] = train_df["Sex"] * train_df["Pclass"]
test_df["Sex_Pclass"] = test_df["Sex"] * test_df["Pclass"]

train_df["Age_Pclass"] = train_df["Age"] * train_df["Pclass"]
test_df["Age_Pclass"] = test_df["Age"] * test_df["Pclass"]

# for safety
test_df = test_df.reindex(columns=train_df.columns.drop("Survived"), fill_value=0)

# checking correlation of new features with target variable to see if they are useful for us
for col in ["family_size", "isAlone", "FarePerPerson", "Sex_Pclass", "Age_Pclass"]:
    corr = train_df[col].corr(train_df["Survived"])
    print(f"Correlation of {col} with Survived: {corr}")

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
    df.drop(columns=["Embarked", "SibSp", "FPPBin_Low", "family_size", "AgeBin_Adult"], inplace=True)
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
drop_cols = ["Embarked_Q", "Embarked_C", "Title_Mrs", "Embarked_S"]
x = train_df.drop(columns=["Survived"] + drop_cols)
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

param_grid = {
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 3, 4],
    "max_features": ["sqrt", "log2", 0.5],
    "bootstrap": [True, False],
    "class_weight": [None, "balanced"]}

cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# extra trees for feature importance evaluation and potential boosting of the main model (rf) with insights from it in feature importance
et = ExtraTreesClassifier(random_state=1, n_jobs=-1)
et_param_grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 8, 12],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]}

et_grid = GridSearchCV(
    et,
    et_param_grid,
    scoring="roc_auc",
    cv=cv_method,
    n_jobs=-1,
    verbose=1)

et_grid.fit(X_train, Y_train)
print("Extra Trees BEST SCORE: ", et_grid.best_score_)
et_cv_scores = cross_val_score(et, x, y, cv=5, scoring="roc_auc")
print(f"Extra Trees mean ROC AUC CV score (*PRUNED) => {et_cv_scores.mean()}")

# main model: rf
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    max_leaf_nodes=20,
    random_state=1,
    n_jobs=-1)

rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv_method,
    n_jobs=-1,
    verbose=1)

rf_grid.fit(X_train, Y_train)

print("Best parameters:", rf_grid.best_params_)
print("Best CV score:", rf_grid.best_score_)    
best_model = rf_grid.best_estimator_

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
feat_imp.plot(kind="bar")
plt.title("Top Feature Importances - Random Forest")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.tight_layout()
plt.show()

# final evaluation on test set (unseen data)
final_model = best_model
final_model.fit(x, y)
test_df_final = test_df.drop(columns=drop_cols, errors="ignore")  # drop same columns as in training
predictions = final_model.predict(test_df_final)
probabilities = final_model.predict_proba(test_df_final)[:, 1]

# permutation importance for more robust feature importance evaluation
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(final_model, X_test, Y_test, 
                                         n_repeats=10, random_state=1, n_jobs=-1)

perm_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance": perm_importance.importances_mean}).sort_values("importance", ascending=False)

print(perm_df)


# save predictions to csv for submission
passenger_ids = test_df["PassengerId"]
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": predictions
})
submission.to_csv("kamran's_titanic_rf_submission.csv", index=False)
print("Submission file created successfully!", 10*"-", "PROJECT 2 DONE!", sep="\n")