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

# Converting direction to vectors
for df in [train_df, test_df]:
    df[["wind_direction", "wind_speed", "wind_gust"]] = df["wind"].str.split("_", expand=True)
    df["wind_direction"] = pd.to_numeric(df["wind_direction"], errors="coerce")
    df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
    df["wind_gust"] = pd.to_numeric(df["wind_gust"], errors="coerce")
    df.drop(columns=["wind"], inplace=True)

import ast
for df in [train_df, test_df]:
    df["route"] = df["route"].apply(ast.literal_eval)

def route_features(route_list):
    route_len = len(route_list)
    route_sum = sum(route_list)
    route_abs_sum = sum(abs(x) for x in route_list)
    route_zero_ratio = route_list.count(0) / len(route_list) if len(route_list) > 0 else 0
    route_positive_ratio = sum(1 for x in route_list if x > 0) / len(route_list) if len(route_list) > 0 else 0
    route_negative_ratio = sum(1 for x in route_list if x < 0) / len(route_list) if len(route_list) > 0 else 0

    return pd.Series([route_len, route_sum, route_abs_sum,
                      route_zero_ratio, route_positive_ratio, route_negative_ratio])


route_features_list = ["route_len", "route_sum", "route_abs_sum", 
                       "route_zero_ratio", "route_positive_ratio", "route_negative_ratio"]
for df in [train_df, test_df]:
    df[route_features_list] = df["route"].apply(route_features)
    df.drop(columns=["route"], inplace=True)

train_df["net_distance"] = train_df["route_sum"].abs()
test_df["net_distance"] = test_df["route_sum"].abs()

# Altitude sequences and basic feature exploration
for df in [train_df, test_df]:
    df["altitude"] = df["altitude"].apply(ast.literal_eval)

def altitude_features(altitude_list):
    altitude_len = len(altitude_list)
    altitude_max = max(altitude_list)
    altitude_min = min(altitude_list)
    altitude_range = altitude_max - altitude_min
    altitude_mean = sum(altitude_list) / altitude_len if altitude_len > 0 else 0
    altitude_std = np.std(altitude_list)

    diffs = [altitude_list[i+1] - altitude_list[i] for i in range(len(altitude_list)-1)]
    altitude_upward_sum = sum(abs(x) for x in diffs if x > 0)
    altitude_downward_sum = abs(sum(x for x in diffs if x < 0))

    return pd.Series([altitude_len, altitude_max, altitude_min, 
                      altitude_range, altitude_mean, altitude_std, 
                      altitude_upward_sum, altitude_downward_sum])

# modeling part
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

x = train_df.drop("on_time", axis=1)
y = train_df["on_time"]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)
model = RandomForestClassifier(n_estimators=100, random_state=0)

X_all = pd.concat([X_train, X_test], axis=0)
# checking for categorical columns (if any) to apply encoding in process
# dropping non-hashable columns (like lists) for now, since .get_dummies can't handle them
list_cols = [
    col for col in X_all.columns
    if X_all[col].apply(lambda x: isinstance(x, list)).any()]
X_all = X_all.drop(columns=list_cols)

categorical_cols = X_all.select_dtypes(include=["object", "category"]).columns
X_all = pd.get_dummies(X_all, columns=categorical_cols, drop_first=True)

X_train = X_all.iloc[:len(X_train)]
X_test = X_all.iloc[len(X_train):]

model.fit(X_train, Y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy)
print("classification_report: ", classification_report(Y_test, predictions))
print("confusion_matrix: ", confusion_matrix(Y_test, predictions))
print(10*"-", "Modeling completed!", 10*"-")

# Generating predictions for the test dataset too to not get error (after applying same preprocessing)
test_df_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)
test_df_encoded = test_df_encoded.reindex(columns=X_train.columns, fill_value=0)
full_dataset_predictions = model.predict(test_df_encoded)
print("Predictions for test dataset: [should've been saved to submit if it was for a competition] \n", full_dataset_predictions)

# feature importance from RF model
importances_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.barplot(data=importances_df, x="Importance", y="Feature")
plt.title("Feature Importances from Random Forest Model")
plt.show()

# hyperparameter tuning with cross-validation
print(10*"-", "After Hyperparameters & Cross-Validation", 10*"-")
from sklearn.model_selection import cross_val_score

max_depth_list = [5, 10, 15, 20]
min_samples_list = [2, 5, 10]
#scores = []

best_score = 0
best_params = None

for max_d in max_depth_list:
    for min_s in min_samples_list:
        model = RandomForestClassifier(
            max_depth=max_d,
            min_samples_split=min_s,  # or min_samples_leaf, etc...
            n_estimators=200,
            random_state=42
        )

        cv_score = cross_val_score(model, X_train, Y_train, cv=5, scoring="roc_auc").mean()
        if cv_score > best_score:
            best_score = cv_score
            best_params = (max_d, min_s)
        #scores.append(cv_score)
        print(f"max_depth: {max_d}, min_samples_split: {min_s}, Cross-validated AUC score: {cv_score}")
#print("Best Cross-validated score: ", max(scores)) #optional
print("Best Cross-validated score: ", best_score)
print("Best hyperparameters: ", best_params)

# we retrain the model on the full training data and generate predictions for the test set with best hyperparameters
final_model = RandomForestClassifier(
    max_depth = 5,
    min_samples_split = 5,
    n_estimators = 200,
    random_state = 0)

final_model.fit(X_train, Y_train)

best_predictions = final_model.predict(test_df_encoded)
best_probabilities = final_model.predict_proba(test_df_encoded)[:, 1]  # for ROC-AUC / submission (if needed...)
print("Modeling procedures are compeletely done!")