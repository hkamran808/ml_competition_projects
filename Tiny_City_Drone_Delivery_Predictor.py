import pandas as pd
import numpy as np

train_df = pd.read_csv("project1_train.csv")
test_df = pd.read_csv("project1_test.csv")

print(train_df.head())
print(train_df.info())