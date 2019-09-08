#%% Import Libraries

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

#%% Read Data

train_df = pd.read_pickle("./data/train_df.pkl")
test_df = pd.read_pickle("./data/test_df.pkl")

print("train_df", train_df.shape)
print("test_df", test_df.shape)

#%% Select Columns

variables = ["TransactionDT", "TransactionAmt", "isFraud"]

sub_train_df = train_df[variables]
variables.remove("isFraud")
sub_test_df = test_df[variables]

print("train_df", sub_train_df.shape)
print("test_df", sub_test_df.shape)

#%%

# DO SOME MORE STUFF HERE
# DO SOME MORE STUFF HERE
# DO SOME MORE STUFF HERE
# DO SOME MORE STUFF HERE

#%% Split DataFrame into Training and Validation Set

idx = int(len(sub_train_df) * 0.8)
training_set, validation_set = sub_train_df[:idx], sub_train_df[idx:]

y_train = training_set['isFraud']
X_train = training_set.drop('isFraud', axis=1)
y_valid = validation_set['isFraud']
X_valid = validation_set.drop('isFraud', axis=1)

print("y_train", y_train.shape)
print("X_train", X_train.shape)
print("y_valid", y_valid.shape)
print("X_valid", X_valid.shape)

#%% Train Model

training_sample = training_set[-100000:]
y_train_sample = training_sample['isFraud']
X_train_sample = training_sample.drop('isFraud', axis=1)

model = RandomForestRegressor(
    n_estimators=400, max_features=0.3,
    min_samples_leaf=20, n_jobs=-1, verbose=1)

model.fit(X_train_sample, y_train_sample)

preds_valid = model.predict(X_valid)

roc_auc_score(y_valid, preds_valid)

#%%
