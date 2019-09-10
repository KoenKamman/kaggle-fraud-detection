#%% Import Libraries

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split

#%% Read Data

train_df = pd.read_pickle("./data/train_df.pkl")
test_df = pd.read_pickle("./data/test_df.pkl")

print("train_df", train_df.shape)
print("test_df", test_df.shape)

#%% Select Columns

variables = ["addr1", "card4",  "card6", "R_emaildomain", "P_emaildomain", "isFraud"]

sub_train_df = train_df[variables]
variables.remove("isFraud")
sub_test_df = test_df[variables]

print("train_df", sub_train_df.shape)
print("test_df", sub_test_df.shape)

#%%

combined_df = pd.concat([sub_train_df, sub_test_df], keys=['train', 'test'])
combined_df.head()

#%%

combined_df_missing = combined_df[variables].isnull().astype(int).add_suffix('_missing')

# Concatenate the two new columns to the data frame
combined_df = pd.concat([combined_df, combined_df_missing], axis=1)

#%%
sub_train_df.head().transpose()

#%%
combined_df = pd.get_dummies(combined_df, columns=["card4", "card6", "R_emaildomain", "P_emaildomain", "addr1"])

#%%

sub_train_df = combined_df.loc['train']
sub_test_df = combined_df.loc['test']

#%% Split DataFrame into Training and Validation Set

training_set, validation_set = train_test_split(sub_train_df, test_size=0.2)

y_train = training_set['isFraud']
X_train = training_set.drop('isFraud', axis=1)
y_validation_set = validation_set['isFraud']
X_validation_set = validation_set.drop('isFraud', axis=1)

print("y_train", y_train.shape)
print("X_train", X_train.shape)
print("y_validation_set", y_validation_set.shape)
print("X_validation_set", X_validation_set.shape)

#%% Train Model
model = RandomForestRegressor(
    n_estimators=400, max_features=0.3,
    min_samples_leaf=20, n_jobs=-1, verbose=1)

model.fit(X_train, y_train)

preds_valid = model.predict(X_validation_set)

roc_auc_score(y_validation_set, preds_valid), mean_squared_error(y_validation_set, preds_valid) 

#%% 
preds_train = model.predict(X_train)
roc_auc_score(y_train, preds_train), mean_squared_error(y_train, preds_train)

#%% 
sub_test_df.isna().sum()

#%%
sub_test_df = sub_test_df.drop('isFraud', axis=1)

#%%
preds_test = model.predict(sub_test_df)

#%%
hand_in_sample = pd.read_csv('./data/sample_submission.csv')
hand_in_sample['isFraud'] = preds_test
hand_in_sample.to_csv('./data/hand_in_submission.csv', index=False)

#%%
