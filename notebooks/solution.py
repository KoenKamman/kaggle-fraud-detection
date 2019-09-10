#%%
# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split


#%%
# We used another script to reduce the memory size by 60 percent
train_df = pd.read_pickle("./data/train_df.pkl")
test_df = pd.read_pickle("./data/test_df.pkl")

print("train_df", train_df.shape)
print("test_df", test_df.shape)


#%%
# These are the features we used. We had a 
variables = ["addr1", "card4",  "card6", "R_emaildomain", "P_emaildomain", "M4",  "isFraud"]

# We only want the datasets with these features in it
sub_train_df = train_df[variables]
variables.remove("isFraud")
sub_test_df = test_df[variables]

print("train_df", sub_train_df.shape)
print("test_df", sub_test_df.shape)


#%%
# We combined the two dataframes to have the keep the same structure
combined_df = pd.concat([sub_train_df, sub_test_df], keys=['train', 'test'])
combined_df.head()


#%%
# We made columns for the missing/empty cells
combined_df_missing = combined_df[variables].isnull().astype(int).add_suffix('_missing')

# Concatenate the two new columns to the data frame
combined_df = pd.concat([combined_df, combined_df_missing], axis=1)


#%%
# To get a little bit of overview. We want to transpose it, so that it's easier to be read
sub_train_df.head().transpose()


#%%
# Create dummies for the relevant columns
combined_df = pd.get_dummies(combined_df, columns=variables)


#%%
# Split the dataframes up again, after the data manipulation is finished
sub_train_df = combined_df.loc['train']
sub_test_df = combined_df.loc['test']


#%%
# Split it into a training and a validation set
training_set, validation_set = train_test_split(sub_train_df, test_size=0.2)

# Create the answer to compare to
y_train = training_set['isFraud']
X_train = training_set.drop('isFraud', axis=1)
y_validation_set = validation_set['isFraud']
X_validation_set = validation_set.drop('isFraud', axis=1)

# Print the shapes
print("y_train", y_train.shape)
print("X_train", X_train.shape)
print("y_validation_set", y_validation_set.shape)
print("X_validation_set", X_validation_set.shape)


#%%
# Train a regression forest
model = RandomForestRegressor(
    n_estimators=400, max_features=0.3,
    min_samples_leaf=20, n_jobs=-1, verbose=1)

# Fit the forest based on the training data
model.fit(X_train, y_train)

# Predict based on the validation set
preds_valid = model.predict(X_validation_set)

# Get the probability of correctness and the mean squared error. 
# The Roc_Auc score gives an estimation of what the final Kaggle score will be
roc_auc_score(y_validation_set, preds_valid), mean_squared_error(y_validation_set, preds_valid) 


#%%
# Just to get some more info, see how accurate the training set in the model is
preds_train = model.predict(X_train)
roc_auc_score(y_train, preds_train), mean_squared_error(y_train, preds_train)


#%%
# Check if there are any more NaN cells
sub_test_df.isna().sum()


#%%
# Drop the isFraud column. The column was added on because of the .concat function
sub_test_df = sub_test_df.drop('isFraud', axis=1)


#%%
# Predict based on the test data
preds_test = model.predict(sub_test_df)


#%%
# Create the hand in CSV file
hand_in_sample = pd.read_csv('./data/sample_submission.csv')
hand_in_sample['isFraud'] = preds_test
hand_in_sample.to_csv('./data/hand_in_submission.csv', index=False)
