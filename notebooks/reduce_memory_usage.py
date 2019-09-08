#%% Import Libraries

import numpy as np
import pandas as pd

#%% Define Functions

# https://www.kaggle.com/kabure/almost-complete-feature-engineering-ieee-data
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


#%% Read & Denormalize Data

train_id_df = pd.read_csv("./data/train_identity.csv")
train_trans_df = pd.read_csv("./data/train_transaction.csv")

test_id_df = pd.read_csv("./data/test_identity.csv")
test_trans_df = pd.read_csv("./data/test_transaction.csv")

train_df = train_trans_df.merge(train_id_df, on="TransactionID", how="left")
test_df = test_trans_df.merge(test_id_df, on="TransactionID", how="left")

del train_id_df, train_trans_df, test_id_df, test_trans_df

print("train_df", train_df.shape)
print("test_df", test_df.shape)

#%% Reduce Memory Usage

train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)

#%% Save to Pickle

train_df.to_pickle("./data/train_df.pkl")
test_df.to_pickle("./data/test_df.pkl")
