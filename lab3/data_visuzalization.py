
import seaborn as sns
import numpy as np
import pandas as pd

dataset = sns.load_dataset('titanic')

# print(type(dataset))

null_columns=dataset.columns[dataset.isnull().any()]

# print("null_columns: ", null_columns)
print(dataset[dataset.isnull().any(axis=1)][null_columns].head())

null_row_index = dataset[dataset.isnull().any(axis=1)][null_columns].index

dataset = dataset.drop(index=null_row_index)

print(dataset[dataset.isnull().any(axis=1)][null_columns])

print(dataset.columns[dataset.isnull().any()])
