from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np


THRESHOLD = 0.001
DATA_LEN = 20


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def separate_into_xy(dff):
    dff['loss_geq_zero'] = np.where(dff['loss'] > THRESHOLD, 1, 0)
    X = dff.drop(columns=['loss', 'loss_geq_zero'])
    y = dff['loss_geq_zero']
    return X, y


def split(dff, split=None):
    if split is None:
        split = int(len(dff) / 2)
    print(split)
    df1 = dff.iloc[:split, :]
    df2 = dff.iloc[split + 1:, :]
    return df1, df2


df1 = pd.read_csv('data/train_v2.csv')
df1 = clean_dataset(df1)
# testing_df = pd.read_csv('data/test_v2.csv')
df, testing_df = split(df1)

df = df.head(DATA_LEN)
testing_df = testing_df.head(DATA_LEN)

X, y = separate_into_xy(df)
test_X, test_y = separate_into_xy(testing_df)

X = X.reset_index()
y = y.reset_index()

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

print("something worked")



