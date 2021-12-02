import pandas as pd
import numpy as np

THRESHOLD = 2.0
TRAIN_DATA_LEN = 10000
TEST_DATA_LEN = 10000
BIGGEST = 2 ** 100


# Below ONLY returns the training set - still need to find whether the testing set can have labels
def import_all_data():
    df1 = pd.read_csv('data/train_v2.csv')
    return df1


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    # indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    df = df.astype(float)
    # df = df[indices_to_keep].astype(np.float64)
    df = df.reset_index()
    # df = np.nan_to_num(df)
    # print(df.max(axis=1))
    # df.drop( > BIGGEST].index, inplace=True)
    df = df.loc[:, (df.max() < BIGGEST)]
    return df


def separate_into_xy(dff):
    dff['loss_geq_zero'] = np.where(dff['loss'] > THRESHOLD, 1, 0)
    X = dff.drop(columns=['loss', 'loss_geq_zero'])
    y = dff['loss_geq_zero']
    return X, y


def separate_into_xy_regression_case(dff):
    X = dff.drop(columns=['loss'])
    y = dff['loss']
    return X, y


def split(dff, split=None, split2=None):
    if split is None:
        split = int(len(dff) / 2)
    if split2 is None:
        split2 = int(len(dff) / 2)
    df1 = dff.iloc[:split, :]
    df2 = dff.iloc[len(dff) - split2 + 1:, :]
    return df1, df2
