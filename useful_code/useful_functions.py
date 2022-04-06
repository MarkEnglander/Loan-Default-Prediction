import pandas as pd
import numpy as np
from sklearn.utils import resample

THRESHOLD = 2.0
TRAIN_DATA_LEN = 10000
TEST_DATA_LEN = 10000
BIGGEST = 2 ** 100


# Below ONLY returns the training set - still need to find whether the testing set can have labels
def import_all_data():
    df1 = pd.read_csv('../data/train_v2.csv')
    return df1


def clean_dataset(df, delete_missing_data=True):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    if delete_missing_data:
        df.dropna(inplace=True)
        df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    else:
        df.replace('NA', np.nan, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
    df = df.astype(float)
    df = df.reset_index()
    df = df.loc[:, (df.max() < BIGGEST)]
    return df


# VERY IMPORTANT: cannot have 'loss' and 'loss_geq_zero' column existing in the same dataframe
def create_loss_col(dff, threshold=THRESHOLD):
    dff['loss_geq_zero'] = np.where(dff['loss'] > threshold, 1, 0)
    dff = dff.drop(columns=['loss'])
    return dff


# VERY IMPORTANT: cannot have 'loss' and 'loss_geq_zero' column existing in the same dataframe
def separate_into_xy(dff):
    dff['loss_geq_zero'] = np.where(dff['loss'] > THRESHOLD, 1, 0)
    X = dff.drop(columns=['loss', 'loss_geq_zero'])
    y = dff['loss_geq_zero']
    return X, y


def separate_into_xy_loss_exists(dff):
    X = dff.drop(columns=['loss_geq_zero'])
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


def upsample_minority(df):
    df_majority = df[df['loss_geq_zero'] == 0]
    df_minority = df[df['loss_geq_zero'] == 1]

    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority),  # to match majority class
                                     random_state=123)  # reproducible results

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    df = df_upsampled
    return df


def downsample_majority(df):
    df_majority = df[df['loss_geq_zero'] == 0]
    df_minority = df[df['loss_geq_zero'] == 1]

    df_majority_downsampled = resample(df_majority,
                                       replace=True,  # sample with replacement
                                       n_samples=len(df_minority),
                                       random_state=123)  # reproducible results

    df_downsampled = pd.concat([df_minority, df_majority_downsampled])
    df = df_downsampled
    return df