from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np


THRESHOLD = 0.001
TRAIN_DATA_LEN = 40000
TEST_DATA_LEN = 10000
BIGGEST = 2 ** 50


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


def split(dff, split=None, split2=None):
    if split is None:
        split = int(len(dff) / 2)
    if split2 is None:
        split2 = int(len(dff) / 2)
    df1 = dff.iloc[:split, :]
    df2 = dff.iloc[split2 + 1:, :]
    return df1, df2


def accuracy(y, test_y):
    y_arr, test_y_arr = np.array(y), np.array(test_y)
    total = len(y_arr)
    correct = 0
    for i in range(len(y_arr)):
        if y_arr[i] == test_y_arr[i]:
            correct += 1
    return correct / total


def recall(y, test_y):
    y_arr, test_y_arr = np.array(y), np.array(test_y)
    total = 0
    recalled = 0
    for i in range(len(y_arr)):
        if y_arr[i] == 1:
            total += 1
            if test_y_arr[i] == 1:
                recalled += 1
    if total == 0 or recalled == 0:
        if 1 not in y_arr or 1 not in test_y_arr:
            raise Exception('This model predicted 0 for everything')
    return recalled / total


df1 = pd.read_csv('data/train_v2.csv')
df1 = clean_dataset(df1)
print("The len of the dataset after cleaning is: " + str(len(df1)))
# testing_df = pd.read_csv('data/test_v2.csv')
df, testing_df = split(df1, TRAIN_DATA_LEN, TEST_DATA_LEN)

X, y = separate_into_xy(df)
test_X, test_y = separate_into_xy(testing_df)

# clf = RandomForestClassifier(max_depth=5, random_state=0)
# clf = BaggingClassifier(random_state=0)
clf = AdaBoostClassifier(n_estimators=100)

clf.fit(X, y)
z = clf.predict(test_X)

print("predicted labels: " + str(z))
print("real labels: " + str(np.array(test_y)))
print("accuracy: " + str(accuracy(test_y, z)))
print("recall: " + str(recall(test_y, z)))

print("if you got here then something might have worked")



