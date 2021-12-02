from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

from useful_functions import clean_dataset, separate_into_xy, split
from evaluation_functions import accuracy, recall


THRESHOLD = 2.0
TRAIN_DATA_LEN = 10000
TEST_DATA_LEN = 10000
BIGGEST = 2 ** 100


def test_1():
    df1 = pd.read_csv('data/train_v2.csv')
    df1 = clean_dataset(df1)
    print("The len of the dataset after cleaning is: " + str(len(df1)))
    # testing_df = pd.read_csv('data/test_v2.csv')
    df, testing_df = split(df1, TRAIN_DATA_LEN, TEST_DATA_LEN)

    X, y = separate_into_xy(df)
    test_X, test_y = separate_into_xy(testing_df)

    # clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf = BaggingClassifier(random_state=0)
    # clf = AdaBoostClassifier(n_estimators=100)
    # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

    clf.fit(X, y)
    z = clf.predict(test_X)

    print("predicted labels: " + str(z))
    print("real labels: " + str(np.array(test_y)))
    print("accuracy: " + str(accuracy(test_y, z)))
    print("recall: " + str(recall(test_y, z)))




