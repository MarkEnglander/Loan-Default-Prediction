import useful_code
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from useful_code import PandasImputer, evaluation_functions
from useful_code import useful_functions

# TEST 4 IS DONE WITH p = 0.5 and cols_w_na_original
# TEST 4.1 IS DONE WITH p = 0.75 and cols_w_na

# print(df.isnull().sum().loc[lambda x : x>0].sort_values(ascending=False))

classifiers = [RandomForestClassifier(n_estimators=200, class_weight='balanced')]

cols_w_na_original = ['f662', 'f663', 'f159', 'f160', 'f170', 'f169', 'f619', 'f618', 'f331', 'f330',
             'f180', 'f179', 'f422', 'f653', 'f189', 'f190', 'f340', 'f341', 'f666', 'f664',
             'f665', 'f669', 'f667', 'f668', 'f726', 'f640', 'f199', 'f200', 'f650', 'f651',
             'f72', 'f587', 'f586', 'f649', 'f648', 'f588', 'f621', 'f620', 'f672', 'f673',
             'f209', 'f210', 'f679']

cols_w_na = ['f662', 'f663', 'f159', 'f160', 'f170', 'f169', 'f619', 'f618', 'f331', 'f330',
             'f180', 'f179', 'f422', 'f653', 'f189', 'f190', 'f340', 'f341', 'f666', 'f664',
             'f665', 'f669', 'f667', 'f668', 'f726', 'f640', 'f199', 'f200', 'f650', 'f651',
             'f72', 'f587', 'f586', 'f649', 'f648', 'f588', 'f621', 'f620', 'f672', 'f673',
             'f209', 'f210', 'f679', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
             'f10', 'f15', 'f16', 'f13', 'f14', 'f300', 'f301', 'f302', 'f303', 'f304',
             'f305', 'f306']


def test_5(strategy='mean', replace_prob=0.5):
    df = useful_functions.import_all_data()
    df = useful_functions.clean_dataset(df)
    df = useful_functions.create_loss_col(df)

    # df can be our dataset that is pure (no synthetic replacement of N/As) (limit to 25000 for sake of laptop + time)
    df = df.head(25000)

    # df2 can be our dataset that has randomly had data deleted and replaced with an imputation method
    # for consistency, anything with '2' in the name refers to this partially synthetic dataset
    df2 = randomly_remove_data(df, cols_w_na, replace_prob=replace_prob)

    df, df_test = useful_functions.split(df, round(0.8 * len(df)))
    df2, df_test2 = useful_functions.split(df2, round(0.8 * len(df)))

    # print('before imputer')
    # print(df2.isnull().sum().loc[lambda x : x>0].sort_values(ascending=False))

    df2 = PandasImputer.PandasImputer(strategy=strategy, missing_values=np.nan).fit_transform(df2)

    # print('after imputer')
    # print(df2.isnull().sum().loc[lambda x : x>0].sort_values(ascending=False))

    df_majority = df[df['loss_geq_zero'] == 0]
    df_minority = df[df['loss_geq_zero'] == 1]
    df_majority2 = df2[df2['loss_geq_zero'] == 0]
    df_minority2 = df2[df2['loss_geq_zero'] == 1]

    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority),  # to match majority class
                                     random_state=123)  # reproducible results
    df_minority_upsampled2 = resample(df_minority2,
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority2),  # to match majority class
                                     random_state=123)  # reproducible results

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    df_upsampled2 = pd.concat([df_majority2, df_minority_upsampled2])

    df = df_upsampled
    df2 = df_upsampled2

    X_train, y_train = useful_functions.separate_into_xy_loss_exists(df)
    X_train2, y_train2 = useful_functions.separate_into_xy_loss_exists(df2)
    X_test, y_test = useful_functions.separate_into_xy_loss_exists(df_test)

    sc = StandardScaler()
    sc2 = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_train2 = sc2.fit_transform(X_train2)
    X_test1 = sc.transform(X_test)
    X_test2 = sc2.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced')
    clf2 = RandomForestClassifier(n_estimators=200, class_weight='balanced')
    clf.fit(X_train, y_train)
    clf2.fit(X_train2, y_train2)
    pred_clf1 = clf.predict(X_test1)
    pred_clf2 = clf.predict(X_test2)

    print("comparison statistic for: " + strategy + " is: " + str(evaluation_functions.accuracy(pred_clf2, pred_clf1)))


def randomly_remove_data(df, columns, replace_prob):
    # not an inplace function
    df2 = df.copy()

    # arg max the cols with na in (probs a faster way of doing this somewhere)
    cols = list(df.columns)
    cols_needed = []
    for i in columns:
        cols_needed.append(cols.index(i))

    # iterate over rows in columns specified and randomly set cells to nan with probability replace_prob
    ix = [(row, col) for row in range(df2.shape[0]) for col in cols_needed]
    for row, col in random.sample(ix, int(round(replace_prob * len(ix)))):
        df2.iat[row, col] = np.nan
    return df2


