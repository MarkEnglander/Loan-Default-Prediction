# This test is for optimising the parameters for y
# The code is essentially modified from the documentation:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

import useful_functions
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample

scores = ["recall", "precision", "accuracy"]

p_test3 = {'learning_rate': [0.5, 0.1, 0.01, 0.001],
           'n_estimators': [100, 200, 500]}

rand_forest_test = {'n_estimators': [100, 200, 300, 400, 500, 1000]}


def test_7():
    df = useful_functions.import_all_data()
    df = useful_functions.clean_dataset(df)

    df = useful_functions.create_loss_col(df)
    df = df.head(10000)

    df, df_test = useful_functions.split(df, round(0.8 * len(df)))

    df_majority = df[df['loss_geq_zero'] == 0]
    df_minority = df[df['loss_geq_zero'] == 1]

    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority),  # to match majority class
                                     random_state=123)  # reproducible results

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    print('check that it is now balanced:')
    print(df_upsampled['loss_geq_zero'].value_counts())

    df = df_upsampled

    X_train, y_train = useful_functions.separate_into_xy_loss_exists(df)
    X_test, y_test = useful_functions.separate_into_xy_loss_exists(df_test)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    for score in scores:
        print("optimising for: " + score)
        # for 3
        tuning = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=rand_forest_test, scoring=score, n_jobs=4, cv=5)

        tuning.fit(X_train, y_train)
        print("best_params_: ")
        print(tuning.best_params_)
        print("best_score_: ")
        print(tuning.best_score_)
        print()
