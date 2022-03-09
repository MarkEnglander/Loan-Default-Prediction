# when wanting to just use a classifier quickly keeping other things more or less 'normal' just run this

import useful_functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample
import PandasSimpleImputer


def run_classifier(clf, data_points=None, suppress_prints=False):
    df = useful_functions.import_all_data()
    df = useful_functions.clean_dataset(df, delete_missing_data=False)

    if data_points is not None:
        # rough estimate since points get removed then added etc etc
        df = df.head(data_points * int(3 / 4))

    df = useful_functions.create_loss_col(df)

    df, df_test = useful_functions.split(df, round(0.8 * len(df)))
    imp = PandasSimpleImputer.PandasSimpleImputer(strategy='median', missing_values=np.nan)
    df = imp.fit_transform(df)
    df_test = imp.transform(df_test)

    df_majority = df[df['loss_geq_zero'] == 0]
    df_minority = df[df['loss_geq_zero'] == 1]

    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority),  # to match majority class
                                     random_state=123)  # reproducible results

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    df = df_upsampled

    X_train, y_train = useful_functions.separate_into_xy_loss_exists(df)
    X_test, y_test = useful_functions.separate_into_xy_loss_exists(df_test)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf.fit(X_train, y_train)
    pred_clf = clf.predict(X_test)

    if not suppress_prints:
        print('Classifier name: ' + type(clf).__name__)
        print(classification_report(y_test, pred_clf))
        print(confusion_matrix(y_test, pred_clf))
    return confusion_matrix(y_test, pred_clf)
