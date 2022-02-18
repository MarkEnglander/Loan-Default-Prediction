# when wanting to just use a classifier quickly keeping other things more or less 'normal' just run this

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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample

classifiers = [svm.SVC(class_weight='balanced'),
               RandomForestClassifier(n_estimators=200, class_weight='balanced'),
               MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=300),
               AdaBoostClassifier(n_estimators=100),
               GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)]


def run_all_test_4():
    for i in classifiers:
        default_run_classifier(i)


def default_run_classifier(clf, data_points=None):
    df = useful_functions.import_all_data()
    df = useful_functions.clean_dataset(df)
    if data_points is not None:
        # rough estimate since points get removed then added etc etc
        df = df.head(data_points * int(3 / 4))

    df = useful_functions.create_loss_col(df)

    df, df_test = useful_functions.split(df, round(0.8 * len(df)))

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

    print('Classifier name: ' + type(clf).__name__)
    print(classification_report(y_test, pred_clf))
    print(confusion_matrix(y_test, pred_clf))
