# when wanting to just use a classifier quickly keeping other things more or less 'normal' just run this

from useful_code import useful_functions
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import useful_code.useful_functions
from useful_code import PandasImputer


def run_classifier(clf, data_points=None):
    df = useful_functions.import_all_data()
    df = useful_functions.clean_dataset(df, delete_missing_data=False)

    if data_points is not None:
        # rough estimate since points get removed then added etc etc
        df = df.head(data_points * int(3 / 4))

    df = useful_functions.create_loss_col(df)

    df, df_test = useful_functions.split(df, round(0.8 * len(df)))
    imp = PandasImputer.PandasImputer(strategy='median', missing_values=np.nan)
    df = imp.fit_transform(df)
    df_test = imp.transform(df_test)

    df_majority = df[df['loss_geq_zero'] == 0]
    df_minority = df[df['loss_geq_zero'] == 1]

    df_majority_downsampled = resample(df_majority,
                                       replace=True,  # sample with replacement
                                       n_samples=len(df_minority),
                                       random_state=123)  # reproducible results

    df_downsampled = pd.concat([df_minority, df_majority_downsampled])
    df = df_downsampled

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
