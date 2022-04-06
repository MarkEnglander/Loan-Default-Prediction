from useful_code import useful_functions
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample

classifiers = [svm.SVC(class_weight='balanced'),
               RandomForestClassifier(n_estimators=200, class_weight='balanced'),
               MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=300),
               AdaBoostClassifier(n_estimators=100),
               GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)]


def run_all_test_2():
    for i in classifiers:
        test_2(i)


def test_2(clf):
    df = useful_functions.import_all_data()
    df = useful_functions.clean_dataset(df)

    df = useful_functions.create_loss_col(df)

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

    X, y = useful_functions.separate_into_xy_loss_exists(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf.fit(X_train, y_train)
    pred_clf = clf.predict(X_test)

    print('Classifier name: ' + type(clf).__name__)
    print(classification_report(y_test, pred_clf))
    print(confusion_matrix(y_test, pred_clf))
