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

classifiers = [svm.SVC(),
               RandomForestClassifier(n_estimators=200),]
               # MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=300),
               # AdaBoostClassifier(n_estimators=100),
               # GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)]


def run_all_test_1():
    for i in classifiers:
        test_1(i)


def test_1(clf):
    df = useful_functions.import_all_data()
    df = useful_functions.clean_dataset(df)
    X, y = useful_functions.separate_into_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf.fit(X_train, y_train)
    pred_clf = clf.predict(X_test)

    print('Classifier name: ' + type(clf).__name__)
    print(classification_report(y_test, pred_clf))
    print(confusion_matrix(y_test, pred_clf))
