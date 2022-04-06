# xgboost parameter tuning
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from useful_code import useful_functions
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

scores = ["precision", "accuracy"]

parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

tuned_parameters = [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
]

lgb_tuning = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15,20,25],
    'num_leaves': [50, 100, 200],
}

lgb_tuning2 = {
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'min_split_gain': [0.3, 0.4],
    'subsample': [0.7, 0.8, 0.9],
}


@ignore_warnings(category=ConvergenceWarning)
def test_9():
    mlp = MLPClassifier()

    df = useful_functions.import_all_data()
    df = useful_functions.clean_dataset(df)

    df = useful_functions.create_loss_col(df)
    df = df.head(4000)

    df, df_test = useful_functions.split(df, round(0.8 * len(df)))

    df = useful_functions.upsample_minority(df)

    print(df['loss_geq_zero'].value_counts())

    X_train, y_train = useful_functions.separate_into_xy_loss_exists(df)
    X_test, y_test = useful_functions.separate_into_xy_loss_exists(df_test)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # for score in scores:
    # print("optimising for: " + score)
    # for 3
    for scoring in ['recall', 'precision', 'accuracy']:
        # tuning = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, scoring=scoring)
        # tuning = GridSearchCV(SVC(), tuned_parameters, scoring=scoring)
        # tuning = GridSearchCV(lgb.LGBMClassifier(subsample_freq=20), lgb_tuning, scoring=scoring)
        tuning = GridSearchCV(lgb.LGBMClassifier(subsample_freq=20, n_estimators=400, num_leaves=100, max_depth=20,
                                                 colsample_bytree=0.7), lgb_tuning2, scoring=scoring)

        tuning.fit(X_train, y_train)
        print("best_params_: ")
        print(tuning.best_params_)
        print("best_score_: ")
        print(tuning.best_score_)
        print()
