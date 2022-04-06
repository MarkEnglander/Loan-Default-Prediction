# xgboost parameter tuning

from useful_code import useful_functions
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample


scores = ["recall", "precision", "accuracy"]

param_test1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}

param_test2 = {
    'max_depth': [6, 7, 8],
    'min_child_weight': [1, 2, 3]
}

param_test3 = {
    'gamma': [i / 10.0 for i in range(0, 5)]
}

param_test4 = {
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)]
}

param_test5 = {
    'subsample': [i / 100.0 for i in range(55, 70, 5)],
    'colsample_bytree': [i / 100.0 for i in range(75, 90, 5)]
}

param_test6 = {
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}

param_test7 = {
    'reg_alpha': [0, 1e-5, 1e-4, 1e-3]
}


def test_8():
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

    # for score in scores:
    # print("optimising for: " + score)
    # for 3

    tuning = GridSearchCV(estimator=XGBClassifier(max_depth=7, min_child_weight=1, learning_rate=0.1, n_estimators=140,
                                                  gamma=0.0,
                                                  objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                  seed=27, colsample_bytree=0.6, subsample=0.6),
                          param_grid=param_test7, scoring='recall', n_jobs=2, cv=5)

    tuning.fit(X_train, y_train)
    print("best_params_: ")
    print(tuning.best_params_)
    print("best_score_: ")
    print(tuning.best_score_)
    print()
