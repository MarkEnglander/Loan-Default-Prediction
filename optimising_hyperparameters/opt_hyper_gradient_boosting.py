# This test is for optimising the parameters for x
# The code is essentially modified from the documentation:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

from useful_code import useful_functions
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample

scores = ["recall", "precision"]

# for svm if i ever do it...
tuned_parameters = [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
]

p_test3 = {'learning_rate': [0.15, 0.1, 0.05, 0.01, 0.001],
           'n_estimators': [100, 500, 1000, 1500]}

p_test2 = {'max_depth': [2, 3, 4, 5, 6, 7]}


def test_6():
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
            estimator=GradientBoostingClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,
                                                 max_features='sqrt', random_state=10),
            param_grid=p_test3, scoring=score, n_jobs=4, cv=5)

        # for 2
        tuning = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.15, n_estimators=500,
        min_samples_split=2, min_samples_leaf=1, subsample=1, max_features='sqrt', random_state=10),
        param_grid=p_test2, scoring=score, n_jobs=4, cv=5)
        tuning.fit(X_train, y_train)
        print("best_params_: ")
        print(tuning.best_params_)
        print("best_score_: ")
        print(tuning.best_score_)
        print()
