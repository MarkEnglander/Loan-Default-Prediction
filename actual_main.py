import pandas as pd
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import main_upsample_run_1
from sklearn.ensemble import GradientBoostingClassifier
import main_downsample_run_1

clf1 = GradientBoostingClassifier(learning_rate=0.15, n_estimators=500, max_depth=7)
clf2 = AdaBoostClassifier(learning_rate=0.5, n_estimators=500)
clf3 = AdaBoostClassifier(learning_rate=0.15, n_estimators=500)
clf4 = RandomForestClassifier(n_estimators=400)
clf5 = RandomForestClassifier(n_estimators=100)
clf6 = XGBClassifier(max_depth=7, min_child_weight=1, gamma=0.1, colsample_bytree=0.8, subsample=0.6, reg_alpha=0,
                     n_estimators=5000, learning_rate=0.01)


main_upsample_run_1.run_classifier(clf3)
main_upsample_run_1.run_classifier(clf1)
main_upsample_run_1.run_classifier(clf6)
