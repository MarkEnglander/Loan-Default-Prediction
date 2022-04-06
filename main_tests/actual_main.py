import main_upsample_run_1
import lightgbm as lgb

clf = lgb.LGBMClassifier(subsample_freq=20, n_estimators=400, num_leaves=100, max_depth=20,
                         colsample_bytree=0.7, min_split_gain=0.3, reg_alpha=1.3, reg_lambda=1.3, subsample=0.8)

main_upsample_run_1.run_classifier(clf)