import xgboost as xgb

def get_xgboost_model_v01(num_class: int):
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

def get_xgboost_model_v02(num_class: int):
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=6,
        random_state=42
    )