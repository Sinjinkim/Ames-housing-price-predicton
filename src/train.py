from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

def build_model(**params):

    default_params = {
        "n_estimators" : 500,
        "learning_rate" : 0.1,
        "max_depth" : 3,
        "subsample" : 1.0,
        "colsample_bytree" : 0.8,
        "random_state" : 32
    }

    default_params.update(params)
    
    return XGBRegressor(**default_params)

def tune_model(pipeline, train_X, train_y, kf):
    param_grid = {
        "model__n_estimators": [200,500],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2,3],
        "model__reg_lambda": [1,5]
    }

    grid_search = GridSearchCV(
        estimator = pipeline,
        param_grid = param_grid,
        scoring = 'neg_root_mean_squared_error',
        cv = kf,
        n_jobs = 2,
        verbose = 1
    )

    grid_search.fit(train_X, train_y)
    return grid_search