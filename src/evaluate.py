from sklearn.metrics import(
    root_mean_squared_error, 
    mean_absolute_error
)

import pandas as pd 


def evaluate_model(model, X, y):
    predictions = model.predict(X)

    rmse = root_mean_squared_error(y,predictions)
    mae = mean_absolute_error(y, predictions)

    return rmse, mae 

    
def residual_analysis(model, X, y):
    predictions = model.predict(X)

    results = pd.DataFrame({
        "Actual": y.to_numpy(),
        "Predicted": predictions
    })

    results["Residual"] = (results["Actual"] - results["Predicted"])

    results["Absolute_Error"] = results["Residual"].abs()

    return results

