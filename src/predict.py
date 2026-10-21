import pandas as pd 
import joblib 
from pathlib import Path

def predict_prices():
    
    model = joblib.load("models/xgboost_model.joblib")

    test_data = pd.read_csv("data/raw/test.csv")


    ids = test_data["Id"]

    X_test = test_data.drop(columns=["Id"])

    predictions = model.predict(X_test)

    results = pd.DataFrame({
        "Id": ids,
        "PredictedSalePrice": predictions
    })

    Path("reports").mkdir(exist_ok=True)

    results.to_csv(
        "reports/predictions.csv",
        index=False
    )

    print(results.head(10))
    print("Predictions saved successfully.")

    return results


if __name__ == "__main__":
    predict_prices()

