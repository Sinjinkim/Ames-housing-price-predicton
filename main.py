import pandas as pd 
import joblib
from pathlib import Path 

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


from src.features import build_preprocessor 
from src.evaluate import evaluate_model
from src.train import build_model


def main():

    data = pd.read_csv("data/raw/train.csv")

    X = data.drop(columns = ["SalePrice", "Id"])
    y = data["SalePrice"]

    final_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(X)),
        ("model", build_model(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            reg_lambda=1
        ))
    ])

    # 3. Train on all labeled observations
    final_pipeline.fit(X, y)

    # 4. Save trained model and preprocessing
    Path("models").mkdir(exist_ok=True)

    model_path = "models/xgboost_model.joblib"

    joblib.dump(final_pipeline, model_path)

    print("Final XGBoost model trained successfully.")
    print(f"Training observations: {len(X)}")
    print(f"Model saved to: {model_path}")




if __name__ == "__main__":
    main()


     

   