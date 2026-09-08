from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "test.csv"

def load_training_data():
    return pd.read_csv(TRAIN_PATH)



def load_testing_data():
    return pd.read_csv(TEST_PATH)







