import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.impute import SimpleImputer 

def build_preprocessor(X):

    numerical_columns = X.select_dtypes(include = ["int64", "float64"]).columns 
    categorical_columns = X.select_dtypes(include = ["object", "category"]).columns 

    zero_impute_columns = [
    'MasVnrArea',
    'GarageYrBlt'
    ]  

    other_numerical_columns = [col for col in numerical_columns if col not in zero_impute_columns]

    #Create imputer for imputing with the value of 0 
    zero_imputer = SimpleImputer(strategy = 'constant', fill_value = 0)

    #Create numerical transformer 
    numerical_transformer = SimpleImputer(strategy = 'mean')

    #Create categorical transformer 

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'None')),
        ('encoder', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False ))
    ])

    #Create preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('zero_impute', zero_imputer, zero_impute_columns),
        ('numerical', numerical_transformer, other_numerical_columns),
        ('categorical', categorical_transformer, categorical_columns)
    ])

    return preprocessor 
