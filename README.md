# Ames Housing Price Prediction 

## Machine Learning Regression Project | Python, Scikit-learn, XGBoost

## Project overview

This project develops an end to end machine learning pipeline to predict residential property sale prices using the Ames Housing dataset.


The objective was to investigate the relationship between housing characteristics and sale prices, build an accurate regression model, and evaluate its performance in predicting sale prices to unseen properites. 

The project includes exploratory data analysis (EDA), feature preprocessing, model training, cross validation, hyperparameter tuning, and residual analysis. 

## Dataset 

Source: Kaggle-House Prices: Advandced Regression Techniques 

The training dataset contains 1,460 residential properites from Ames, Iowa, with 79 explanatory variables after excluding the property identifier. 

The target variable is SalePrice, representing the final sale price of each property.

Features include:
- Property size and living area
- Overall material and finish quality 
- Neighborhood and location 
- Garage capacity and characteristics 
- Year built and remodeling history 
- Basement and property amenities 



## Technologies Used

- **Python:** Primary programming language for data analysis, model development, and prediction.
- **Pandas:** Data cleaning, manipulation, aggregation, and exploratory data analysis.
- **NumPy:** Numerical operations and array processing.
- **Matplotlib:** Static visualizations for exploratory analysis, residual analysis, and feature importance.
- **Plotly (Express & Graph Objects):** Interactive visualizations for exploring relationships between housing characteristics and sale prices.
- **SciPy:** Statistical analysis and calculations.
- **Scikit-learn:** Data preprocessing, missing-value imputation, categorical encoding, machine learning pipelines, train-test splitting, cross-validation, hyperparameter tuning, and model evaluation.
- **XGBoost:** Gradient-boosted regression model for predicting residential property prices.
- **Joblib:** Serialization and loading of the trained machine learning pipeline.
- **Jupyter Notebook / IPython:** Interactive experimentation, data visualization, and documentation of exploratory analysis and model evaluation.
- **VS Code:** Development environment for organizing and executing the project.




## Methodology 

## 1. Exploratory Data Analysis 

Investigated relationships between property characteristics and sale prices, including overall qualtiy, building type, garage capacity, remodeling history, and missing value patterns.

## 2. Data Preprocessing 

Developed a Scikit-learn pipeline that:
- imputes missing numerical values
- Handles missing values representing absent property features
- Encodes categorical variables using OneHotEncoder
- Applies consistent transformations during training and prediction 


## 3. Model Development 

Trained XGBoost regression model and compared its performance against a mean-based baseline model using RMSE and MAE. 

## 4. Cross Validation and Hyperparameter tuning

Implemented five fold cross validation to evaluate model generalization. 

Used GridSeachCV to evaluate 16 hyperparameter combinations across five folds, totaling 80 fits. 

The original XGBoost configuration was selected as the best performing configuration within the tested search space.


## Model Performance

Training RMSE : $6,673.84
Validation RMSE: $25,440.83
Validation MAE: $16,282.43 
Mean 5 fold CV RMSE: $28,638.80


The results reflect the model configuration evaluated after excluding the Id feature. Cross validation RMSE is an average across folds, while validation RMSE is calculated on the held out validation split.

## Residual Analysis 

Residual analysis revealed that the model generally produced larger absolute prediction errors for higher priced properties. 

For properties valued above $400,000, the model had a mean absolute error of approximately $64,130. All seven properties in this validation price group were underpredicted. 

This suggests a potential limitation in predicting high value properties, although given the small number of observations, further analysis is required to make a definite conclusion. 


# Project Structure

ames-housing-price-prediction/
├── data/
│   └── raw/
├── models/
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_model_evaluation.ipynb
├── reports/
│   └── predictions.csv
├── src/
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── main.py
├── requirements.txt
└── README.md



# How to Run

## 1. Clone the repository

git clone https://github.com/YOUR_USERNAME/ames-housing-price-prediction.git
cd ames-housing-price-prediction

## 2. Instal dependencies 

pip install -r requirements.txt

## 3. Download the dataset 

Download train.csv and test.csv from the Kaggle dataset link above and place them in data/raw/, unless the repository already includes these files.

## 4. Train the final model 

python main.py

This trains the selected XGBoost pipeline using all available labeled observations and saves it to:
models/xgboost_model.joblib

## 5. Generate predictions

python -m src.predict

This loads the saved pipeline, predicts sale prices for the unlabeled test dataset, and saves the results to:

reports/predictions.csv




## Limitations and Future Improvements

Additional feature engineering may improve predictive accuracy.
More extensive hyperparameter searches could identify better configurations.
The model's larger errors on expensive properties warrant further investigation.
Performance on Ames housing data may not generalize to different cities or housing markets.

# Conclusion

This project demonstrates an end-to-end machine learning workflow, from exploratory analysis and preprocessing to model evaluation, selection, serialization, and prediction.

The final XGBoost model achieved a mean five-fold cross-validation RMSE of approximately $28,639 and was used to generate predictions for unseen housing records.