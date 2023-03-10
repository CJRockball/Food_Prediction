# Food Demand Prediction

This is a setup to try out engineering different features. The code is set up to create a model and then predict one week ahead and then use that weekly prediction to predict next week. To do this zero order weeks are added back into the dataset, the raw data was preprocessed and saved as parquet to preserve data type.

## Code Structure
The code has the following components
- **main.py**: Make and test models, train/test/validate, final model, AV submission file.
- **fit_call.py**: Functions to run training, testing and prediction.
- **prep_data.py**: Gets train and test df to format for training and testing
- **pred_data.py**: Helper functions for prediction_func in fit_call
- **feature_engineering.py**: Functions for all calls to do feature engineering
- **feature_engineering_fcn.py**: Functions to do feature engineering
- **train_model.py**: Functions to make train/test as well as final model
- **check_feature.py**: Helper functions to check new engineered features before implementation

## How to Use
- Add new features in feature_engineering. Use the functions that are there or add new ones to feature_engineering_fcn. 
- Test the new feature on smaller dfs in check_feature to make sure you get the desired results
- Run the new features in main. The file can be called from the terminal or each section can be run as a Notebook. 
- Main has the following sections:
    - Load data
    - Train/Test
    - Final Model
    - Random Optimization
 
