# Food Demand Prediction

This is a setup to try out engineering different features. The code is set up to create a model and then predict one week ahead and then use that weekly prediction to predict next week.

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

