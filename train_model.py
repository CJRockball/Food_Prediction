import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import time 

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



# Modelling, fit and eval  
def make_model(x_train:pd.DataFrame, y_train:pd.DataFrame, x_test:pd.DataFrame, y_test:pd.DataFrame, \
                type:bool='rf_sklearn', params:dict={}):
    """ Train predictive model, print some metrics and plot feature importance.
        Use this one for train/test. This model contains test data for xgboost.
        Choose model type with type variable"""
    # Using the unscaled features and labels here as RF takes care of it
    # Use random forrest from sklearn to make model and predict on test set
    if type == 'rf_sklearn': 
        rf = RandomForestRegressor(n_jobs=-1)
        rf.fit(x_train, y_train)
        rf_train_score = rf.score(x_train, y_train)
        rf_test_score = rf.score(x_test, y_test)
        y_test_pred = rf.predict(x_test)

    # Creating an XGBoost model.
    elif type == 'rf_xgb':
        train_time_start = time.time()
        
        rf = xgb.XGBRegressor(n_jobs=-1) 
        rf.set_params(**params)
        rf.fit(x_train, y_train, eval_metric="rmse",  
               eval_set=[(x_train, y_train), (x_test, y_test)], verbose=True)
        rf_train_score = rf.score(x_train, y_train)
        rf_test_score = rf.score(x_test, y_test)
        y_test_pred = rf.predict(x_test)
        finish_train_time = round(time.time() - train_time_start,2)
        logging.info(f'training time: {finish_train_time}s')
        
        # retrieve performance metrics
        results = rf.evals_result()
        logging.info(f"training rmse: {results['validation_0']['rmse'][-1]}")
        logging.info(f"test rmse: {results['validation_1']['rmse'][-1]}")
        # plot learning curves
        plt.figure()
        plt.plot(results['validation_0']['rmse'], label='train')
        plt.plot(results['validation_1']['rmse'], label='test')
        plt.show() 
    
    # Calculate and print metrics
    rf_mse = mean_squared_error(y_test, y_test_pred)
    rf_rmsle = 100*np.sqrt(mean_squared_log_error(y_test, y_test_pred))

    print(f"The mean squared error with {type} on the validation set is: {rf_mse}")
    print(f'RMSLE: {rf_rmsle}')
    print(f"The R squared of {type} on the train set is:  {rf_train_score}")
    print(f"The R squared of {type} on the validation set is:  {rf_test_score}")
    logging.info(f"Training metrics, R2 on test {rf_test_score}, rmsle on test: {rf_rmsle}")
    
    # Feature importance
    feat_importances = pd.Series(rf.feature_importances_, index=x_train.columns)
    ax = feat_importances.sort_values(ascending= True)[-20:].plot(kind='barh', color = 'blue', figsize = (15,15), fontsize = 15)
    ax.set_xlabel("Relative Importance", fontsize = 15)
    ax.set_ylabel("Features", fontsize = 15)
    logging.info(f'feature importance: {feat_importances.sort_values(ascending=True)[-10:].to_dict()}')
    
    return rf

# Modelling, fit and eval  
def make_final_model(x_train:pd.DataFrame, y_train:pd.DataFrame, type:str='rf_sklearn', params:dict={}):
    """  Train predictive model, print some metrics and plot feature importance.
        Use this one for final model. This model does not have test data for xgboost.
        Choose model type with type variable """
    # Using the unscaled features and labels here as RF takes care of it
    if type == 'rf_sklearn': 
        rf = RandomForestRegressor(n_jobs=-1)
        rf.fit(x_train, y_train)

    elif type == 'rf_xgb':
        rf = xgb.XGBRegressor(n_jobs=-1) 
        rf.set_params(**params)
        rf.fit(x_train, y_train, verbose=True)
           
        # Feature importance
        feat_importances = pd.Series(rf.feature_importances_, index=x_train.columns)
        ax = feat_importances.sort_values(ascending= True)[-20:].plot(kind='barh', color = 'blue', figsize = (12,12), fontsize = 13)
        ax.set_xlabel("Relative Importance", fontsize = 13)
        ax.set_ylabel("Features", fontsize = 13)
    
    elif type == 'booster':
        b_params = {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
           }
        rf = GradientBoostingRegressor(**b_params)
        rf.fit(x_train, y_train)
        
    return rf
