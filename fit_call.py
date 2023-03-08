#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from typing import List, Tuple
import time 
import logging
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from sklearn.metrics import mean_squared_error, mean_squared_log_error

from feature_engineering import feat_eng
from feature_engineering_fcn import basic
from prep_data import prep_test, prep_sum_final, df_fresh, prep_sum
from pred_util import set_up_train_test, save_prediction, add_prediction_to_train, add_new_cols
from train_model import make_model, make_final_model


def train_func(train:pd.DataFrame, lags:int, final_flag:bool, model_type:str='rf_xgb', params:dict={}):
    """ - Function to run train/test, and final train.
        - df_fresh (from prep_data) mods cols and sets up cols_dict with col names and types
        - feat_eng (from feature_engineering) is a helper function to call all the feature engineering functions
        - prep_sum or prep_sum_final (from prep_data) does one hot encoding, sets up training and test set and
            separate features and label.
        - make_model, make_final_model (from train_model) trains a model
        
        """
    logging.info(f'running model type: {model_type}')
    logging.info(f'with params: {params}')
    
    print('***** Set up *****')
    df_full = train.copy(deep=True)
    df_full, cols_dict = df_fresh(df_full)
    df_full, cols_dict = feat_eng(df_full, cols_dict, lags, train_flag=1)
    logging.info(f'Number of features: {len(df_full.columns)}')
    logging.info(f'Feature names: {df_full.columns.values}')
    
    if final_flag == 0:
        x_train, y_train, x_test, y_test, ohe = prep_sum(df_full, cols_dict, geo_flag=1)        
        print('***** Training *****')
        trained_model = make_model(x_train, y_train, x_test, y_test,type = model_type, params=params) #, params={'subsample': 0.6, 'n_estimators': 1000, 'max_depth': 10, 'learning_rate': 0.2, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7})
    else: 
        x_train, y_train, ohe = prep_sum_final(df_full, cols_dict, geo_flag=1)
        print('***** Traing *****')
        trained_model = make_final_model(x_train, y_train,type = model_type, params=params) 

    print('***** Done *****')
    
    return trained_model, ohe


def random_opt_func(train:pd.DataFrame, lags:int, n:int=10):
    # # # # # Hyper param tuning XGB model
    df_full = train.copy(deep=True) # train[train.week < 125].copy(deep=True).sort_values(['meal_id','center_id','week']).reset_index(drop=True) # 
    df_full, cols_dict = df_fresh(df_full)
    df_full, cols_dict = feat_eng(df_full, cols_dict, lags, train_flag=1)
    x_train, y_train, x_val, y_val, ohe = prep_sum(df_full, cols_dict, geo_flag=1)
                    
    params = { 'max_depth': [3, 4, 5], #, 7, 8],
            'learning_rate': [0.001, 0.01, 0.1],
            'subsample': np.arange(0.5, 1.0, 0.1),
            'colsample_bytree': np.arange(0.4, 1.0, 0.1),
            'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
            'n_estimators': [100, 200, 400, 500]} #[20, 50, 100, 150, 200]}

    param_list = list(ParameterSampler(params, n_iter=n))

    param_list_of_dicts = [dict((k, round(v, 2)) for (k, v) in d.items()) for d in param_list]

    eval_set = [(x_train,y_train), (x_val, y_val)]
    eval_metric = ["rmse"]
    metric_lst = []
    for i in range(0,n):
        print('checking model nr:',i)
        xgb_reg = xgb.XGBRegressor(n_jobs=-1)
        xgb_reg.set_params(**param_list_of_dicts[i])
        xgb_rand = xgb_reg.fit(x_train, y_train , eval_metric = eval_metric,
                            eval_set=eval_set,
                            #early_stopping_rounds=5,
                            verbose=False)
        results = xgb_rand.evals_result()    
        metric_lst.append((i,results['validation_1']['rmse'][-1]))

    sorted_metric = sorted(metric_lst,key=lambda x: x[1])

    return param_list, sorted_metric


def prediction_func(test:pd.DataFrame, train:pd.DataFrame, lags:int, ohe_transformer, trained_model, pred_flag:bool):
    """ Prediction function calculates predictions for a test set. 
        It predicts one week at a time and then recalculates features dependent on num_orders and makes a new prediction
        """
    start_time = time.time()
    # Choose dataset either for train/test or final prediction and submission (from pred_util).
    df_train, df_test, df_pred_store, cols_dict = set_up_train_test(train, test, pred_flag)

    # Basic feature engineering. 
    # Sets up features that are known for the whole dataset and can be 'leading' (from feature_engineering_fcn)
    df_test, cols_dict = basic(df_test, cols_dict, train_flag=0)
    df_train, _ = basic(df_train, cols_dict, train_flag=1)

    # Some meal_id/center_id are only in the test set. These are "added back" to the training set
    if pred_flag == True:
        df_train = add_new_cols(df_test, df_train)

    print('****** PREDICTION ******')
    # Get some constants and vars
    weekn = df_test.week.min()
    testn = df_test.week.max() + 1
    logging.info(f"Prediction range w: {weekn} to {testn-1}")
    org_cols = df_train.columns.to_list()
    df_train_lag = df_train.copy(deep=True)
    
    # Loop through the weeks of the test set and predict on week at a time.
    for i in range(weekn,testn):
        # Get week to predict on and add it to the test df
        df_week_pred = df_test[df_test.week == i].reset_index(drop=True)
        df_train_lag = (pd.concat([df_train_lag, df_week_pred], axis=0)
                                            .sort_values(['meal_id', 'center_id', 'week']).reset_index(drop=True))
        
        # Add engineered features
        df_train_lag, cols_dict = feat_eng(df_train_lag, cols_dict, lags, train_flag=0)
        
        # Get data for week 146 for prediction. Remove num_orders and rename df_TEST_lag
        df_test_lag = df_train_lag[df_train_lag.week == i].drop(columns=['num_orders']).reset_index(drop=True)
          
        # Use ohe pipeline to set up df for prediction
        df_test_lag_ohe, _ =  prep_test(df_test_lag, cols_dict, geo_flag=1, ohe_transformer=ohe_transformer)
        cols_dict['cat_cols'] += ['city_region']

        
        # Check lenght of one week prediction df (3600)
        # If it is wrong, print out some diagnostics
        len_test_lag = len(df_test_lag)
        len_ohe = len(df_test_lag_ohe)
        if len_test_lag != len_ohe:
            print('length df_ohe_test_lag: ', len_ohe)
            print('length df_test_lag', len_test_lag)
            a = df_test_lag.isnull().sum()
            print(a[a.values != 0])
            #display(df_test_lag.iloc[3233:3239, :])
            #df_test_lag = df_test_lag.sort_values['meal_id', 'center_id', 'week'].reset_index(drop=True)
            #display(df_test_lag[df_test_lag.num_orders_lag_9w.isnull()])
            #152	2956
            df2 = df_train_lag[df_train_lag.num_orders_lag_9w.isnull()]
            display(df2[(df2.meal_id == 2956) & (df2.center_id == 152)])	
            break
        
        # Remove calculate feature data for train_lag
        df_train_lag = df_train_lag[org_cols] 
        
        ###################
        # Predict for week i
        y_pred = trained_model.predict(df_test_lag_ohe)
        if i in [146, 150]: 
            print(time.time() - start_time, y_pred[:5])
        
        # Make df for predict with meal, center and week for sorting predictions
        # # Save prediction on real scale to submission df    
        df_pred_store = save_prediction(df_test_lag, df_pred_store, y_pred, i)
        
        # Get predict df with only pred data
        # # Get train w 146 and add in predictions
        df_train_lag = add_prediction_to_train(df_test_lag, df_train_lag, y_pred, i)
        # Set not served meals to 0 num_orders, not prediction
        df_train_lag.loc[df_train_lag.id.isnull(),['num_orders']] = 0
         
        #if i == 146: break   
    
    # Information print out
    if pred_flag == False:
        # Calculate rmsle for test
        print_rmsle(df_pred_store, df_test)
        
    # Time from start
    prediction_time = (time.time() - start_time)/60
    print(prediction_time, 'min')  
    logging.info(f'Total prediction time {prediction_time}')
    return df_pred_store


def print_rmsle(df_pred_store:pd.DataFrame, df_test_org:pd.DataFrame):
    """ Calculate rmsle for train/test data """
    df_test_pred = df_test_org.loc[:,['meal_id', 'center_id', 'week', 'num_orders']]
    df_pred = df_pred_store.merge(right=df_test_pred, how='left', on=['meal_id', 'center_id', 'week'])
    df_pred = df_pred.dropna()
    
    rmsle = 100*np.sqrt(mean_squared_log_error(df_pred.num_orders, df_pred.pred))
    print(f'rf_rmsle: {rmsle}')
    logging.info(f'validation rmsle: {rmsle}')
    return



def plot_sample(df:pd.DataFrame, train:pd.DataFrame):
    """ Plot predicted data for sample meal_id and center_id """
    val_data = train[(train.week > 135) & (train.meal_id == 1062) & (train.center_id == 10)]
    pred_data = df[(df.meal_id == 1062) & (df.center_id == 10)]

    plt.figure()
    plt.plot(val_data.week, val_data.num_orders, label='val_data')
    plt.plot(pred_data.week, pred_data.pred, label='prediction')
    plt.legend()
    plt.show()


def submit_save(df:pd.DataFrame):
    """ Save prediction for submission file"""
    df_sub = pd.read_csv('data/raw/sample_submission.csv').sort_values(['id']).reset_index(drop=True)

    df_pred = df[['id', 'pred']].astype(int).sort_values(['id']).reset_index(drop=True)
    df_sub['num_orders'] = df_pred.pred
    display(df_sub.head())

    df_sub.to_csv('submission.csv', index=False)
    logging.info('submission file saved')
    return


# %% Use to run by calling file

if __name__ == "__main__":
    # # Function for reproducible results
    def set_seed(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
    set_seed(1)

    train = pd.read_parquet('data/train/train_w_zeros.parquet')
    #train = train.dropna()
    print(train.shape)
    test = pd.read_parquet('data/test_w_zeros2.parquet')
    #test = test.dropna()
    print(test.shape)
    test = test.sort_values(['meal_id', 'center_id', 'week']).reset_index(drop=True)

    lags = 12

    # train/test
    #trained_model, ohe = train_func(train, lags, final_flag=0, model_type='rf_xgb', params={'n_estimators': 18})
    # make final model
    trained_model, ohe = train_func(train, lags, final_flag=1, model_type='rf_xgb', params={})
    # predict on sample
    df_predictions = prediction_func(test, train, lags, ohe, trained_model, pred_flag=1)
    df_predictions = df_predictions.dropna()
    display(df_predictions.head(20))
    
    # plot test meal/center
    plot_sample(df_predictions, train)
    # make submission file    
    submit_save(df_predictions)

