import pandas as pd
import numpy as np
from prep_data import df_fresh
from typing import Tuple, List


def set_up_train_test(train:pd.DataFrame, test:pd.DataFrame, pred:bool=0) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """ Function to set up test dataset for prediction
        For training we choose the last 10 weeks of the train file. For submission we use the AV test file
        """
    if pred == False: # Set up data for train/test/val testing
        # Get the last 10 weeks of the train file
        df_train = train[train.week < 135].copy(deep=True).sort_values(['meal_id','center_id','week']).reset_index(drop=True)
        df_test = train[train.week >= 135].copy(deep=True).sort_values(['meal_id','center_id','week']).reset_index(drop=True)
        # Set up df for saving predictions
        df_pred_store = train[train.week >= 135].copy(deep=True).sort_values(['meal_id','center_id','week']).reset_index(drop=True)
        df_pred_store = df_pred_store.drop(columns=['num_orders'])
        df_pred_store['pred'] = np.nan
        # Do initial data processing 
        df_train, _ = df_fresh(df_train)
        df_test, cols_dict = df_fresh(df_test)
        
    elif pred == True: 
        # Set up data for predicting the test set.
        # Make new copies and sort. use df_fresh to set up cols_dict and make city-region
        df_train = train.copy(deep=True).sort_values(['meal_id','center_id','week']).reset_index(drop=True)
        df_test = test.copy(deep=True).sort_values(['meal_id','center_id','week']).reset_index(drop=True)
        df_pred_store = test.copy(deep=True).sort_values(['meal_id','center_id','week']).reset_index(drop=True)
        df_pred_store['pred'] = np.nan
        
        df_train, _ = df_fresh(df_train)
        df_test, cols_dict = df_fresh(df_test)
        
    return df_train, df_test, df_pred_store, cols_dict


def save_prediction(df_week_pred:pd.DataFrame, df_pred_store:pd.DataFrame, y_pred:np.ndarray, week_nr:int) -> pd.DataFrame:
    """ Add week prediction as "pred" to the original test df for submission"""
    # Sort prediction df, to align with df for saving, convert prediction from log
    df_temp = df_week_pred[['meal_id', 'center_id', 'week']].copy(deep=True)
    df_temp['pred'] = np.expm1(y_pred)

    # Get this week feature from dataframe and remove nan-pred col
    df_pred_store_week = df_pred_store[df_pred_store.week == week_nr].drop(columns=['pred']).reset_index(drop=True)        
    # Remove this week from df_pred_store so it can be added with the new predition
    df_pred_store = df_pred_store[df_pred_store.week != week_nr].reset_index(drop=True)
    # Add prediction to this week 
    df_pred_store_week = df_pred_store_week.merge(right = df_temp, on = ['meal_id', 'center_id', 'week'], how = 'left')
    # Add prediction back to df_pred_store
    df_pred_store = pd.concat([df_pred_store, df_pred_store_week], axis=0).sort_values(['meal_id', 'center_id', 'week']).reset_index(drop=True)

    return df_pred_store

def add_prediction_to_train(df_week_pred:pd.DataFrame, df_train_lag:pd.DataFrame, y_pred:np.ndarray, week_nr:int) -> pd.DataFrame:
    """ Add prediction to train_lag df to be used for calculating feature columns with this weeks predictions"""
    # Set up prediction df for copying the pred column to train_lag
    df_temp = df_week_pred[['meal_id', 'center_id', 'week']].copy(deep=True)
    df_temp['pred'] = y_pred
    df_temp.rename(columns={'pred':'num_orders'}, inplace=True)
    df_temp = df_temp.drop(columns=['week'])
    
    # Get this week, remove num_orders
    df_train_lag_1w = df_train_lag[df_train_lag.week == week_nr].drop(columns='num_orders').reset_index(drop=True)
    # Add predictions
    df_train_lag_1w = df_train_lag_1w.merge(right=df_temp, how='left', on=['meal_id', 'center_id']).reset_index(drop=True)
    # Put this week back
    df_train_lag = pd.concat([df_train_lag[df_train_lag.week != week_nr].reset_index(drop=True), df_train_lag_1w], axis = 0).sort_values(['meal_id', 'center_id', 'week']).reset_index(drop=True)
    return df_train_lag

def add_new_cols(df_test, df_train):
    """ Three meal/center_id combinations appear in the test file but not the train file.
        This function adds dummy cols to the train file for these"""
    # Make list with new meal/centers and week cols
    new_cols = [[1571, 73], [2104, 92], [2956, 73]]
    new_list = []
    for item in new_cols:
        new_list_temp = [item + [i] for i in range(1,146)]  
        new_list.append(new_list_temp)

    # Put in df
    new_arr = np.array(new_list).reshape(-1,3)
    main_cols = ['meal_id', 'center_id', 'week']
    df_new = pd.DataFrame(data=new_arr, columns=main_cols)
    df_new2 = df_new.merge(right=df_test[df_test.week == 146], how='left', on=['meal_id', 'center_id'], suffixes = (None, '_y'))
    df_new2.drop(columns=['week_y', 'dscnt_pct'], inplace=True)
    df_new2['num_orders'] = 0
    # Add in new cols to df_train from test set (1571, 73), (2104, 92), (2956, 73)
    df_train = pd.concat([df_train, df_new2], axis=0).sort_values(['meal_id', 'center_id', 'week']).reset_index(drop=True)

    return df_train
