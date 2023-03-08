import pandas as pd
from typing import Tuple, List
from feature_engineering_fcn import basic, num_lag, two_grp, single_grp, global_stats_lag, weekly_stats, weekly_center_stats


def feat_eng(df:pd.DataFrame, cols_dict:dict, num_of_lags:int, train_flag:bool=1)-> Tuple[pd.DataFrame, dict]:
    """ Helper function calling feature engineering functions"""
    if train_flag == True:
        # For test file we call this function in a different place
        df, cols_dict   = basic(df, cols_dict)
    df                  = num_lag(df, num_of_lags)
    df, cols_dict       = run_global_stats_lag(df, cols_dict)
    df, cols_dict       = run_weekly_stats(df, cols_dict)
    #df, cols_dict       = run_weekly_center_stats(df,cols_dict)
    df, cols_dict       = run_two_grp(df, cols_dict)
    df, cols_dict       = run_single_grp(df, cols_dict)
    
    return df, cols_dict


def run_two_grp(df:pd.DataFrame, cols_dict:dict)-> Tuple[pd.DataFrame, dict]:
    """ Features grouping over two other columns
        For features using num_orders we need to lag 1w not to get data leakage"""
    # Total number per email adds per week per meal_id
    df, cols_dict = two_grp(df, cols_dict, ["meal_id", "week"], "num_orders", "meal_week_count", 'sum', 1, 1)
    # Number of weekly offering in a certain group (i.e. italian from cuisine)
    df, cols_dict = two_grp(df, cols_dict, ["week", "cuisine"], "ordered", "cuisine_week_count", 'sum',0, 0)
    df, cols_dict = two_grp(df, cols_dict, ["week", "category"], "ordered", "category_week_count", 'sum',0, 0)
    # Gives number of dishes offered in a certain region
    df, cols_dict = two_grp(df, cols_dict, ["week", "center_type"], "ordered", "centert_week_count", 'sum',0, 0)
    df, cols_dict = two_grp(df, cols_dict, ["week", "op_area"], "ordered", "op_area_week_count", 'sum',0, 0)
    df, cols_dict = two_grp(df, cols_dict, ["week", "city_region"], "ordered", "cityregion_week_count", 'sum',0, 0)
    
    return df, cols_dict


def run_single_grp(df:pd.DataFrame, cols_dict:dict)-> Tuple[pd.DataFrame, dict]:
    """ Features grouping on one column
        After testing I got better results with these with 1w lag"""
    # Total number of email adds per week 
    df, cols_dict = single_grp(df, cols_dict, 'week', "emailer_for_promotion", "email_week_sum", 'sum', 1, 1)
    # Total number of homepage adds per week 
    df, cols_dict = single_grp(df, cols_dict, 'week', "homepage_featured", "homepage_week_sum", 'sum', 1, 1)
    
    return df, cols_dict



def run_global_stats_lag(df:pd.DataFrame, cols_dict:dict)-> Tuple[pd.DataFrame, dict]:
    """ Stat feature on a certain meal in a certain center. For example min number of orders for meal 1062 in center 10
        This function uses an expanding window, which means it calculates up to the particular week thats being calculated
        Since it uses num_orders we need to lag 1 not to get data leakage"""
    df, cols_dict = global_stats_lag(df, cols_dict, ['meal_id', 'center_id'], 'expanding_min', 'num_orders', 'min', 1, 1)
    df, cols_dict = global_stats_lag(df, cols_dict, ['meal_id', 'center_id'], 'expanding_max', 'num_orders', 'max', 1, 1)
    df, cols_dict = global_stats_lag(df, cols_dict, ['meal_id', 'center_id'], 'expanding_mean', 'num_orders', 'mean', 1, 1)
    df, cols_dict = global_stats_lag(df, cols_dict, ['meal_id', 'center_id'], 'expanding_std', 'num_orders', 'std', 1, 1)

    df, cols_dict = global_stats_lag(df, cols_dict, ['meal_id', 'center_id'], 'expanding_bp', 'base_price', 'mean', 0, 0)
    df, cols_dict = global_stats_lag(df, cols_dict, ['meal_id', 'center_id'], 'expanding_cp', 'checkout_price', 'mean', 0, 0)
    
    return df, cols_dict

def run_weekly_stats(df: pd.DataFrame, cols_dict: dict) -> Tuple[pd.DataFrame, dict]:
    """ The average of the Category (say Beverages) in the corresponding week (week 1, 2, 3, etc)
        Need to lag 1w not to get data leakage because were using num_orders"""
    df, cols_dict = weekly_stats(df, cols_dict, 'category', 'num_orders', 'Avg_orders_category_week', 1, 1)
    df, cols_dict = weekly_stats(df, cols_dict, 'cuisine', 'num_orders', 'Avg_orders_cuisine_week', 1, 1)
    df, cols_dict = weekly_stats(df, cols_dict, 'city_region', 'num_orders', 'Avg_orders_city_region_week', 1, 1)
    df, cols_dict = weekly_stats(df, cols_dict, 'center_type', 'num_orders', 'Avg_orders_centert_week', 1, 1)
    df, cols_dict = weekly_stats(df, cols_dict, 'op_area', 'num_orders', 'Avg_orders_oparea_week', 1, 1)
    
    return df, cols_dict

def run_weekly_center_stats(df: pd.DataFrame, cols_dict: dict) -> Tuple[pd.DataFrame, dict]:
    """ The average of the Category (say Beverages) in the corresponding week (week 1, 2, 3, etc)
        Need to lag 1w not to get data leakage because were using num_orders"""
    df, cols_dict = weekly_center_stats(df, cols_dict, 'category', 'food_nonfood', 'Center_total_cetegory_food_week0', 0, 0)
    df, cols_dict = weekly_center_stats(df, cols_dict, 'category', 'food_nonfood', 'Center_total_cetegory_food_week1', 1, 1)
    
    return df, cols_dict