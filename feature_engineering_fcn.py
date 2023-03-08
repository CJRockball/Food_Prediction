import pandas as pd
import numpy as np
from typing import List, Tuple

# Functionalize features
# 1. Stat features
# 2. Lag features
# 3. Lead features

# Log transform prices and label (num_orders)
def basic(df: pd.DataFrame, cols_dict: dict, train_flag: bool=1) -> Tuple[pd.DataFrame, dict]:
    """ This function transforms numerical features with log.
        It also calculates new features that are simple and known before hand (i.e. not based on num_orders)"""
    # Test data don not have the num_orders col so only transform on train data
    if train_flag == 1:
        df['num_orders'] = np.log1p(df['num_orders'])
    
    # Transform other numerical features   
    df['checkout_price']    = np.log1p(df['checkout_price'])
    df['base_price']        = np.log1p(df['base_price'])
    # Calculate basic statistics
    df['dscnt_pct']         = (df['base_price'] - df['checkout_price']) / df['base_price']
    # Bool indicator for discount
    df['neg_dscnt']         = (df['dscnt_pct'] < 0).astype(int)

    # Bool indicator for non zero col, this is known before hand because zero order cols are not offered that week
    df['ordered'] = 1 
    df.loc[df.id.isnull(),'ordered'] = 0
    # Create food count
    food = ['Rice Bowl', 'Pasta', 'Biryani', 'Pizza', 'Seafood', 'Salad', 'Fish', 'Soup']
    other = ['Beverages', 'Starters', 'Sandwich', 'Extras', 'Other Snacks']
    df['food_nonfood'] = np.where(df['category'].isin(food), 1, 0)

    # rank_cp = df.groupby(['week', 'center_id'])[['checkout_price']].rank().rename(columns = {'checkout_price':'rank_on_cp'})
    # df = df.join(rank_cp)
    
    df.sort_values(['meal_id', 'center_id', 'week'], inplace=True)
    
    # Add new cols to cols_dict for processing future
    cols_dict['num_cols'] += ['dscnt_pct', 'rank_on_cp']
    cols_dict['bin_cols'] += ['neg_dscnt']
    
    return df, cols_dict


def single_grp(df, cols_dict, grp_by:str, col_to_process:str, new_name:str, type:str, lag_flag:bool, n_shift:int) -> Tuple[pd.DataFrame, dict]:
    """ This function is used to group and aggregate over one column """
    # Total number of meal email add
    if type == 'sum':
        grpby_df = df.groupby([grp_by])[col_to_process].sum().reset_index() 
    elif type == 'count':
        grpby_df = df.groupby([grp_by])[col_to_process].count().reset_index() 

    grpby_df.columns = [grp_by, new_name]
    df = pd.merge(df,grpby_df, on=[grp_by], how="left")
    df.sort_values(['meal_id', 'center_id'], inplace = True)
    
    # Shift to avoid data leakage
    if lag_flag == True:
        df = data_shift1(df, [new_name], n_shift)    

    # Add new column names to cols_dict
    if new_name not in cols_dict['num_cols']:
        cols_dict['num_cols'].append(new_name)    
    
    return df, cols_dict


def two_grp(df, cols_dict, grp_by:List, col_to_process:str, new_name:str, type_flag:str, lag_flag:bool, n_shift:int) -> Tuple[pd.DataFrame, dict]:
    """ This function is used to group and aggregate over one column """
    if type_flag == 'sum':
        grpby_df = df.groupby(grp_by)[col_to_process].sum().reset_index()
    elif type_flag == 'mean':
        grpby_df = df.groupby(grp_by)[col_to_process].mean().reset_index() 
    elif type_flag == 'count':
        grpby_df = df.groupby(grp_by)[col_to_process].count().reset_index() 

    grpby_df.columns = grp_by + [new_name] #["meal_id", "week", "meal_week_count"]
    df = pd.merge(df,grpby_df, on=grp_by, how="left")
    df.sort_values(['meal_id', 'center_id'], inplace = True)
    
    # Shift to avoid data leakage
    if lag_flag == True:
        df = data_shift1(df, [new_name], n_shift)    

    # Add new column names to cols_dict
    if new_name not in cols_dict['num_cols']:
        cols_dict['num_cols'].append(new_name)    
    
    return df, cols_dict


# Lag num_orders 0 to x steps
def num_lag(df: pd.DataFrame, num_of_lags: int=4) -> pd.DataFrame:
    """ Function to lag num_orders column """
    # Make df copu
    df_shift = df.sort_values(by=['meal_id', 'center_id']).reset_index(drop=True)
    
    # Iterate number or lags and create a lag df
    name_list = []
    for i in range(1,num_of_lags+1,1):
        tmp = df.groupby(['meal_id', 'center_id'])['num_orders'].shift(i).reset_index(drop=True).to_frame()
        fname = f'num_orders_lag_{i}w'
        df_shift[fname] = tmp.iloc[:,0]
        name_list.append(fname)

    # Add lag df to original df
    df_shift = df_shift[name_list]
    # Sorting the columns so that it is easy to concatenate 
    df.sort_values(['meal_id', 'center_id'], inplace = True)
    # Merging the columns to the original data 
    df = pd.concat([df, df_shift], axis = 1)
    df = df.loc[:,~df.columns.duplicated()]

    return df


def data_shift1(df: pd.DataFrame, add_cols: List, n_shift=1) -> pd.DataFrame:
    """ Help function to shift columns lag or lead"""
    df_shift = df.sort_values(by=['meal_id', 'center_id']).reset_index(drop=True)
    
    for item in add_cols:
        tmp = df.groupby(['meal_id', 'center_id'])[item].shift(n_shift).reset_index(drop=True).to_frame()
        df_shift[item] = tmp.iloc[:,0]
        
    df_shift = df_shift[add_cols]
    # Sorting the columns so that it is easy to concatenate
    df = df.drop(columns= add_cols)
    df.sort_values(['meal_id', 'center_id'], inplace=True)
    df = pd.concat([df, df_shift], axis = 1)
    df = df.loc[:,~df.columns.duplicated()]
    return df


# Add mixed stats features
def global_stats_lag(df:pd.DataFrame, cols_dict:dict, grp_cols: List, new_col_name:str, col_to_act_on:str, stat_fcn:str, lag_flag:bool, n_shift:int) -> Tuple[pd.DataFrame, dict]:
    """ Function to aggregate over 'up to now'. """
    df.set_index('week', inplace=True)
    df[col_to_act_on] = df[col_to_act_on].replace(0,np.NaN)

    # Different aggregation functions
    if stat_fcn == 'min':
        expand_col = df.groupby(grp_cols)[col_to_act_on].expanding().min().reset_index()
    elif stat_fcn == 'max':
        expand_col = df.groupby(grp_cols)[col_to_act_on].expanding().max().reset_index()
    elif stat_fcn == 'mean':
        expand_col = df.groupby(grp_cols)[col_to_act_on].expanding().mean().reset_index()
    elif stat_fcn == 'std':
        expand_col = df.groupby(grp_cols)[col_to_act_on].expanding().std().reset_index()
    
    # Put new col in a df    
    expand_col.set_index('week', inplace = True)
    expand_col.rename(columns = {col_to_act_on: new_col_name}, inplace = True)
    
    ########################### Matrix works
    # Add new df to original df
    # Sorting the columns to concatenate 
    df.sort_values(grp_cols, inplace = True)
    # Merging the columns to the original data 
    df = pd.concat([df, expand_col], axis = 1)
    df = df.loc[:,~df.columns.duplicated()]
    df[col_to_act_on] = df[col_to_act_on].replace(np.NaN,0)
    df[new_col_name] = df[new_col_name].replace(np.NaN,0)
    df['week'] = df.index
    df = df.reset_index(drop=True)
    
    # Shift to avoid data leakage
    if lag_flag == True:
        df = data_shift1(df, [new_col_name], n_shift)

    # Add new column names to cols_dict
    if new_col_name not in cols_dict['num_cols']:
        cols_dict['num_cols'].append(new_col_name) 
    
    return df, cols_dict


# Seeing weekly trend across the categorical variables
def weekly_stats(df: pd.DataFrame, cols_dict: dict, col_category:str, col_to_ave_over:str, new_col_name:str, lag_flag:bool, n_shift:int) -> Tuple[pd.DataFrame, dict, List]:
    """ Calculate averages for week and categories """
    df.set_index('week', inplace=True)
    df['num_orders'] = df['num_orders'].replace(0,np.NaN)
    
    # The average of the Category (say Beverages) in the corresponding week (week 1, 2, 3, etc)
    new_column = df.groupby(['week', col_category])[col_to_ave_over].mean().reset_index().sort_values([col_category, 'week']).set_index('week').rename(columns = {col_to_ave_over:new_col_name})
    df = pd.merge(df, right = new_column, on = ['week', col_category], how = 'inner', validate = 'many_to_one')
    
    df.sort_values(['meal_id', 'center_id'], inplace = True)
    df['week'] = df.index
    df = df.reset_index(drop=True)
    df['num_orders'] = df['num_orders'].replace(np.NaN,0)
    df[new_col_name] = df[new_col_name].replace(np.NaN, 0)
     
    # Shift to avoid data leakage 
    if lag_flag == True: 
        df = data_shift1(df, [new_col_name], n_shift)
        
    # Add new column names to cols_dict
    if new_col_name not in cols_dict['num_cols']:
        cols_dict['num_cols'].append(new_col_name) 
    
    return df, cols_dict

# Seeing weekly trend across the categorical variables
def weekly_center_stats(df: pd.DataFrame, cols_dict: dict, col_category:str, col_to_ave_over:str, new_col_name:str, lag_flag:bool, n_shift:int) -> Tuple[pd.DataFrame, dict, List]:
    """ Calculate averages for week and categories """
    
    # The average of the Category (say Beverages) in the corresponding week (week 1, 2, 3, etc)
    sum_df = df.groupby(['week', 'center_id', col_category])[col_to_ave_over].sum().to_frame().reset_index()
    total_week = sum_df.groupby(['week', 'center_id'])[col_to_ave_over].sum().to_frame().reset_index().rename(columns = {col_to_ave_over:new_col_name})

    df = pd.merge(df, right = total_week, on = ['week', 'center_id'], how = 'inner', validate = 'many_to_one')
    
    df.sort_values(['meal_id', 'center_id'], inplace = True)
     
    # Shift to avoid data leakage 
    if lag_flag == True: 
        df = data_shift1(df, [new_col_name], n_shift)
        
    # Add new column names to cols_dict
    if new_col_name not in cols_dict['num_cols']:
        cols_dict['num_cols'].append(new_col_name) 
    
    return df, cols_dict
