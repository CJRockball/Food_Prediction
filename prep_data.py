import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, List

def df_fresh(df:pd.DataFrame)-> Tuple[pd.DataFrame, dict]:
    """ Set up a new dataframe for processing. 
        Create a column dict with column names and categories
        Change city_code and region_code to one category, city_region"""
    # Make dict with original df column names and types categories
    bin_cols = ['emailer_for_promotion', 'homepage_featured']
    cat_cols = ['center_id', 'meal_id', 'city_code', 'region_code', 'center_type', 'op_area', 'category', 'cuisine']
    num_cols = ['checkout_price', 'base_price', 'num_orders']
    cols_dict = {'bin_cols':bin_cols, 'cat_cols':cat_cols, 'num_cols':num_cols}

    # Making one column for city_code and region_code
    df['city_region'] = df['city_code'].astype('str') + '_' + df['region_code'].astype('str')
    df.drop(columns = ['city_code', 'region_code'], inplace = True)
    cols_dict['cat_cols'] += ['city_region']
    cols_dict['cat_cols'] = [item for item in cols_dict['cat_cols'] if item not in ('city_code', 'region_code')]
    
    return df, cols_dict


def prep_sum(df:pd.DataFrame, cols_dict:dict, geo_flag:bool=1):
    """ Prepares train/test.
        Create pipeline for making one hot encoding of categorica variables.
        Split train dataset in train,test,validate"""
    # If geoflag is true, then do ohe on city_region cat feature
    if geo_flag == True:
        try:
            cols_dict['cat_cols'].remove('city_region')
            df = df.drop('city_region', axis=1)
        except:
            print("city_region not in list")
        
    # Fit and apply one hot encoding pipeline
    df_ohe, ohe = make_ohe(df, cols_dict)
    # Process df and extract labels and features
    df_prep, labels, features = prep_df(df_ohe)
    # Split data in train/test/val
    x_train, y_train, x_val, y_val = train_val_test(df_prep, labels, features)
    
    return x_train, y_train, x_val, y_val, ohe 



def prep_test(df:pd.DataFrame, cols_dict:dict, geo_flag:bool=1, ohe_transformer = None) -> Tuple[pd.DataFrame, List]:
    """ Df processing for test dataset, i.e. missing label (num_orders).
        Add city_region for ohe if geo_flag set.
        Apply ohe transformer from training and move it to df and concatenate for non processed df
                 
        """
        # If geoflag is true, then do ohe on city_region cat feature
    if geo_flag == True:
        try:
            cols_dict['cat_cols'].remove('city_region')
            df = df.drop('city_region', axis=1)
        except:
            print("city_region not in list")
    
    # Apply ohe transformer
    mod_cols_arr = ohe_transformer.transform(df[cols_dict['cat_cols']]).toarray()   
    # Make compound names for new columns
    new_name_cols = [f'{i}_{j}' for i in cols_dict['cat_cols'] for j in df[i].unique()]
    # Put transformed data in a df
    df_proc = pd.DataFrame(data=mod_cols_arr, columns=new_name_cols)

    # Concatenate processed columns to non processed df
    # Drop categorical cols
    df = df.drop(columns=cols_dict['cat_cols'])
    df = df.drop(['id'], axis=1)
    # Concate 
    df_ohe = pd.concat([df, df_proc], axis=1)
    df_ohe.dropna(axis = 0, inplace = True)
    
    # Get all feature names for future use
    features = df_ohe.columns.to_list()
    
    return df_ohe, features


def prep_df(df:pd.DataFrame) -> Tuple[pd.DataFrame, List, List]:
    """ Small helper function to clean up df and extract feature names"""
    # Dropping the NaN rows which would be there in the data due to the lags
    df.dropna(axis = 0, inplace = True)
    # Drop id col. Make Feature and label lists
    df = df.drop(['id'], axis=1)
    labels = ['num_orders']
    features = df.columns.to_list()
    features.remove('num_orders')

    return df, labels, features

def prep_sum_final(df:pd.DataFrame, cols_dict:dict, geo_flag:bool=0):
    """ Df processing for training dataset for final model training (i.e. all data in x_train).
        Use city_region ohe if geo_flag set.
        Train, apply ohe training pipeline
        Move 'num_orders' feature to y_train as label"""
    if geo_flag == True:
        try:
            cols_dict['cat_cols'].remove('city_region')
            df = df.drop('city_region', axis=1)
        except:
            print("city_region not in list")
        
    # Set up, apply ohe pipeline
    df_ohe, ohe = make_ohe(df, cols_dict)
    df_prep, labels, features = prep_df(df_ohe)

    # Get x,y dfs
    x_train = df_prep[features]
    y_train = df_prep[labels].squeeze()
    
    return x_train, y_train, ohe


def make_ohe(df:pd.DataFrame, cols_dict:dict):
    """ Function to make and apply ohe pipeline"""
    # Get cols to process
    cat_cols = cols_dict['cat_cols']
    
    # Set up object, fit and apply to train df
    ohe = OneHotEncoder()
    ohe.fit(df[cat_cols])
    mod_cols_arr = ohe.transform(df[cat_cols]).toarray()

    # Create df for transformed columns
    # Make compound names for new columns
    new_name_cols = [f'{i}_{j}' for i in cat_cols for j in df[i].unique()]
    # Add processed columns to df
    df_proc = pd.DataFrame(data=mod_cols_arr, columns=new_name_cols)
    
    # Drop org cols and combine org df and new ohe cols
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, df_proc], axis=1)
    
    return df, ohe



def train_val_test(df:pd.DataFrame, labels:List[str], features:List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Function to split data set into train and test."""

    # Training set from week 125
    train_data = df[df['week'] <= 125]
    x_train = train_data[features]
    y_train = train_data[labels].squeeze()

    # Validation set of week 125 to 135    
    test_data = df[(df['week'] > 125) & (df['week'] <= 135)]
    x_test = test_data[features]
    y_test = test_data[labels].squeeze()

    return x_train, y_train, x_test, y_test




