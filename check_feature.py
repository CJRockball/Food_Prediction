#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from typing import List, Tuple

from feature_engineering import feat_eng
from feature_engineering_fcn import basic
from prep_data import prep_test, prep_sum_final, df_fresh, prep_sum
from pred_util import set_up_train_test, save_prediction, add_prediction_to_train, add_new_cols
from train_model import make_model, make_final_model
from feature_engineering_fcn import basic, num_lag, two_grp, single_grp, global_stats_lag, weekly_stats, weekly_center_stats

%load_ext autoreload
%autoreload 2

# Function for reproducible results
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(1)


train = pd.read_parquet('data/train/train_w_zeros.parquet')
#train = train.dropna()
print(train.shape)
train = train.sort_values(['meal_id', 'center_id', 'week']).reset_index(drop=True)
test = pd.read_parquet('data/test/test_w_zeros.parquet')
#test = test.dropna()
print(test.shape)
test = test.sort_values(['meal_id', 'center_id', 'week']).reset_index(drop=True)

# Get testing matrices
# Standard matrix to check basic calculations (15 rows)
check_standard = train.iloc[0:15,:]
# Matrix with 0 num_orders to check nans (20 rows)
check_mixed_nan = train.iloc[-90:-70,:]
# Large matrix to check groupby (20 rows, 2 groups)
check_mixed_groups = train.iloc[135:155,:]

# List week 110 to 115 for meal/center combo 200 to 205
grp_list = [j+(i*145) for i in range(200,205) for j in range(110,115)]
check_emails = train.iloc[grp_list,:]
display(check_emails)

#%% ---------------------------------------------------------------------

print('***** Set up *****')
df = check_standard.copy(deep=True)  
df_full, cols_dict = df_fresh(df)
org_cols_list = df_full.columns.values
df, cols_dict   = basic(df_full, cols_dict)
#df_full, cols_dict = feat_eng(df_full, cols_dict, 12, train_flag=1)

# Print out new columns
check_list = [i for i in df_full.columns.values if i not in org_cols_list]
df_check = df_full[check_list]
display(df_check)

# Calculate new col to compare


# %% 
# Check Weekly status feature

def week_stats_test(df: pd.DataFrame, col_category:str, col_type:str, col_to_ave_over:str, new_col_name:str, lag_flag:bool, n_shift:int):
    # Set up test df
    df = train.loc[(train.index < 400) & (train.center_type == 'TYPE_B') & (train.week < 5), :].copy(deep=True)    
    # Standard preprocess
    df_full, cols_dict = df_fresh(df)
    df_full, cols_dict = basic(df_full, cols_dict)
    #display(df_full)
    org_cols_list = df_full.columns.values
 
    # Use a different way to calculate the new colulmns 
    # to compare the test function result with a correct answer 
    def category_week_mean_function(df):
        calc_list = []
        for i in range(1,df.week.max()+1):
            calc_temp = df.loc[(df.week == i) & (df[col_category] == col_type),['num_orders']].mean().values[0]
            calc_list.append(calc_temp)
        return calc_list

   # Run function to be tested
    df_full, cols_dict = weekly_stats(df_full, cols_dict, col_category, col_to_ave_over, new_col_name, lag_flag, n_shift)
    # Print out new columns added by function
    check_list = [i for i in df_full.columns.values if i not in org_cols_list]
    df_check = df_full[check_list]
    display(df_check)


    calc_list = category_week_mean_function(df_full)
    # Manual comaprison
    print(calc_list)
    # Automate comparison
    df_list = df_full.loc[df_full.index < (len(df_full) // 2),new_col_name].dropna().to_list()
    print(df_list)
    assert df_list == calc_list[:-n_shift]
     
    return

week_stats_test(train, 'category', 'Beverages', 'num_orders', 'Avg_orders_category_week', 1, 1)
week_stats_test(train, 'center_type', 'TYPE_B', 'num_orders', 'Avg_orders_centert_week', 1, 1)
week_stats_test(train, 'city_region', '590_56', 'num_orders', 'Avg_orders_centert_week', 1, 1)
week_stats_test(train, 'cuisine', 'Italian', 'num_orders', 'Avg_orders_centert_week', 1, 1)


#%% 
# Chek single grp feature (for group_by with week)

def single_grp_check(check_emails, grp_by:str, col_to_process:str, new_name:str, lag_flag:bool, n_shift):
    # Make test df
    grp_list = [j+(i*145) for i in range(200,205) for j in range(108,113)]
    check_emails = train.iloc[grp_list,:].copy(deep=True)
    #display(check_emails)
    
    # Make answer
    week_min = check_emails.week.min()
    week_max = check_emails.week.max()
    weekly_prom = []
    for i in range(week_min, week_max+1):
        num_weekly_email_prom = check_emails.loc[check_emails.week == i, col_to_process].sum()
        weekly_prom.append(num_weekly_email_prom)

    # Add in lag shift
    weekly_prom = [np.nan] + weekly_prom[:-1]
    

    # Get test function answer
    df_full, cols_dict = df_fresh(check_emails)
    df_full, cols_dict = basic(df_full, cols_dict)
    df_full, cols_dict = single_grp(df_full, cols_dict, grp_by, col_to_process, new_name, lag_flag, n_shift)
    #display(df_full)


    # Compare
    num_of_weeks = week_max - week_min
    df_list = df_full.iloc[:num_of_weeks+1, :][new_name].to_list()
    # Manual compare
    print(weekly_prom)
    print(df_list)

    # Automated
    # [nan] != [nan]
    assert weekly_prom[n_shift:] == df_list[n_shift:]

single_grp_check(check_emails, 'week', 'emailer_for_promotion', "email_week_sum", 'sum', 1, 1)
single_grp_check(check_emails, 'week', "homepage_featured", "homepage_week_sum", 'sum', 1, 1)

# %%
# Ad hoc test for new feature
# Get df with 5 row random sample of meal/center
rnd_list = np.random.randint(1,300,5)
grp_list = [j+(i*145) for i in rnd_list for j in range(108,113)]
check_emails = train.iloc[grp_list,:].copy(deep=True)
#display(check_emails)

# Make answer
week_min = check_emails.week.min()
week_max = check_emails.week.max()

# Get test function answer
df_full, cols_dict = df_fresh(check_emails)
df_full, cols_dict = basic(df_full, cols_dict)
df_full, cols_dict = two_grp(df_full, cols_dict, ["week", "cuisine"], "ordered", "cuisine_week_count", 'sum',0, 0)
display(df_full)


# %%
df_test = train.copy()
# Get test function answer
df_full, cols_dict = df_fresh(df_test)
df_full, cols_dict = basic(df_full, cols_dict)
df, cols_dict = weekly_center_stats(df_full, cols_dict, 'category', 'food_nonfood', 'Center_total_cetegory_food_week', 0, 1)

display(df)



#%%

#df_test = train.copy()
# Set up data set
# all_center_numbers = train.center_id.unique()
# five_random = np.random.choice(all_center_numbers, size=5)
# df_test = train.loc[(train.center_id.isin(five_random)) & (train.week < 6),:].copy()
df_test = train.loc[(train.center_id.isin([10, 11])) & (train.week < 3),:].copy()

# Create food count
food = ['Rice Bowl', 'Pasta', 'Biryani', 'Pizza', 'Seafood', 'Salad', 'Fish', 'Soup']
other = ['Beverages', 'Starters', 'Sandwich', 'Extras', 'Other Snacks']
# Add new columns
df_test['ordered'] = 1 
df_test.loc[df_test.id.isnull(),'ordered'] = 0
df_test['food_nonfood'] = np.where(df_test['category'].isin(food), 1, 0)
print(df_test.shape)
display(df_test.head(20))


#sum_on_cat = df_test.groupby(['week', 'center_id', 'category'])['food_nonfood'].sum().to_frame().reset_index()
#display(sum_on_cat)
#total_week = sum_on_cat.groupby(['week', 'center_id']).food_nonfood.sum().to_frame().reset_index()
#display(total_week)

sum_on_cat = df_test.groupby(['week', 'center_id', 'category', 'cuisine'])[['checkout_price']].count().reset_index()
print(sum_on_cat.shape)
display(sum_on_cat)
#total_week = sum_on_cat.groupby(['week', 'center_id']).food_nonfood.sum().to_frame().reset_index()
#display(total_week)

# %%
