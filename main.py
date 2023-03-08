#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import logging


from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from fit_call import train_func, prediction_func, plot_sample, submit_save, random_opt_func

%load_ext autoreload
%autoreload 2

# Function for reproducible results
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(1)

logging.basicConfig(
    filename='experiment_log.log',
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
)

# Load train and test data
train = pd.read_parquet('data/train/train_w_zeros.parquet')
print(train.shape)
test = pd.read_parquet('data/test/test_w_zeros.parquet')
print(test.shape)
test = test.sort_values(['meal_id', 'center_id', 'week']).reset_index(drop=True)
# Constants
lags = 12

#%%

# cat_check = train.category.unique()
# print(cat_check)

food = ['Rice Bowl', 'Pasta', 'Biryani', 'Pizza', 'Seafood', 'Salad', 'Fish', 'Soup']
other = ['Beverages', 'Starters', 'Sandwich', 'Extras', 'Other Snacks']

# for i in range(5):
#     meal_dict = {}
#     for k in food:
#         meal_dict[k] = train.loc[(train.center_id == 10) & (train.week == i) & (train.category == k),['meal_id']].count()
#     print(meal_dict)

all_center_numbers = train.center_id.unique()
five_random = np.random.choice(all_center_numbers, size=5)
df_test = train.loc[(train.center_id.isin(five_random)) & (train.week < 6),:].copy()
#df_test = train.loc[(train.center_id == 10) & (train.week < 6),:].copy()

df_test['ordered'] = 1 
df_test.loc[df_test.id.isnull(),'ordered'] = 0
df_test['food_nonfood'] = np.where(df_test['category'].isin(food), 1, 0)
print(df_test.shape)
#display(df_test.head(20))


sum_on_cat = df_test.groupby(['week', 'center_id', 'category'])['food_nonfood'].sum().to_frame().reset_index()
display(sum_on_cat)
total_week = sum_on_cat.groupby(['week', 'center_id']).food_nonfood.sum().to_frame().reset_index()
display(total_week)


    
#%% --------------------------------------------------------------
# Train/test 
logging.info('-------------------------------------------------------------------')
logging.info("train_test_fcn")
# Add params for xgb
params = {'n_estimators': 50} #{'subsample': 0.9, 'n_estimators': 250, 'max_depth': 7, 'learning_rate': 0.1, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.8}
# Train model, val model
trained_model, ohe = train_func(train, lags, final_flag=0, model_type='rf_xgb', params=params)

# Run a one week at a time prediction test
logging.info('Model Test')
df_predictions = prediction_func(test, train, lags, ohe, trained_model, pred_flag=0)
df_predictions = df_predictions.dropna()

# plot test meal/center
plot_sample(df_predictions, train)


# %% Final model ----------------------------------------------------
logging.info('-------------------------------------------------------------------')
logging.info("final_train")
# make final model, train with whole dataset
params = {'n_estimators': 45} #{'subsample': 0.9, 'n_estimators': 250, 'max_depth': 7, 'learning_rate': 0.1, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.8}
trained_model, ohe = train_func(train, lags, final_flag=1, model_type='rf_xgb', params=params)
# predict on AV test sample, one week at a time
df_predictions = prediction_func(test, train, lags, ohe, trained_model, pred_flag=1)
df_predictions = df_predictions.dropna()
display(df_predictions.head(20))

# plot test meal/center
plot_sample(df_predictions, train)

# %%
# make submission file    
logging.info('saving submission')
submit_save(df_predictions)

# %% 
# Run random sample optimization
# Returns list of sorted metrics to show the best one
# Returns list with all paramters run
param_list, sorted_metric = random_opt_func(train, lags, n=10)

# Get best set of params
print(sorted_metric[0])
# Print out list of parameters
print(param_list[3])

# %%
