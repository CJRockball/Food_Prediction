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

#%% --------------------------------------------------------------
# Train/test 
logging.info('-------------------------------------------------------------------')
logging.info("train_test_fcn")
# Add params for xgb
params = {'subsample': 0.9, 'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 0.9, 'colsample_bylevel': 0.9}
#{'n_estimators': 100} 
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
params = {'subsample': 0.9, 'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 0.9, 'colsample_bylevel': 0.9}
#{'n_estimators': 100}
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
param_list, sorted_metric = random_opt_func(train, lags, n=65)

# Get best set of params
print(sorted_metric)
# Print out list of parameters
#print(param_list[3])

# %%

# Print out list of parameters
print(param_list[46])

# %%
