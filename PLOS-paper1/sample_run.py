import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import plotly.express as px
import plotly.graph_objects as go
import plotly
from plotly import version
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from datetime import datetime
import random
import pickle
from rolling_LDA import rolling_LDA
from rolling_analysis import rolling_analysis
from regression import Regression

# Corpus
df = pd.read_csv('full_content.csv' , index_col=0)

# HSI
std_resid = pd.read_csv('HSI_with_GARCH_resid.csv', index_col=0)

# Join the dataframe to prepare the finalised dataframe
joined_df = df.join(std_resid['abs_resid']).reset_index()

# Deal with the news releasing dates which are non-trading days
index = 0
for ix, row in joined_df.iterrows():
    is_na = pd.isna(joined_df.loc[ix, 'abs_resid'])
    if not is_na:
        index = ix
    if is_na:
        joined_df.loc[ix, 'index'] = joined_df.loc[index, 'index']

joined_df.rename(columns={'index': 'date'}, inplace=True)
trading_days = joined_df.set_index('date')
trading_days.rename(columns={'body': 'content'},inplace=True)
trading_days.drop(columns='abs_resid', inplace=True)

# Entire pipeline
top_word, doc_top, gini, top_word_div_ranked = rolling_LDA(trading_days, '2017-01-01','2020-12-31', max_df= 0.3, min_df=0.001, n_topics= 15, window_size=30, roll_size=1, score = 'all').period_summary()

# Then call the rolling_analysis function to get the full dataframes (data with lags), the parameters are not important
full_df_word_div, top_mult_corr_word_div, lag_mult_corr_word_div, timeframes = rolling_analysis(top_word,std_resid[['abs_resid']], '2015-10-20','2015-11-29', 120,1).Multiple_Correlations()
full_df_original, top_mult_corr_original, lag_mult_corr_original, timeframes = rolling_analysis(doc_top,std_resid[['abs_resid']], '2015-10-20','2015-11-29', 120,1).Multiple_Correlations()
gini.columns = [0]
full_df_gini, top_mult_corr_gini, lag_mult_corr_gini, timeframes = rolling_analysis(gini,std_resid[['abs_resid']], '2015-10-20','2015-11-29', 120,1).Multiple_Correlations()
full_df_div_ranked, top_mult_corr_div_ranked, lag_mult_corr_div_ranked, timeframes = rolling_analysis(top_word_div_ranked,std_resid[['abs_resid']], '2015-10-20','2015-11-29', 120,1).Multiple_Correlations()

# Data Preparation
original = full_df_original[[x for x in list(full_df_original.columns) if re.findall(r'\d+t',str(x))]]

word_gini = full_df_word_div[[x for x in list(full_df_word_div.columns) if re.findall(r'\d+t',str(x))]]
word_gini.columns = [str(x) + 'gini_w' for x in word_gini.columns]

top_gini = full_df_gini[[x for x in list(full_df_gini.columns) if re.findall(r'\d+t',str(x))]]
top_gini.columns = [str(x) + 'gini_t' for x in top_gini.columns]

div_ranked_gini = full_df_div_ranked[[x for x in list(full_df_div_ranked.columns) if re.findall(r'\d+t',str(x))]]
div_ranked_gini.columns = [str(x) + 'gini_d' for x in div_ranked_gini.columns]

# Declare the input for data preparation
pre_processed_dfs = [original, top_gini, word_gini, div_ranked_gini]
Y = np.log(std_resid[['abs_resid']])
Y.index = pd.to_datetime(Y.index)

# Preprocess the dataframes using logit transformation
proc_original, proc_top_gini, proc_word_gini, proc_div_ranked = Regression.Input_data_Preparation('logit', pre_processed_dfs)

# Choose all lag one - Data Preparation for X
def choose_all_lag_x(input_dataframes, lag):
    all_lag_one = pd.DataFrame()
    for df in input_dataframes:
        for col_name in list(df.columns):
            pattern = '.+t-' + str(lag)
            if re.findall(pattern, col_name) != []:
                all_lag_one = pd.concat([all_lag_one, df[[col_name]]], axis = 1)
                
    return all_lag_one

input_dataframes = [proc_original, proc_top_gini, proc_word_gini, proc_div_ranked]
final_Xs = choose_all_lag_x(input_dataframes, 1)
final_Xs = [final_Xs]

# Data Preparation for Y
# Y1 - log_abs_GARCH_resid
Y1 = Y.loc[final_Xs[0].index]

# Configuration for running the regression
Xs = [
    final_Xs,
    final_Xs,
    final_Xs,
    final_Xs
]
params_dicts = [
    {'decision_threshold':[0], 'n':[1], 'interaction_terms':[False]},
    {'decision_threshold':[0], 'n':[2], 'interaction_terms':[False]},
    {'decision_threshold':[0], 'n':[2], 'interaction_terms':[True]}
]
Ys = [Y1, Y1, Y1, Y1]
p = [0, 0, 0, 0]
rolling_mean_y = [True, True, True, True]
tolerances = ['high','high','high', 'high']
X_inrange_rules = [True, True, True, True]
Y_inrange_rules = [False, False, False, False]
check_inf_obs = [True, True, True, True]
model_names = ['model-1','model-2','model-3', 'model-4']
window_size = [30, 60, 90, 120]

# Run the regression
model_setting = Regression('2017-02-15', '2020-12-31', 1)
result = model_setting.fit_with_Experiments(Xs, Ys, model_names, params_dicts, window_size, p, rolling_mean_y, 
                                            [15,30,45,60], tolerances, X_inrange_rules, Y_inrange_rules, check_inf_obs)

# Lastly, run the 3x3 graphs report
Pred_Results = Regression.Model_Prediction_Comparison_Report(result, 0.1, 0.9, 0.01, show = False)

# Save the regression result
with open('{}-RESULT.pkl'.format(datetime.now().strftime('%Y%m%d%H%M')), 'wb') as f:
    pickle.dump(result, f)

# Save the Pred result
with open('{}-PREDRESULT.pkl'.format(datetime.now().strftime('%Y%m%d%H%M')), 'wb') as f:
    pickle.dump(Pred_Results, f)

# Save the dependent variable used
with open('{}-Y1.pkl'.format(datetime.now().strftime('%Y%m%d%H%M')), 'wb') as f:
    pickle.dump(Y1, f)
