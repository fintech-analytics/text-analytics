from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import kurtosis, skew
import statsmodels.api as sm
from scipy.stats import kurtosis, skew
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import KFold
from sklearn.linear_model import lasso_path
from sklearn.linear_model import Lasso
import warnings
from sklearn.preprocessing import StandardScaler
import itertools
from itertools import combinations
from matplotlib.backends.backend_pdf import PdfPages
import os
from statsmodels.stats.outliers_influence import OLSInfluence
warnings.filterwarnings("ignore")



class Regression:
    def __init__(self, start, end, roll_size):
        self.start = start
        self.end = end
        self.rs = roll_size
        self.Y = None
        self.X = None
        self.rolling_mean_df = None
        self.window_Y_test = None
        self.n = None
        self.interaction_terms = None
        self.p = None
        self.rolling_mean_y = None

    def fit(self, window_size, Y, X_list, decision_threshold , n, interaction_terms = False, p = 0, 
            rolling_mean_y = False, tolerance = 'high', X_inrange_rule = True, Y_inrange_rule = False, check_inf_obs = True):
        """
        Note that the input variables should have different column names
        """
        # Declare parameters
        end_index = 0
        start_date = datetime.strptime(self.start, '%Y-%m-%d')
        end_date = datetime.strptime(self.end, '%Y-%m-%d')
        roll_start = start_date + timedelta(days=0)
        roll_end = roll_start + timedelta(days=window_size)
        self.n = n
        self.interaction_terms = interaction_terms
        self.p = p
        self.rolling_mean_y = rolling_mean_y

        # Join the input Xs together
        if X_list != []:
            X = pd.concat(X_list, axis = 1)
            params = pd.DataFrame(index = list(X.columns))
            self.X = X
        else:
            params = pd.DataFrame()
        self.Y = Y
        # For regression statistics of the final chosen model
        Regression_statistics = []
        predictions = pd.DataFrame(index = Y[Y.index >= str(roll_end)].index, columns = ['predictions','benchmark','true_y','acc_RMSE'])

        # Create rolling mean if necessary
        if rolling_mean_y == True:
            # Because start_date must be a window_size earlier, have to reset start_date for Y
            rolling_mean_start_date = start_date - timedelta(days=window_size+5)
            rolling_mean_start_date = str(rolling_mean_start_date.date())
            rolling_mean_df = self.rolling_mean_residual(Y, rolling_mean_start_date, self.end, window_size, self.rs)
            self.rolling_mean_df = rolling_mean_df
        
        while roll_end <= end_date:
            
            # Declare Y train
            window_Y = Y[(Y.index >= str(roll_start)) & (Y.index < str(roll_end))]
            
            # Prediction window - test sets of X and Y
            window_Y_test = Y[Y.index >= str(roll_end)].iloc[:1]
            self.window_Y_test = window_Y_test
            
            # Track the index
            index = list(window_Y.index)[-1]
            if index == end_index:
                roll_start = roll_start + timedelta(days = self.rs)
                roll_end = roll_end + timedelta(days=self.rs)
                continue
            else:
                end_index = index
                pass
            
            # Build and compare topic score models with the list of predictors (X) provided, else return benchmark model
            if X_list != []:
                window_X = X[(X.index >= str(roll_start)) & (X.index < str(roll_end))]
                sorted_models = self.Best_Subset_Regression(window_X, window_Y, n, interaction_terms, p, rolling_mean_y, check_inf_obs)
                # Determine the final model by checking whether X_test and prediction are within range
                best_topic_model, pred_from_topic_model = self.choose_appropriate_model(sorted_models, window_X, window_Y, tolerance, X_inrange_rule, Y_inrange_rule)                
            elif X_list == [] and rolling_mean_y == True:
                window_X = self.rolling_mean_df[(self.rolling_mean_df.index >= str(roll_start)) & (self.rolling_mean_df.index < str(roll_end))]
                if p > 0:
                    AR_to_add = self.create_lag_p_df(self.Y, p)
                    window_X = window_X.join(AR_to_add)
                fitted_model = self.simple_OLS(window_X, window_Y, has_constant = 'add')
                best_topic_model = (fitted_model, fitted_model.rsquared, None)
            else:
                best_topic_model = (None, -1, None)

            # Check whether best model is larger than the threshold
            if best_topic_model[1] >= decision_threshold and X_list != []:
                final_model = best_topic_model[0]
                final_model_name = 'Topic Scores'
                pred = pred_from_topic_model
            elif X_list == [] and rolling_mean_y == True:
                final_model = best_topic_model[0]
                final_model_name = 'Rolling Mean Y with Intercept'
                window_X_test = self.rolling_mean_df.loc[window_Y_test.index,:]
                if p > 0:
                    window_X_test = window_X_test.join(self.create_lag_p_df(Y, p)) 
                pred = final_model.predict(sm.add_constant(window_X_test, has_constant='add'))
            else:
                if rolling_mean_y == True:
                    window_X = self.rolling_mean_df[(self.rolling_mean_df.index >= str(roll_start)) & (self.rolling_mean_df.index < str(roll_end))]
                    final_model = self.simple_OLS(window_X, window_Y, has_constant = 'add')
                    final_model_name = 'benchmark'
                    window_X_test = self.rolling_mean_df.loc[window_Y_test.index,:]
                    pred = final_model.predict(sm.add_constant(window_X_test, has_constant='add'))
                else:
                    final_model = sm.OLS(window_Y, np.ones(window_Y.shape)).fit()
                    final_model_name = 'benchmark'
                    pred = window_Y.mean()
            
            # Append the regression statistics, parameters and prediction
            modelrsqaure, modelrsqaure_adj, fstat, db, modelskew, modelkurtosis, bp = self.regression_stats(final_model)
            if final_model_name == 'benchmark':
                modelrsqaure, modelrsqaure_adj = (0,0)
            statistics = (window_Y.index[0],window_Y.index[-1],final_model_name,modelrsqaure,modelrsqaure_adj,fstat,db,modelskew,modelkurtosis,bp)
            Regression_statistics.append(statistics)
            # Save the chosen variables
            params[window_Y.index[-1]] = np.nan
            for variable, beta in zip(final_model.params.index, final_model.params.values):
                if variable in list(params.index):
                    params.loc[variable, window_Y.index[-1]] = beta
            # Benchmark model calculation
            if rolling_mean_y == True:
                benchmark_X = self.rolling_mean_df[(self.rolling_mean_df.index >= str(roll_start)) & (self.rolling_mean_df.index < str(roll_end))]
                benchmark_model = self.simple_OLS(benchmark_X, window_Y, has_constant = 'add')
                benchmark_X_test = self.rolling_mean_df.loc[window_Y_test.index,:]
                benchmark_pred = benchmark_model.predict(sm.add_constant(benchmark_X_test, has_constant='add'))
            else:
                benchmark_pred = window_Y.mean().values
            # Save the prediction results
            predictions.loc[window_Y_test.index, 'predictions'] = pred.values
            predictions.loc[window_Y_test.index, 'benchmark'] = benchmark_pred
            predictions.loc[window_Y_test.index, 'true_y'] = window_Y_test.values
            
            # Calculate the accumulated RMSE
            acc_RMSE = Regression.prediction_performance(predictions[['predictions','benchmark','true_y']].dropna())['RMSE_with_topic_score']
            predictions.loc[window_Y_test.index, 'acc_RMSE'] = acc_RMSE
                  
            # Move the window
            roll_start = roll_start + timedelta(days = self.rs)
            roll_end = roll_end + timedelta(days= self.rs)


        # Create Dataframe to organsize all results
        model_stat = pd.DataFrame({'period_start':[x[0] for x in Regression_statistics],'period_end':[x[1] for x in Regression_statistics],
                                   'model_name': [x[2] for x in Regression_statistics],
                                   'rsquare':[x[3] for x in Regression_statistics],'adj_rsquare':[x[4] for x in Regression_statistics],
                                   'f_statistic_p_val':[x[5] for x in Regression_statistics], 'durbinwatson':[x[6] for x in Regression_statistics], 
                                   'skewness':[x[7] for x in Regression_statistics],'kurtosis':[x[8] for x in Regression_statistics],
                                   'breusch_pagan_p_val':[x[9] for x in Regression_statistics]})

        
        predictions['predictions']= predictions['predictions'].astype('float')
        predictions['benchmark'] = predictions['benchmark'].astype('float')
        predictions['true_y'] = predictions['true_y'].astype('float')
        predictions['acc_RMSE'] = predictions['acc_RMSE'].astype('float')
        
        return model_stat, params.T, predictions
         
    def regression_stats(self, results):
        modelrsqaure = results.rsquared
        modelrsqaure_adj = results.rsquared_adj
        fstat = results.f_pvalue
        db = durbin_watson(results.resid)
        modelskew = skew(results.resid_pearson)
        modelkurtosis= stats.kurtosis(results.resid,fisher=False)
        bp=het_breuschpagan(results.resid, results.model.exog)
        return modelrsqaure, modelrsqaure_adj, fstat, db, modelskew, modelkurtosis, bp[-1]
    
    def simple_OLS(self, window_X, window_Y, has_constant = 'None'):
        model = sm.OLS(window_Y, sm.add_constant(window_X, has_constant=has_constant))
        results = model.fit()
        return results
        
    def Best_Subset_Regression(self, window_X, window_Y, n, interaction_terms, p, rolling_mean_y, check_inf_obs):
        # Create list to store the performance
        R2_list = []
        
        # Create subset variables as the X of the regression
        combs = list(combinations(list(window_X.columns), n))
        for comb in combs:
            subset_variables = window_X[list(comb)]
            
            # Set interaction term if necessary
            if n > 1 and interaction_terms == True:
                subset_variables = self.Add_interaction_terms(subset_variables)
                
            if p > 0:
                # Create a dataframe with lag (p) and join to the variable
                AR_to_add = self.create_lag_p_df(self.Y, p)
                subset_variables = subset_variables.join(AR_to_add)
                
            # if rolling mean == True then join rolling mean
            if rolling_mean_y == True:
                subset_variables = subset_variables.join(self.rolling_mean_df)

            # Call Regression and store the result
            fitted_model = self.simple_OLS(subset_variables, window_Y, has_constant = 'add')
            
            # Check whether there are inflential observations, if exists then drop it and rerun regression
            exist_inf_obs = self.check_influential_obs(fitted_model)
            if check_inf_obs == True:
                if exist_inf_obs[0]:
                    subset_variables = subset_variables.drop(exist_inf_obs[1])
                    refined_window_Y = window_Y.drop(exist_inf_obs[1])
                    fitted_model = self.simple_OLS(subset_variables, refined_window_Y, has_constant = 'add')
            
            R2_list.append((fitted_model, fitted_model.rsquared, list(comb)))
            
        # Sort models by R2
        sorted_models = sorted(R2_list, key=lambda x: x[1], reverse=True)
                
        return sorted_models
    
    def choose_appropriate_model(self, sorted_models, window_X, window_Y, tolerance, X_inrange_rule, Y_inrange_rule):
        best_topic_model = (None, -1, None)
        pred = None
        for model in sorted_models:
            # Predict with each model
            final_model = model[0]
            window_X_test = self.X.loc[self.window_Y_test.index,:][model[2]]
            if self.n > 1 and self.interaction_terms == True:
                window_X_test = self.Add_interaction_terms(window_X_test)
            if self.p > 0:
                window_X_test = window_X_test.join(self.create_lag_p_df(self.Y, self.p)) 
            if self.rolling_mean_y == True:
                window_X_test = window_X_test.join(self.rolling_mean_df)
            pred = final_model.predict(sm.add_constant(window_X_test, has_constant='add'))
            # If prediction and X_Test satistfy the criteria then return the final model
            finalise = True
            if X_inrange_rule == True:
                if not self.X_test_within_range(window_X, model[2]):
                    finalise = False
            if Y_inrange_rule == True:
                if not self.pred_within_range(pred, window_Y):
                    finalise = False
            if finalise == True:
                best_topic_model = model
                break
            else:
                if tolerance == 'high':
                    continue
                elif tolerance == 'low':
                    best_topic_model = (None, -1, None)
                    break
        # Return all things we need
        return best_topic_model, pred
    
    def COOKD_Filtering_Criteria(self, COOKD):
        By_one = ((COOKD >= 1).any(), COOKD[COOKD >= 1].index)
        return By_one

    def check_influential_obs(self, fitted_model):
        COOKD = OLSInfluence(fitted_model).cooks_distance[0]
        criterion = ((COOKD >= 1).any(), COOKD[COOKD >= 1].index)
        return criterion
        
    def pred_within_range(self, prediction, window_Y):
        within_range = True
        max_val = window_Y.max().values
        min_val = window_Y.min().values
        if prediction.values < min_val or prediction.values > max_val:
            within_range = False
        return within_range   
    
    def X_test_within_range(self, window_X, used_predictors):
        within_range = True
        for predictor in used_predictors:
            X_to_view = window_X[predictor]
            X_test_val = self.X.loc[self.window_Y_test.index,predictor].values
            max_val = X_to_view.max()
            min_val = X_to_view.min()
            if X_test_val < min_val or X_test_val > max_val:
                within_range = False
        return within_range
    

    def Add_interaction_terms(self, variables):
        nC2_to_interact = list(combinations(list(variables.columns), 2))
        for combination in nC2_to_interact:
            interaction_name = "*".join([str(x) for x in combination])
            variables[interaction_name] = variables[list(combination)].prod(axis = 1)
        return variables
    
    def create_lag_p_df(self, input_y, p):
        output = pd.DataFrame(index = input_y.index)
        for i in range(1, p+1):
            name = 'Y_t-{}'.format(i)
            output[name] = input_y.shift(i)
        output.dropna(inplace = True)
        return output
 
    
    def rolling_mean_residual(self, Y, start, end, window_size, roll_size):
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
        roll_start = start_date + timedelta(days=0)
        roll_end = roll_start + timedelta(days=window_size)
        rs = roll_size

        # Set the df index to datetime
        Y.index = pd.to_datetime(Y.index)

        # Create storing objects
        rolling_mean_df = pd.DataFrame()
        rolling_var_df = pd.DataFrame()
        period_start = []
        period_end = []

        end_index = 0

        # Roll the window
        period = 0
        while roll_end <= end_date:
            window = Y[(Y.index >= roll_start) & (Y.index < roll_end)]
            print(roll_start, roll_end)
            # Track the index
            index = list(window.index)[-1]
            if index == end_index:
                roll_start = roll_start + timedelta(days = rs)
                roll_end = roll_end + timedelta(days=rs)
                continue
            else:
                end_index = index
                pass

            # Print current work        
            periodstart = window.index[0]
            period_start.append(periodstart)  
            periodend = window.index[-1]
            timet = Y[Y.index >= str(roll_end)].iloc[:1]
            period_end.append(str(list(timet.index)[0].date()))

            # Calculate the rolling statistics - mean
            rolling_mean = window.mean().to_frame().T
            rolling_mean.index = [period]
            rolling_mean_df = pd.concat([rolling_mean_df,rolling_mean])

            # Move the window
            roll_start = roll_start + timedelta(days = rs)
            roll_end = roll_end + timedelta(days=rs)
            period += 1

        # Append period starts and ends back to the dataframes
        rolling_mean_df['period_end'] = period_end
        rolling_mean_df.set_index('period_end', inplace = True)
        rolling_mean_df.index = pd.to_datetime(rolling_mean_df.index)

        return rolling_mean_df
    
    @staticmethod
    # Function to store all the results statistics
    def prediction_performance(predictions):
        # Calculate R square (prevent division by zero if using benchmark model)
        try:
            R_sq = 1 - (sum((predictions['true_y'] - predictions['predictions'])**2)/sum((predictions['true_y'] - predictions['benchmark'])**2))
        except:
            R_sq = 'R_sq for benchmark Model not valid'
        # Calculate RMSE with topic score
        RMSE_with_top_score = np.sqrt(np.mean((predictions['true_y']-predictions['predictions'])**2))
        # Calculate RMSE without topic score
        RMSE_without_top_score = np.sqrt(np.mean((predictions['true_y']-predictions['benchmark'])**2))
        # Calculate MAE with topic score
        MAE_with_top_score = np.mean(np.abs(predictions['true_y']-predictions['predictions']))
        # Calculate MAE without topic score
        MAE_without_top_score = np.mean(np.abs(predictions['true_y']-predictions['benchmark']))
        d = {}
        d['R_sq'] = R_sq
        d['RMSE_with_topic_score'] = RMSE_with_top_score
        d['RMSE_without_topic_score'] = RMSE_without_top_score
        d['MAE_with_topic_score'] = MAE_with_top_score
        d['MAE_without_topic_score'] = MAE_without_top_score
        return d
        
    @staticmethod 
    def model_stat_plot(model, title, show = True):
        fig, axs = plt.subplots(1,2,figsize = (15,5))
        model.drop(columns = ['period_start','period_end','durbinwatson','kurtosis']).plot(kind = 'box', ax = axs[0])
        axs[0].axhline(0, linestyle = '--', color = 'red')
        axs[0].set_ylim(-1,1)
        model[['durbinwatson','kurtosis']].plot(kind = 'box', ax = axs[1])
        axs[1].axhline(2, linestyle = '--', color = 'red')
        axs[1].axhline(3, linestyle = '--', color = 'red')
        plt.suptitle(str(title))
        if show == False:
            plt.close()
        else:
            plt.show()
        return fig
        
    @staticmethod 
    def timeseries_plot_for_results(analysis, show_HSI=False, show_return = False, show_volume = False):
 
        n_start_date = 0
        n_end_date = -1
        n_start_date1 = '2016-02-24'
        n_end_date1 = '2020-04-28'
        n_width=1000
        n_height=250
        n_margin=dict(l=0, r=0, b=0, t=30)


        fig = px.line(analysis, x=analysis.index, y=analysis['rsquare'])
        fig.update_layout(  width=n_width, height=n_height, margin=n_margin)
        fig.show()

        
        fig = go.Figure()
        fig.update_layout(title="Chosen_RMSE",  width=n_width, height=n_height,margin=n_margin,
                          yaxis=dict(title="RMSE"),legend=dict(orientation="h"))
        fig.add_trace(go.Scatter(x=analysis.index,
                                 y=analysis['chosen_RMSE'],
                                 mode='lines'))

        # fig.update_yaxes(range=[0, 0.9])
        fig.show()


        fig = go.Figure()
        fig.update_layout(title="Chosen_MAE",  width=n_width, height=n_height,margin=n_margin,
                          yaxis=dict(title="MAE"),legend=dict(orientation="h"))
   
        fig.add_trace(go.Scatter(x=analysis.index,
                                 y=analysis['chosen_MAE'],
                                 mode='lines'))

        # fig.update_yaxes(range=[0, 0.9])
        fig.show()

        if show_HSI == True:
            fig = px.line(HSI_stat_df, x=HSI_stat_df.index, y=HSI_stat_df['Adj Close'], range_x=[n_start_date1 ,n_end_date1])
            fig.update_layout(title="HSI mean closing price",  width=n_width, height=n_height, margin=n_margin)
            fig.show()

        if show_return == True:
            fig = px.line(HSI_stat_df, x=HSI_stat_df.index, y=HSI_stat_df['log_return'], range_x=[n_start_date1 ,n_end_date1])
            fig.update_layout(title="HSI mean return",width=n_width, height=n_height,margin=n_margin)
            fig.show()

        if show_volume == True:
            fig = px.line(HSI_stat_df, x=HSI_stat_df.index, y=HSI_stat_df['Volume'], range_x=[n_start_date1 ,n_end_date1])
            fig.update_layout(title="Volume",width=n_width, height=n_height,margin=n_margin)
            #fig.show()


  
    @staticmethod
    def Input_data_Preparation(transform, pre_processed_dfs, Y = None):

        """
        This is to make transformation for the dataframes to fit in the regression
        transform = standard, None, logit
        pre_processed_dfs = the list of dataframes to be transformed
        Y = the dependant variable to join to the dataframe
        """

        scaler = StandardScaler()
        input_dataframes = []
        for df in pre_processed_dfs:
            if transform == 'standard':
                processed_df = pd.DataFrame(scaler.fit_transform(df), index = df.index, columns= df.columns)
            elif transform == 'None':
                processed_df = df
            elif transform == 'logit':
                processed_df = np.log(df/(1-df))
                if processed_df.isna().sum().sum() > 0:
                    processed_df = np.log(df*0.001/(1 - df*0.001))
            
            if Y != None:
                processed_df = processed_df.join(Y)
            input_dataframes.append(processed_df)

        return input_dataframes
    
    def fit_with_Parameter_Tuning(self, window_size, Y, X_list, parameters_dict, p = 0, rolling_mean_y=False, 
                                  tolerance = 'high' , X_inrange_rule = True, Y_inrange_rule = False, check_inf_obs = True):
        """
        This is to tune the model parameters in class Regression.
        Parameters that are able to tune:
        1. decision_threshold
        2. n
        3. interaction_terms
        """      
        # Create a list to store the result
        performance = []
        
        # Loop through the listed values and store the RMSE as the choosing criteria
        for threshold in parameters_dict['decision_threshold']:
            for n in parameters_dict['n']:
                for interaction_terms in parameters_dict['interaction_terms']:
                    model = self.fit(window_size, Y, X_list, threshold , n, interaction_terms, p, 
                                     rolling_mean_y, tolerance, X_inrange_rule, Y_inrange_rule, check_inf_obs)
                    overall_RMSE = model[2]['acc_RMSE'][-1]
                    performance.append((overall_RMSE, threshold, n, interaction_terms, model))
        
        # Choose the one with best performance
        best_performance = sorted(performance, key=lambda x: x[0])[0]
        best_RMSE = best_performance[0]
        best_threshold = best_performance[1]
        best_n = best_performance[2]
        best_interaction_terms = best_performance[3]
        best_model = best_performance[4]

        print("Best model with these predictors are RMSE = {}".format(best_RMSE))
        print("threshold = {}, n = {}, interaction terms = {}, rolling_mean_y = {}, X_inrange_rule = {}, Y_inrange_rule = {}, check_inf_obs = {}".\
              format(best_threshold, best_n, best_interaction_terms, rolling_mean_y, X_inrange_rule, Y_inrange_rule, check_inf_obs))
        
        return best_model
        
    def fit_with_Experiments(self, Xs, Ys, model_names, params_dicts, window_size, p, rolling_mean_y, rolling_size,
                             tolerance, X_inrange_rule, Y_inrange_rule, check_inf_obs):
        """
        Xs = List of list of predictors
        Ys = List of response
        params_dicts = list of dictionary of parameters to tune
        window_size = list of number of days each rolling window regression spans
        model_names = list of names of each model to compare
        p = list of numbers to denote the AR(p) process involved in the model
        rolling_mean_y = list of booleans to denote whether rolling mean of y is included in the regression
        rolling_size = list of rolling size to show for the OOS R_sq plots
        """
        
        cwd = os.getcwd()
        all_tested_models = {}
        
        for window in window_size:
            # Create a dictionary for comparison plots
            all_tested_models[window] = []
            tested_models = {}
            tested_Y = ['']
            Y_counter_list = ['']
            Y_counter = 0
            for X, Y, name, params, lag, roll_mean, tolerance, X_inrange_rule, Y_inrange_rule, check in zip(Xs, Ys, model_names, params_dicts, p, rolling_mean_y, tolerances, X_inrange_rules, Y_inrange_rules, check_inf_obs):
                model = self.fit_with_Parameter_Tuning(window, Y, X, params, lag, roll_mean, tolerance, X_inrange_rule, Y_inrange_rule, check)
                if roll_mean == True:
                    benchmark_model = self.fit(window, Y, [], 1, 1, rolling_mean_y = True)
                else:
                    benchmark_model = self.fit(window, Y, [], 1, 1)
                # Save different Y in different folders
                times_not_matched = 0
                for testedy, ycounter in zip(tested_Y, Y_counter_list):
                    if Y.equals(testedy):
                        # Use existing path name if the Y exists before
                        tested_Y.append(Y)
                        Y_counter_list.append(ycounter)
                        full_path = cwd + '/Y' + str(ycounter)
                        tested_models[ycounter].append((model, name))
                        print(ycounter, window, name)
                        break
                    else:
                        times_not_matched += 1
                        if times_not_matched >= len(tested_Y):
                            Y_counter += 1
                            tested_Y.append(Y)
                            Y_counter_list.append(Y_counter)
                            # Set new path name to store the file
                            full_path = cwd + '/Y' + str(Y_counter)
                            try:
                                os.makedirs(full_path)
                            except FileExistsError:
                                print('Directory exists! Writing to the directory.')
                            tested_models[Y_counter] = [(model, name)]
                            print('NewY!', Y_counter, window, name)
                            break
                # Set unique names for each model and save the result to the path
                print(full_path)
                all_tested_models[window].append({name: model})
                model_name = 'w' + str(window) + name
                path_name = full_path + '/{}.pdf'.format(model_name)
                Regression.report_graphs(model, benchmark_model, rolling_size, path_name)
                print('------------------------------------------------------------------')
                
            # Do comparison within each window_size
            print('number of models to be compared: ',tested_models.keys())
            for ix in list(tested_models.keys()):
                list_of_models = [x[0] for x in tested_models[ix]]
                list_of_model_names = [x[1] for x in tested_models[ix]]
                fig, axs = plt.subplots(2,1,figsize = (20,10))
                for m, label in zip(list_of_models, list_of_model_names):
                    m[0].set_index('period_end').rsquare.plot(ax = axs[0], label = label)
                    m[2].acc_RMSE.plot(ax=axs[1], label = label)
                axs[0].set_title('In sample R2 plot')
                axs[0].legend()
                axs[1].set_title('Overall RMSE Plot')
                benchmark_model[2].acc_RMSE.plot(ax = axs[1], label = 'benchmark')
                legend_names = list_of_model_names.extend(['benchmark'])
                axs[1].legend()
            plt.title('window size = '+str(window))
            plt.show()
            

        return all_tested_models

    @staticmethod           
    def report_graphs(result, benchmark_model, rolling_size, path):
        """
        rolling_size = list of rolling size to show for the OOS R_sq plot
        results = list of model results
        """
        with PdfPages(path) as pdf:
            # Bar chart to show the proportion of topic score vs ybar
            fig, ax = plt.subplots()
            result[0].model_name.value_counts().plot(kind='bar', ax = ax, title = 'Number of times choosing Topic Score and ybar')
            totals = []
            for i in ax.patches:
                totals.append(i.get_height())
            total = sum(totals)
            for i in ax.patches:
                ax.text(i.get_x()-.03, i.get_height()+.5, \
                        str(round((i.get_height()/total)*100, 2))+'%', fontsize=12,
                            color='dimgrey')
            pdf.savefig()
            plt.close()

            # Model statistics plot
            fig = Regression.model_stat_plot(result[0], 'Model Statistics', show=False)
            pdf.savefig(fig)
            plt.close(fig)

            # Parameter time series plot
            fig, ax = plt.subplots(figsize = (20,7))
            rolling_analysis.heatmap(np.abs(result[1].T), ax = ax)
            plt.xticks([])
            plt.title('Parameter time series plot')
            pdf.savefig()
            plt.close()

            # In-sample performance
            fig, ax = plt.subplots(figsize = (20,7))
            result[0].set_index('period_end').rsquare.plot(ax = ax, title = 'Model R square')
            pdf.savefig()
            plt.close()

            # Overall RMSE plot performance
            fig, ax = plt.subplots(1,1,figsize = (20,7))
            benchmark_model[2].acc_RMSE.plot(ax = ax, label = True, title = 'Overall RMSE')
            result[2].acc_RMSE.plot(ax=ax, legend = True)
            ax.legend(["benchmark", "model"])
            pdf.savefig()
            plt.close()
            
            # Error_per_day_comparison plot
            fig = Regression.error_per_day_comparison_report(result[2], show = False)[0]
            pdf.savefig(fig)
            plt.close(fig)
            
            # Rolling window OOS R_sq plot
            for rs in rolling_size:
                fig = Regression.Rolling_OOS_R2_plot(result, rs, show = False)
                pdf.savefig(fig)
                plt.close(fig)
               
    @staticmethod
    def Rolling_OOS_R2_plot(result, rs, show = True):
        roll_df = rolling_analysis.rolling_OOS_R2(result[2], str(list(result[2].index)[0])[:10], str(list(result[2].index)[-1])[:10], rs, 1)
        fig_name = 'OOS R_sq plot - '+str(rs)+'days rolling window'
        fig, ax = plt.subplots(1,1,figsize = (20,5))
        roll_df.OOS_R_sq.plot(ax = ax)
        ax.set_title(fig_name)
        plt.axhline(color='red')
        plt.ylim(-10,5)
        if show == False:
            plt.close()
        else:
            plt.show()
        return fig
    
    @staticmethod
    def add_sq_error_per_day(predictions):
        input_df = predictions.copy()
        input_df['sq_error_diff'] = ((input_df['true_y'] - input_df['benchmark'])**2) - ((input_df['true_y'] - input_df['predictions'])**2)
        input_df['abs_error_diff'] = np.abs(input_df['true_y'] - input_df['benchmark']) - np.abs(input_df['true_y'] - input_df['predictions'])
        return input_df

    @staticmethod
    def error_per_day_comparison_report(predictions, show = True):
        input_df = Regression.add_sq_error_per_day(predictions)
        # Print number of times better in sq difference and abs difference respectively
        total_days, times_topic_model_sq, times_better_sq, proportion_better_sq = Regression.Conditional_Probability_Calculation(input_df, 'sq_error_diff')
        total_days, times_topic_model_abs, times_better_abs, proportion_better_abs = Regression.Conditional_Probability_Calculation(input_df, 'abs_error_diff')        
        
        # Simply print one out for brief reference
        result_sentence = 'Number of times the model is better than the benchmark model over {} days: '.format(total_days) + str(times_better_abs) + \
                            '/' + str(times_topic_model_abs) + ' = ' + str(round(proportion_better_abs, 4))

        # Plot graph
        fig, ax = plt.subplots(1,1,figsize = (20,7))
        input_df['abs_error_diff'].plot(ax = ax)
        ax.axhline(y = 0, color = 'red', linestyle = '--')
        ax.set_ylim(-10,10)
        ax.set_title(result_sentence)
        if show == False:
            plt.close()
        else:
            print(result_sentence)
            plt.show()
        
        times_better = (times_better_sq, times_better_abs)
        times_topic_model = (times_topic_model_sq, times_topic_model_abs)
        proportion_better = (proportion_better_sq, proportion_better_abs)
        
        return fig, times_better, times_topic_model, proportion_better, total_days
    
    @staticmethod
    def Conditional_Probability_Calculation(input_df, criteria):
        total_days = len(input_df)
        times_topic_model = len(input_df[input_df[criteria] != 0])
        times_better = len(input_df[input_df[criteria] > 0])
        if times_topic_model == 0:
            proportion_better = 0
        else:
            proportion_better = times_better/times_topic_model
        return total_days, times_topic_model, times_better, proportion_better
    
    @staticmethod
    def Threshold_Prediction_Impact(input_result, threshold):
        # Combine In-sample and Out-sample dataframe for convenience
        combined_df = pd.concat([input_result[0], input_result[2].reset_index().rename(columns = {'index':'prediction_date'})], axis = 1)
        # Compute the prediction using the hybrid approach with the threshold set
        combined_df['Hybrid_Prediction'] = (combined_df['predictions']*(combined_df['rsquare'] > threshold)) + (combined_df['benchmark']*(combined_df['rsquare'] <= threshold))
        # Form new prediction dataframe
        new_pred_df = pd.DataFrame(index = combined_df['prediction_date'])
        new_pred_df['predictions'] = combined_df['Hybrid_Prediction'].values
        new_pred_df['benchmark'] = combined_df['benchmark'].values
        new_pred_df['true_y'] = combined_df['true_y'].values
        # Get the statistics
        RMSE_with_topic_score = Regression.prediction_performance(new_pred_df)['RMSE_with_topic_score']
        RMSE_without_topic_score = Regression.prediction_performance(new_pred_df)['RMSE_without_topic_score']
        MAE_with_topic_score = Regression.prediction_performance(new_pred_df)['MAE_with_topic_score']
        MAE_without_topic_score = Regression.prediction_performance(new_pred_df)['MAE_without_topic_score']
        # Refine the statistics
        RMSE_ratio = RMSE_without_topic_score/RMSE_with_topic_score
        MAE_ratio = MAE_without_topic_score/MAE_with_topic_score
        total_days = Regression.error_per_day_comparison_report(new_pred_df, show=False)[-1]
        Cond_Prob = Regression.error_per_day_comparison_report(new_pred_df, show=False)[-2]
        times_topic_model = Regression.error_per_day_comparison_report(new_pred_df, show=False)[-3]
        output_dfs = (combined_df, new_pred_df)
        # Use the output_dfs to compute the Conditional RMSE and the Conditional MAE
        error_df = Regression.add_sq_error_per_day(output_dfs[1])
        days_top_score_sq = error_df[error_df['sq_error_diff'] != 0]
        days_top_score_abs = error_df[error_df['abs_error_diff'] != 0]
        cond_criteria_sq = Regression.prediction_performance(days_top_score_sq.drop(columns = ['sq_error_diff','abs_error_diff']))
        cond_criteria_abs = Regression.prediction_performance(days_top_score_abs.drop(columns = ['sq_error_diff','abs_error_diff']))
        Cond_RMSE_ratio = cond_criteria_sq['RMSE_without_topic_score']/cond_criteria_sq['RMSE_with_topic_score']
        Cond_MAE_ratio = cond_criteria_abs['MAE_without_topic_score']/cond_criteria_abs['MAE_with_topic_score'] 
        Cond_RMSE = cond_criteria_sq['RMSE_with_topic_score']
        Cond_MAE = cond_criteria_abs['MAE_with_topic_score']
        return RMSE_ratio, MAE_ratio, Cond_Prob[0], Cond_Prob[1], Cond_RMSE_ratio, Cond_MAE_ratio, Cond_RMSE, Cond_MAE, total_days, times_topic_model[0], times_topic_model[1], output_dfs

    @staticmethod
    def Dynamic_Criteria_plot(input_result, start_threshold, end_threshold, ax, step = 0.01, show = True, model_name = '', ax2 = None):
        # Get the list of criteria with the changing threshold
        criteria_list = []
        threshold = start_threshold
        while threshold <= end_threshold:
            all_criteria = Regression.Threshold_Prediction_Impact(input_result, threshold)
            criteria_list.append((all_criteria[:-1], threshold))
            threshold += step
        # Plot the graph to show the criteria
        columns = ['RMSE_ratio','MAE_ratio','Cond_Prob_SE', 'Cond_Prob_AE', 'Cond_RMSE_ratio',
                   'Cond_MAE_ratio','Cond_RMSE', 'Cond_MAE', 'Total_Days','Times_Topic_Model_SE','Times_Topic_Model_AE']
        criteria_df = pd.DataFrame([x[0] for x in criteria_list], index = [x[1] for x in criteria_list], columns = columns)
        ax.set_prop_cycle(color = ['purple','blue','green'])
        ax.plot(criteria_df.index, criteria_df[['Cond_Prob_AE', 'Cond_RMSE_ratio', 'Cond_MAE_ratio']], lw = 0.7)
        ax.legend(['Cond_Prob', 'Cond_RMSE_ratio', 'Cond_MAE_ratio'], loc = 'lower left', fontsize = 6)
        ax.axhline(0.5, color = 'grey', ls = '--', alpha = 0.7, lw = 0.2)
        ax.axhline(1, color = 'grey', ls = '--', alpha = 0.7, lw = 0.2)
        thres_under_10_obs = list(criteria_df[criteria_df['Times_Topic_Model_AE'] <= 10].index)
        if thres_under_10_obs:
            thres_under_10_obs = thres_under_10_obs[0]
        else:
            thres_under_10_obs = end_threshold
        thres_under_20_obs = list(criteria_df[criteria_df['Times_Topic_Model_AE'] <= 20].index)
        if thres_under_20_obs:
            thres_under_20_obs = thres_under_20_obs[0]
        else:
            thres_under_20_obs = end_threshold
        thres_under_25_obs = list(criteria_df[criteria_df['Times_Topic_Model_AE'] <= 25].index)
        if thres_under_25_obs:
            thres_under_25_obs = thres_under_25_obs[0]
        else:
            thres_under_25_obs = end_threshold
        thres_under_30_obs = list(criteria_df[criteria_df['Times_Topic_Model_AE'] <= 30].index)
        if thres_under_30_obs:
            thres_under_30_obs = thres_under_30_obs[0]
        else:
            thres_under_30_obs = end_threshold
        ax.axvspan(thres_under_30_obs, thres_under_20_obs,  alpha = 0.2)
        # ax.axvspan(thres_under_25_obs, thres_under_20_obs,  alpha = 0.25)
        ax.axvspan(thres_under_20_obs, thres_under_10_obs,  alpha = 0.4)
        ax.axvspan(thres_under_10_obs, end_threshold, alpha = 0.6)
        ax.set_ylim(0,1.5)
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
        if ax2 != None:
            # Add another y axis for plotting the Conditional RMSE and MAE
            ax2.set_prop_cycle(color = ['darkred','lightcoral'])
            ax2.plot(criteria_df.index, criteria_df[['Cond_RMSE', 'Cond_MAE']], lw = 0.7)
            ax2.legend(['Cond_RMSE','Cond_MAE'], loc = 'lower right', fontsize = 6)
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.yaxis.set_tick_params(labelsize=8)
            ax2.set_xticks([])
            ax2.set_ylim(0,2.5)
            ax2.set_title(model_name, fontsize = 8)
            ax2.axvspan(thres_under_30_obs, thres_under_20_obs,  alpha = 0.2)
            # ax2.axvspan(thres_under_25_obs, thres_under_20_obs,  alpha = 0.3)
            ax2.axvspan(thres_under_20_obs, thres_under_10_obs,  alpha = 0.4)
            ax2.axvspan(thres_under_10_obs, end_threshold, alpha = 0.6)
        else:
            ax.set_title(model_name, fontsize = 8)
            
        return criteria_df

    @staticmethod
    def Model_Prediction_Comparison_Report(result, start_threshold, end_threshold, step, show = True):
        # Set saving directory for the report
        cwd = os.getcwd()
        full_path = cwd + '/Model_Prediction_Comparison_Report/'
        try:
            os.makedirs(full_path)
            print('Created New Directory at this working folder (named Model_Prediction_Comparison_Report)')
        except FileExistsError:
            print('Directory exists! Writing to the directory.')
        file_name = full_path + '{}.pdf'.format(str(datetime.now()))
        # Create a dictionary to store all the dataframes
        D = {}
        # Run the result
        with PdfPages(file_name) as pdf:
            window_size = list(result.keys())
            for size in window_size:
                D[size] = {}
                number_of_models = len(result[size])
                fig, axs = plt.subplots(2, number_of_models, figsize = (20,5), gridspec_kw={'height_ratios': [2, 3]})
                for ix in range(number_of_models):
                    name = list(result[size][ix].keys())[0]
                    input_result = result[size][ix][name]
                    criteria_df = Regression.Dynamic_Criteria_plot(input_result, start_threshold, end_threshold, axs[1,ix], step, show = False, model_name = name, ax2=axs[0,ix])
                    # criteria_df = Regression.Dynamic_Criteria_plot(input_result, start_threshold, end_threshold, axs[1], step, show = False, model_name = name, ax2=axs[0])
                    D[size][name] = criteria_df
                suptitle = 'Models in window size {}'.format(size)
                plt.suptitle(suptitle, y = 1, fontsize = 8)
                fig.tight_layout(pad = 0.5)
                if show == False:
                    plt.close()
                else:
                    plt.show()
                pdf.savefig(fig)
        return D