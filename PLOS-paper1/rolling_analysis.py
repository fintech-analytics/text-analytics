class rolling_analysis:
    
    def __init__(self, df, std_resid, start, end, window_size, roll_size, threshold = 0.5, lag = 5):
        
        """
        df --> topics generated from rolling_LDA
        std_resid --> a dataframe of residual
        start --> start date of the period to analyse
        end --> end date of the period to analyse
        window_size --> size of the rolling window
        roll_size --> step size of each move in the rolling analysis
        threshold --> Show lines exceeding the threshold only in linked_top_line_plot()
        """  
        
        self.df = df
        self.std_resid = std_resid
        self.start = start
        self.end = end
        self.window_size = window_size
        self.roll_size = roll_size
        self.threshold = threshold
        self.lag = lag + 1
        self.end_index = 0 # To be declared
            
  
    def Multiple_Correlations(self):
        """
        top_by_period --> topics generated from rolling_LDA
        resid --> dataframe of the residual
        start --> start date of the analysis
        end --> end date of the analysis
        window_size --> size of the rolling window
        rs --> step size or roll size of the rolling window
        -------------------------------------------------------
        Return: topic multiple correlation, lag multiple correlation, a list of rolled windows
        """

        # Set all attributes
        start = self.start
        end = self.end
        rs = self.roll_size
        top_by_period = self.df
        resid = self.std_resid
        window_size = self.window_size
        top_by_period.columns = [float(x) for x in list(top_by_period.columns)]

        # Set the df index to datetime
        top_by_period.index = pd.to_datetime(top_by_period.index)

        # Create a dataframe to store all the lagged_topics
        full_df = pd.DataFrame()

        for topic in top_by_period.columns:
            topic_df = top_by_period[[topic]]
            # Create shift 1 to lag in the dataframe of each topic
            for lag in range(1,self.lag):
                col_name = str(topic) + "t-" + str(lag)
                topic_df[col_name] = topic_df.iloc[:,0].shift(lag)
            full_df = pd.concat([full_df,topic_df], axis=1)

        # Attach the residual column to the full dataframe
        full_df = full_df.join(resid).fillna(method='ffill').dropna()

        # Create a dictionary to store the mult_corr
        top_mult_corr = {}
        lag_mult_corr = {}

        # Create timeframe to track the period
        timeframes = []

        # Roll the window
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
        roll_start = start_date + timedelta(days=0)
        roll_end = roll_start + timedelta(days=window_size)
        
        while roll_end <= end_date:
            window = full_df[(full_df.index >= roll_start) & (full_df.index < roll_end)]
            
            # Track the index
            index = list(window.index)[-1]
            if index == self.end_index:
                roll_start = roll_start + timedelta(days = rs)
                roll_end = roll_end + timedelta(days=rs)
                continue
            else:
                self.end_index = index
                pass

            # Print current work
            timeframe = str(roll_start) + " to " + str(roll_end)
            timeframes.append(timeframe)
            # print(timeframe)

            # Find multiple correlation for the topics
            for topic in top_by_period.columns:
                re_pattern = '^' + str(topic)
                topic_cols = list(set([x for x in list(window.columns) if re.findall(re_pattern, str(x))]) - set([topic]))
                X = window[topic_cols]
                Y = window.iloc[:,-1]
                fit = LinearRegression().fit(X,Y)
                R_sq = fit.score(X,Y)
                mult_corr = np.sqrt(R_sq)
                # Add the value to the dictionary
                if topic not in top_mult_corr.keys():
                    top_mult_corr[topic] = [mult_corr]
                else:
                    top_mult_corr[topic].append(mult_corr)

            # Find multiple correlation for the lags
            for lag in range(self.lag):
                if lag == 0:
                    re_pattern = '\d+\.0$'
                else:
                    re_pattern = '.+' + 't-' + str(lag) + '$'
                topic_cols = [x for x in list(window.columns) if re.findall(re_pattern, str(x))]
                X = window[topic_cols]
                Y = window.iloc[:,-1]
                fit = LinearRegression().fit(X,Y)
                R_sq = fit.score(X,Y)
                mult_corr = np.sqrt(R_sq)
                # Add the value to the dictionary
                if lag not in lag_mult_corr.keys():
                    lag_mult_corr[lag] = [mult_corr]
                else:
                    lag_mult_corr[lag].append(mult_corr)

            # Move the window
            roll_start = roll_start + timedelta(days = rs)
            roll_end = roll_end + timedelta(days=rs)

        return full_df, top_mult_corr, lag_mult_corr, timeframes


    def Single_Correlation(self):
        start_date = datetime.strptime(self.start, '%Y-%m-%d')
        end_date = datetime.strptime(self.end, '%Y-%m-%d')
        roll_start = start_date + timedelta(days=0)
        roll_end = roll_start + timedelta(days=self.window_size)
        rs = self.roll_size

        # Set the df index to datetime
        self.df.index = pd.to_datetime(self.df.index)

        # Create a dataframe to store all correlations
        corr_df = pd.DataFrame()
        timeframes = []
        
        # Roll the window to show graphs and correlations
        period = 0
        while roll_end <= end_date:
            window = self.df[(self.df.index >= roll_start) & (self.df.index < roll_end)]
            
            # Track the index
            index = list(window.index)[-1]
            if index == self.end_index:
                roll_start = roll_start + timedelta(days = rs)
                roll_end = roll_end + timedelta(days=rs)
                continue
            else:
                self.end_index = index
                pass

            # Print current work
            timeframe = str(roll_start) + " to " + str(roll_end)
            print("Period", timeframe)
            timeframes.append(timeframe)

            # Draw the graph and return the correlation
            
            corr = self.linked_top_line_plot(self.std_resid, window, self.threshold, lag= self.lag)
            new_index = ['P{}_'.format(str(period)) + str(int(x)) for x in list(corr.index)]
            corr.index = new_index
            corr_df = pd.concat([corr_df, corr])
            
            # Move the window
            roll_start = roll_start + timedelta(days = rs)
            roll_end = roll_end + timedelta(days=rs)
            period += 1
            
        return corr_df, timeframes

    @staticmethod
    def mult_corr_graph(mult_corr_df, timeframes):
    
        mult_corr = pd.DataFrame.from_dict(mult_corr_df)

        new_list = [ls[0:10] for ls in timeframes]

        mult_corr.index = new_list

        plt.style.use('seaborn-darkgrid')
        plt.figure(figsize=(50,200))
        palette = plt.get_cmap('tab20')

        num=0

        for column in mult_corr:
            num+=1

            plt.subplot(mult_corr.shape[1],1, num)

            for v in mult_corr:
                mult_corr[v].plot(marker='', color='grey', linewidth=0.3, alpha=0.3)

            mult_corr[column].plot(marker='', color=palette(num), linewidth=3, alpha=1, label=column)

            #plt.xlim(0,54)
            #plt.ylim(0.45,1)

            plt.tick_params(axis='x', labelsize=20, rotation=70)
            plt.tick_params(axis='y', labelsize=20)

            plt.title(column, loc='left', fontsize=20, fontweight=10, color=palette(num))

    @staticmethod
    def linked_top_line_plot(std_resid, linked_topics, threshold, lag):
        lag = lag+1
        top_vol_df = linked_topics.join(std_resid).fillna(method = 'ffill').dropna()

        corr_df = pd.DataFrame()
        pass_topics = []

        grab_columns = top_vol_df

        # Create dataframe to store the correlations of lag_date

        lag_df = pd.DataFrame()

        for lag in range(0,lag):
            lag_resid = grab_columns[list(std_resid.columns)[0]].shift(-lag)
            lag_df_plot = grab_columns.iloc[:,:-1].join(lag_resid)
            lag_df_plot.dropna(inplace = True)

            # Calculate the correlation
            corr = lag_df_plot.corr().iloc[-1,:-1]
            lag_df[lag] = corr

        # extract the columns which satisfy the threshold for line chart
        pass_df = lag_df[(lag_df >= threshold) | (lag_df <= -threshold)]
        all_lag = list(pass_df[pass_df.sum(axis = 1) != 0].index)
        pass_topics.extend(all_lag)

        print(pass_df.dropna(how='all').dropna(axis=1,how='all'))

        corr_df = pd.concat([corr_df,lag_df])

        # Standardise the data for better comparison in the plot
        grab_rows = grab_columns.loc[grab_columns[grab_columns.columns[0]] != 0]
        grab_rows.dropna(inplace = True)
        grab_rows = (grab_rows - grab_rows.mean())/ grab_rows.std()

        # Plot the graphs
        try:
            fig, ax = plt.subplots(1,1, figsize=(20,5))
            topics_by_period_plot = grab_rows[list(pd.Series(all_lag).unique())]
            std_resid_plot = grab_rows.iloc[:,-1]

            topics_by_period_plot.plot(ax= ax, lw=0.3, alpha = 0.5)
            std_resid_plot.plot(ax= ax, lw= 1, legend = True, color = 'blue')
        except:
            pass

        #plt.savefig("Graph" + str(i) +".png", format="PNG")
        plt.title(str(list(linked_topics.index)[0]) + ' To ' + str(list(linked_topics.index)[-1]))
        plt.show()
        return corr_df

    @staticmethod
    def heatmap(corr_df, linewidth=0, ax = None):
        cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
        sns.heatmap(corr_df, annot = False, vmin=-1, vmax=1, center= 0, 
                    linewidths=linewidth, linecolor='grey', cmap = cmap, ax = ax, cbar = False)
        

    @staticmethod
    def rolling_statistics(HSI_df, start, end, window_size, roll_size):
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
        roll_start = start_date + timedelta(days=0)
        roll_end = roll_start + timedelta(days=window_size)
        rs = roll_size

        # Set the df index to datetime
        HSI_df.index = pd.to_datetime(HSI_df.index)

        # Create storing objects
        rolling_mean_df = pd.DataFrame()
        rolling_var_df = pd.DataFrame()
        period_start = []
        period_end = []

        end_index = 0

        # Roll the window
        period = 0
        while roll_end <= end_date:
            window = HSI_df[(HSI_df.index >= roll_start) & (HSI_df.index < roll_end)]

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
            period_end.append(periodend)

            # Calculate the rolling statistics - mean
            rolling_mean = window.mean().to_frame().T
            rolling_mean.index = [period]
            rolling_mean_df = pd.concat([rolling_mean_df,rolling_mean])

            # Calculate the rolling statistics - variance
            rolling_var = window.var().to_frame().T
            rolling_var.index = [period]
            rolling_var_df = pd.concat([rolling_var_df, rolling_var])

            # Move the window
            roll_start = roll_start + timedelta(days = rs)
            roll_end = roll_end + timedelta(days=rs)
            period += 1

        # Append period starts and ends back to the dataframes
        rolling_mean_df['period_start'] = period_start
        rolling_mean_df['period_end'] = period_end
        rolling_var_df['period_start'] = period_start
        rolling_var_df['period_end'] = period_end

        return rolling_mean_df, rolling_var_df
    
    @staticmethod
    def rolling_OOS_R2(predictions, start, end, window_size, roll_size):
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
        roll_start = start_date + timedelta(days=0)
        roll_end = roll_start + timedelta(days=window_size)
        rs = roll_size

        # Set the df index to datetime
        predictions.index = pd.to_datetime(predictions.index)

        # Create storing objects
        R_sqs = []
        period_start = []
        period_end = []

        end_index = 0

        # Roll the window
        period = 0
        while roll_end <= end_date:
            window = predictions[(predictions.index >= roll_start) & (predictions.index < roll_end)]

            # Track the index
            if list(window.index) == []:
                roll_start = roll_start + timedelta(days = rs)
                roll_end = roll_end + timedelta(days=rs)
                continue
            else: 
                index = list(window.index)[-1]
            if index == end_index:
                roll_start = roll_start + timedelta(days = rs)
                roll_end = roll_end + timedelta(days=rs)
                continue
            else:
                end_index = index
                pass
                
            # Print current work        
            period_start.append(window.index[0])  
            period_end.append(window.index[-1])

            # Calculate the rolling OOS R_sq
            R_sq = 1 - (sum((window['true_y'] - window['predictions'])**2)/sum((window['true_y'] - window['benchmark'])**2))
            R_sqs.append(R_sq)

            # Move the window
            roll_start = roll_start + timedelta(days = rs)
            roll_end = roll_end + timedelta(days=rs)
            period += 1

        output = pd.DataFrame({'OOS_R_sq':R_sqs,'period_start':period_start,'period_end':period_end})

        return output
    
    @staticmethod
    def ccf_plot(Y, topic_scores, p, K):

        # Settings
        number_of_columns = len(topic_scores.columns)
        fig_n_col = 5
        fig_n_row = int(np.ceil(number_of_columns/fig_n_col))
        fig, axs = plt.subplots(fig_n_row, fig_n_col, figsize = (20, int(fig_n_row*3)))

        for column in range(number_of_columns):
            # Create the AR model
            AR_model = ARIMA(topic_scores.iloc[:,column],(p,0,0)).fit(trend = 'nc')

            # Extract parammeter in the model with corresponding lags to calculate Y_tilde
            for ix, phi in enumerate(AR_model.params):
                lag = 'Y_t-{}'.format(ix+1)
                Y[lag] = Y.iloc[:,0].shift(ix+1)
                Y[lag] = Y[lag]*(-phi)
            Y_tilde = Y.dropna().sum(axis = 1).loc[topic_scores.index]

            # Calculate correlation between a_t-k and Y_tilde
            CORR_dict = {}
            for lag in range(-K,K):
                CORR = Y_tilde.corr(AR_model.resid.shift(lag))
                CORR_dict[lag] = CORR

            # Plot the correlation graph from k = -K to K 
            ax = axs[column//fig_n_col, column%fig_n_col]
            ax.bar(CORR_dict.keys(), CORR_dict.values())
            ax.set_ylim(-1,1)
            ax.axhline(0,color = 'red', linestyle = '--', lw = 0.3)
            ax.axvline(0,color = 'red', linestyle = '--', lw = 0.3)
            ax.set_title(topic_scores.columns[column])
        plt.show()
        
    @staticmethod
    def rolling_ccf_plot(Y, X, start, end, window_size, roll_size, p, K):
        
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
        roll_start = start_date + timedelta(days=0)
        roll_end = roll_start + timedelta(days=window_size)
        rs = roll_size

        # Set the df index to datetime
        Y.index = pd.to_datetime(Y.index)
        X.index = pd.to_datetime(X.index)

        end_index = 0

        # Roll the window
        while roll_end <= end_date:
            window = X[(X.index >= roll_start) & (X.index < roll_end)]

            # Track the index
            if list(window.index) == []:
                roll_start = roll_start + timedelta(days = rs)
                roll_end = roll_end + timedelta(days=rs)
                continue
            else: 
                index = list(window.index)[-1]
            if index == end_index:
                roll_start = roll_start + timedelta(days = rs)
                roll_end = roll_end + timedelta(days=rs)
                continue
            else:
                end_index = index
                pass
                
            # Print current work  
            print(window.index[0], ':', window.index[-1])
            
            # Do the plotting
            rolling_analysis.ccf_plot(Y, window, p, K)

            # Move the window
            roll_start = roll_start + timedelta(days = rs)
            roll_end = roll_end + timedelta(days=rs)