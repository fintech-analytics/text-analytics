import pickle

# Get the useful info 
def testing(selected, threshold_analysis):
    analysis = selected[-1][0][['rsquare','Hybrid_Prediction','true_y', 'benchmark', 'Partial_F']]
    analysis.index = selected[-1][0]['prediction_date']
    analysis['RMSE'] = ((analysis['true_y'] - analysis['Hybrid_Prediction'])**2)
    analysis.loc[analysis['rsquare'] > threshold_analysis, 'chose_topic?'] = 1
    analysis.loc[analysis['rsquare'] < threshold_analysis, 'chose_topic?'] = 0
    analysis['model_performance'] = abs(analysis['true_y'] - analysis['Hybrid_Prediction'])
    analysis['benchmark_performance'] = abs(analysis['true_y'] - analysis['benchmark'])
    return analysis

def chosen_day_parameter_heatmap(model_result, threshold, show = 'chosen'):
    result_at_threshold = Regression.Threshold_Prediction_Impact(model_result, threshold)[-1][0]
    abs_error = np.abs(result_at_threshold['true_y'] - result_at_threshold['Hybrid_Prediction'])
    bench_error = np.abs(result_at_threshold['true_y'] - result_at_threshold['benchmark'])
    result_at_threshold['good_days'] = bench_error - abs_error
    good_days = list(result_at_threshold[result_at_threshold['good_days'] > 0]['period_end'])
    bad_days = list(result_at_threshold[result_at_threshold['good_days'] < 0]['period_end'])
    other_days = set(list(model_result[1].index)) - set(good_days) - set(bad_days)

    good_days_df = np.abs(model_result[1].loc[good_days])
    bad_days_df = -np.abs(model_result[1].loc[bad_days])
    other_days_df = pd.DataFrame(columns = model_result[1].columns, index = other_days)
    good_days_df.index = good_days_df.index.date
    bad_days_df.index = bad_days_df.index.date
    try:
        other_days_df.index = other_days_df.index.date
    except:
        pass
    good_x_bad = pd.concat([good_days_df,bad_days_df], axis = 0).sort_index()
    good_x_bad_x_other = pd.concat([good_days_df,bad_days_df, other_days_df], axis = 0).sort_index()

    fig, axs = plt.subplots(1,1, figsize = (20,10))
    if show == 'all':
        rolling_analysis.heatmap(good_x_bad_x_other.T, 0)
    elif show == 'chosen':
        rolling_analysis.heatmap(good_x_bad.T, 0)
    plt.gca().yaxis.grid(True, linestyle='--')
    
    return good_x_bad_x_other

def histogram(Y1, chosen):
    fig, ax = plt.subplots(1,1, figsize = (20,10))
    avg = Y1.loc['2017-02-15':'2020-12-31']['abs_resid'].mean()
    mdn = Y1.loc['2017-02-15':'2020-12-31']['abs_resid'].median()
    plt.hist(Y1.loc['2017-02-15':'2020-12-31']['abs_resid'], bins=200)

    labelled_r = 0
    labelled_k = 0

    for ix, row in chosen.iterrows():
        if chosen.loc[ix, 'better']:
            color = 'r'
            label = 'Prediction - better than benchmark'
            labelled_r += 1
        else:
            color = 'k'
            label = 'Prediction - worse than benchmark'
            labelled_k += 1
        if labelled_r == 1:
            labelled_r += 1
            plt.axvline(x=chosen.loc[ix, 'Hybrid_Prediction'], color=color, linestyle='dotted', linewidth=0.8, label=label)
        elif labelled_k == 1:
            labelled_k += 1
            plt.axvline(x=chosen.loc[ix, 'Hybrid_Prediction'], color=color, linestyle='dotted', linewidth=0.8, label=label)
        else:
            plt.axvline(x=chosen.loc[ix, 'Hybrid_Prediction'], color=color, linestyle='dotted', linewidth=0.8)

    plt.axvline(x=avg, color='green', linestyle='dotted', linewidth=4, label="Mean - {}".format(str(round(avg, 4))))
    plt.axvline(x=mdn, color='purple', linestyle='dotted', linewidth=4, label="Median - {}".format(str(round(mdn, 4))))
    plt.title('Better and worse predictions distributions', fontsize = 16)
    plt.legend(fontsize=16)
    plt.xlabel('Volatility measure in the timeframe (log absolute standardized GARCH residual)', fontsize=16)
    plt.ylabel('Number of appearance', fontsize=16)
    plt.show()
    
def confusion_matrix_data_cal(true_y, pred_y, cutoff):
    
    if true_y >= cutoff and pred_y >= cutoff:
        idx = 'high_high'

    elif true_y < cutoff and pred_y < cutoff:
        idx = 'low_low'

    elif true_y > cutoff and pred_y < cutoff:
        idx = 'high_low'

    elif true_y < cutoff and pred_y > cutoff:
        idx = 'low_high'
    
    return idx
    
def classification_table(Y1, chosen, cutoff):
#     avg = Y1.loc['2017-02-15':'2020-12-31']['abs_resid'].mean()
#     mdn = Y1.loc['2017-02-15':'2020-12-31']['abs_resid'].median()
    
    #             | Y_true_high  | Y_true_low
    # -----------------------------------------
    # Y_pred_high |              |
    # -----------------------------------------
    # Y_pred_low  |              |
    # -----------------------------------------
    
    confusion_dict = {
        'high_high' : [0,0],
        'low_low' : [0,0],
        'high_low' : [0,0],
        'low_high' : [0,0],
    }
    
    for ix, row in chosen.iterrows():
        true_y = chosen.loc[ix, 'true_y']
        pred_y = chosen.loc[ix, 'Hybrid_Prediction']
        better = chosen.loc[ix, 'better']
        bench_pred_y = chosen.loc[ix, 'benchmark']
        
        confusion_dict[confusion_matrix_data_cal(true_y, pred_y, cutoff)][0] += 1
#         if better:
#             confusion_dict[confusion_matrix_data_cal(true_y, pred_y, cutoff)][1] += 1
        confusion_dict[confusion_matrix_data_cal(true_y, bench_pred_y, cutoff)][1] += 1
                
    # maintain the spaces for better presentation
    spaces_1 = 8 - len(str(confusion_dict['high_high']))
    spaces_2 = 8 - len(str(confusion_dict['low_high']))

    print(
        "            |     Y_pred_high      |      Y_pred_low  \n"
        "-------------------------------------------------------------\n"
        "Y_true_high |       {}{}       |      {}     \n"
        "--------------------------------------------------------------\n"
        "Y_true_low  |       {}{}       |      {}     \n"
        "------------------------------------------------------------\n".format(str(confusion_dict['high_high']), 
                                                                                   ' '*spaces_1,
                                                                                str(confusion_dict['high_low']),
                                                                                str(confusion_dict['low_high']),
                                                                                   ' '*spaces_2,
                                                                                str(confusion_dict['low_low'])))
    
    # Precision recall of the model
    precision_recall(confusion_dict['high_high'][0], confusion_dict['low_high'][0], confusion_dict['high_low'][0], 'Model results')
    #     precision_recall(high_high[1], low_high[1], high_low[1], 'Better results')
    
    # Precision recall of the benchmark
    precision_recall(confusion_dict['high_high'][1], confusion_dict['low_high'][1], confusion_dict['high_low'][1], 'Benchmark results')
    print(" ")
                
        
def precision_recall(tp, fp, fn, label):
    precision = tp/(tp + fp)
    recall = tp/(tp+fn)
    
    print(label)
    print('precision: ', precision, '; recall: ', recall)
    print("=============================================\n")
    
    
def gen_wordcloud(text, label, ax):
    wordcloud = WordCloud(width = 500, height= 500, max_font_size=60, max_words=20, background_color='white')
    wordcloud.generate_from_frequencies(text)
    ax.imshow(wordcloud,interpolation="bilinear")
    ax.axis("off")
    ax.set_title(label)
    
def run_wordcloud(selected, chosen):

    procedure = selected[-1][0].set_index('prediction_date')
    procedure.index = pd.to_datetime(procedure.index)
    good_days = procedure.loc[chosen.index]

    fig = plt.figure(figsize=(15,200))
    counter = 0
    n = np.ceil(good_days.shape[0]/2)

    for ix, row in good_days.iterrows():
        start = good_days.loc[ix, 'period_start']
        end = good_days.loc[ix, 'period_end']
        ax = fig.add_subplot(n,2,counter+1)
        counter += 1

        # generate the dtm and remove by countvectorizer like what the model does
        test_text_df = trading_days.loc[start:end]
        vectorizer = CountVectorizer(max_df=0.1, min_df=0.001)
        X = vectorizer.fit_transform(test_text_df['content'].to_list())
        columns = vectorizer.get_feature_names()
        tf_matrix = X.toarray()
        tf = pd.DataFrame(tf_matrix, columns=columns)
        tf.index = test_text_df.index
        wordcloud_input = dict(tf.sum())

        gen_wordcloud(wordcloud_input, str(start)[:10] + '-' + str(end)[:10], ax)

    plt.savefig('wordcloud_for_used_days.png', bbox_inches = 'tight', pad_inches = 0)

def read_result(result_file, pred_file, Y1_file, window_size, model_num, threshold=None, wordcloud=False):
    """
        window_size = 30,60,90
        model_num = 1, 2, 3
    """
    
    model_dict = ['model-1', 'model-2', 'model-3']
    model_num = model_num - 1
    model_name = model_dict[model_num]
    
    with open(result_file, 'rb') as f:
        result = pickle.load(f)
        
    with open(pred_file, 'rb') as f:
        Pred_Results = pickle.load(f)
        
    with open(Y1_file, 'rb') as f:
        Y1 = pickle.load(f)
    
    # Print the dataframe of the queried model
    accepted_df = Pred_Results[window_size][model_name][Pred_Results[window_size][model_name]['MAE_ratio'] > 1]
    print("_____________________ DF Accepted _____________________")
    print(accepted_df)
    print(' ')
    
    if not threshold:
        threshold = accepted_df.index.to_list()[0]
    
    selected = Regression.Threshold_Prediction_Impact(result[window_size][model_num][model_name] , threshold )
    
    print("_____________________ Test stat  _____________________")
    df_with_chose_topic = testing(selected, threshold)
    chosen = df_with_chose_topic[df_with_chose_topic['chose_topic?'] == 1]
    chosen[['model_performance', 'benchmark_performance']].boxplot()
    plt.show()
    print(stats.ttest_ind(chosen['benchmark_performance'], chosen['model_performance']))
    
    
    print("_____________________ Partial F Test  _____________________")
    fig, axs = plt.subplots(1,2, figsize=(20,10))
    chosen[['Partial_F']].boxplot(figsize=(10,10), ax=axs[0])
    axs[0].set_title("Distribution of the p_values from Partial F-Test in the chosen days", fontsize=10)
    axs[0].set_ylim(0,0.2)
    df_with_chose_topic[df_with_chose_topic['chose_topic?']==0][['Partial_F']].boxplot(figsize=(10,10), ax=axs[1])
    axs[1].set_title("Distribution of the p_values from Partial F-Test in the not chosen days", fontsize=10)
    axs[1].set_ylim(0,0.2)
    plt.show()
    
    model_result = result[window_size][model_num][model_name]
    filtered_param_df = chosen_day_parameter_heatmap(model_result, threshold, show='chosen')
    
    print("_____________________ Distribution of predicted values  _____________________")
    chosen['better'] = (chosen['model_performance'] - chosen['benchmark_performance']) <= 0
    histogram(Y1, chosen)
    
    cutoff_avg = Y1.loc['2017-02-15':'2020-12-31']['abs_resid'].mean()
    cutoff_mdn = Y1.loc['2017-02-15':'2020-12-31']['abs_resid'].median()
    print("_____________________ Classification Table (Mean as cutoff) _____________________")
    classification_table(Y1, chosen, cutoff_avg)
    print("_____________________ Classification Table (Median as cutoff) _____________________")
    classification_table(Y1, chosen, cutoff_mdn)
    
    if wordcloud:
        print("_____________________ Word Cloud  _____________________")
        run_wordcloud(selected, chosen)
    
    return (accepted_df, selected, chosen)


# (result_file, pred_file, Y1_file, window_size, model_num, threshold=None, wordcloud=False)
accepted_df, selected, chosen = read_result('result.pkl', '202106121438-PREDRESULT.pkl', '202106121441-Y1.pkl', 60, 2, wordcloud=True)
