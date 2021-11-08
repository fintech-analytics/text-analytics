from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns
from sklearn.linear_model import LinearRegression
import re
from statsmodels.tsa.arima_model import ARIMA


class rolling_LDA:
    """
    This is a class to conduct LDA in each period and generate a dictionary and/or wordsclouds for comparison
    - dataframe --> input dataframe (The index should be dates, values should be the cleaned news for each observation)
    - start --> start date of the entire period (in str)
    - end --> end date of the entire period (in str)
    - window_size --> size of the rolling window
    - roll_size --> the step size of the rolling window
    - n_topics --> number of topics generated in each year
    - max_df --> the threshold for data filtering: those appear more than max_df% of documents are ignored
    - min_df --> the threshold for data filtering: those appear less than min_df% of documents are ignored
    - top_n_words --> the top n components to display in each topic
    - vectorizer --> 'count' or 'tfidf'
    - show_wordclouds --> to display wordcloud as the output or not
    """
    
    
    def __init__(self, dataframe, start, end, window_size, roll_size, score = 'all', top_n_words = None, n_topics = None, max_df = None, min_df = None, vectorizer = 'count'):
        self.dataframe = dataframe
        self.start = start
        self.end = end
        self.window_size = window_size
        self.roll_size = roll_size
        self.n_topics = n_topics
        self.score = score
        self.max_df = max_df
        self.min_df = min_df
        self.top_n_words = top_n_words
        self.vectorizer = vectorizer
        self.content = None # To be declared
        self.lda_df = None # To be declared
        self.doc_topic_matrix = None # To be declared
        self.doc_mean_df = pd.DataFrame() # To be declared
        self.doc_gini_df = pd.DataFrame() # To be declared
        self.topic_gini_df = pd.DataFrame() # To be declared
        self.diversity_ranked_topics_df = pd.DataFrame() # To be declared
        self.dictionary = None # To be declared
        self.labels = None # To be declared
        self.diversity_label = None # TO be declared
        self.end_index = 0
        self.doc_topic_df = None
        self.last_date = None
        self.doc_length = pd.Series()
        self.word_appearence = pd.Series()
        self.DTM = None
        self.last_index = 0

        
    # (Main) Find all words appeared in the topics in the period and plot the distribution of probability of words in each year
    def period_summary(self):
        
        # Declare the variables
        df = self.dataframe
        top_n_words = self.top_n_words
        vectorizer = self.vectorizer
        
        # Set window size
        start_date = datetime.strptime(self.start, '%Y-%m-%d')
        end_date = datetime.strptime(self.end, '%Y-%m-%d')
        roll_start = start_date + timedelta(days=0)
        roll_end = roll_start + timedelta(days=self.window_size)
        rs = self.roll_size
        
        # Set the df index to datetime
        df.index = pd.to_datetime(df.index)
        df['index'] = df.reset_index().index
        
        # Create item to store the years, topics, words and probability
        summary_d = {}

        # Roll the window to do LDA
        period = 0 # To create the column names for doc_top_df
        while roll_end <= end_date:
            window = df[(df.index >= roll_start) & (df.index <= roll_end)]
            
            # Track the index
            try:
                index = list(window.index)[-1]
            except:
                continue
            if index == self.end_index:
                roll_start = roll_start + timedelta(days = rs)
                roll_end = roll_end + timedelta(days=rs)
                continue
            else:
                self.end_index = index
                pass
        
            # Print current work
            print("Period", str(roll_start) + " to " +str(roll_end))
            
            # Run LDA in the window
            content = window.content
            self.content = content

            self.LDA_tf(window['index'])
            
            # Document-topic matrix
            doc_top_df = pd.DataFrame(self.doc_topic_matrix, index = content.index)
            
            # Rank the topics by mean
            ranking = doc_top_df.mean()
            labels = list(ranking.rank(ascending=0).sort_values().index)
            self.labels = labels # This is defined for ranking topic word distributions as well
            doc_top_df = doc_top_df[labels]
            doc_top_df.columns = list(range(self.doc_topic_matrix.shape[1]))
                 
            # Extract last date of the window
            last_date = list(doc_top_df.index)[-1]
            self.last_date = last_date
            
            # Calculate the topic score by taking into account of mean
            self.doc_top_df = doc_top_df
            topic_score_mean = self.Topic_Score_Calculation()
            
            # Calculate the gini coefficient as the topic score
            p_square = doc_top_df**2
            doc_gini = 1 - p_square.sum(axis = 1)
            doc_top_df = doc_gini.to_frame()
            doc_top_df.columns = ['gini']
            self.doc_top_df = doc_top_df
            topic_score_gini = self.Topic_Score_Calculation()
            
            # Concat
            #self.doc_topic_df = pd.concat([self.doc_topic_df,doc_top_df], axis = 1, sort = False)
            self.doc_mean_df = pd.concat([self.doc_mean_df,topic_score_mean])
            self.doc_gini_df = pd.concat([self.doc_gini_df,topic_score_gini])
            
            # Topic-word-probability-matrix
            d = self.top_n_words_each_topic()

            if self.top_n_words != None:
                # Import data to the summary dictionary
                timeframe = str(roll_start.date()) + " to " + str(roll_end.date())
                summary_d[timeframe] = d
            else:
                topic_gini = pd.DataFrame(index = [last_date], columns = list(range(self.doc_topic_matrix.shape[1])))
                for topic in list(topic_gini.columns):
                    N = len(d[topic])
                    gini = sum([(x[1]**2)*N for x in d[topic]])
                    topic_gini.loc[last_date,topic] = gini
                self.topic_gini_df = pd.concat([self.topic_gini_df, topic_gini])
                
            # Rank topics by diversity
            if self.top_n_words == None:
                diversity_label = list(topic_gini.T.rank(ascending=0).iloc[:,0].sort_values().index)
                diversity_ranked_topics = topic_gini[diversity_label]
                diversity_ranked_topics.columns = list(range(diversity_ranked_topics.shape[1]))
                self.diversity_ranked_topics_df = pd.concat([self.diversity_ranked_topics_df, diversity_ranked_topics])
                
            # Move the window
            roll_start = roll_start + timedelta(days = rs)
            roll_end = roll_end + timedelta(days=rs)
                                      
            # Add one to the period index
            period += 1

        # After rolling the window
        if self.top_n_words != None:
            word_freq = summary_d
        else:
            word_freq = self.topic_gini_df
        
        return word_freq, self.doc_mean_df, self.doc_gini_df, self.diversity_ranked_topics_df
    
            
    # (Peripheral) LDA for tf matrix
    def LDA_tf(self, index):

        # Declare the variables
        content = self.content
        n_topics = self.n_topics
        vectorizer = self.vectorizer        
        
        # Allow choosing tfidf, default is TF
        # Make tf from the corpus
        if self.max_df != None and self.min_df != None and self.n_topics != None:
            max_df, min_df, n_components = self.max_df, self.min_df, self.n_topics
        else:
            max_df, min_df, n_components = self.optimal_params(content, vectorizer, n_topics)
        
        # Run the optimal LDA
        if vectorizer == 'tfidf':
            vec = TfidfVectorizer(max_df=max_df,min_df=min_df)
        else:
            vec = CountVectorizer(max_df=max_df,min_df=min_df)
        tf = vec.fit_transform(content)
        
        
        # get document term matrix
        columns = vec.get_feature_names()
        DTM = np.array(tf.toarray())
        self.DTM = DTM
        DocSeries = pd.Series((DTM>0).sum(axis=1), index=index)
        self.doc_length = self.doc_length.combine(DocSeries, max, fill_value=0) 
        WordSeries = pd.Series((DTM[self.last_index:]).sum(axis=0), index=columns)
        self.word_appearence = self.word_appearence.add(WordSeries, fill_value=0)
        self.last_index = index[-1] + 1

        
        lda = LatentDirichletAllocation(n_components = n_components)
        lda.fit(tf)
        
        # Compute document topic matrix
        doc_topic_matrix = lda.fit_transform(tf).round(3)

        # Compute probability of words
        probs = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

        # Create data frame of the lda model
        lda_df = pd.DataFrame(lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis], columns = columns)
        
        # Return the matrixes
        self.lda_df = lda_df
        self.doc_topic_matrix = doc_topic_matrix
    
    # (Peripheral) Find the optimal number of topics for each period
    def optimal_params(self, content, vectorizer, n_topics):
    
        if self.max_df != None and self.min_df != None:
            max_df = [self.max_df]
            min_df = [self.min_df]
        else:
            max_df = list(np.linspace(0.2,0.3,5))
            min_df = list(np.linspace(0.0001,0.001,5))

        log_likelihood = {}

        # Loop for 15 times to calculate the best max_df, min_df and n_components
        for loop in range(15):
            for x in max_df:
                for y in min_df:            
                    if vectorizer == 'tfidf':
                        vec = TfidfVectorizer(max_df=x,min_df=y)
                    else:
                        vec = CountVectorizer(max_df=x,min_df=y)
                    
                    tf = vec.fit_transform(content)

                    # If number of topics is set, no loop for topic
                    if n_topics != None:
                        lda = LatentDirichletAllocation(n_components = n_topics)
                        lda.fit(tf)
                        tup = (x,y,n_topics)
                        if tup in list(log_likelihood.keys()):
                            log_likelihood[tup] = log_likelihood[tup] + lda.score(tf)
                        else:
                            log_likelihood[tup] = lda.score(tf)
                        
                    # If number of topics not set, do loop
                    else:
                        for i in range(6, 15, 2):
                            lda = LatentDirichletAllocation(n_components = i)
                            lda.fit(tf)

                            tup = (x,y,i)

                            if tup in list(log_likelihood.keys()):
                                log_likelihood[tup] = log_likelihood[tup] + lda.score(tf)
                            else:
                                log_likelihood[tup] = lda.score(tf)
                                  
                                
            print(str((loop+1)*6) + '%' + 'Done')

        opt_params = sorted(log_likelihood.items(), key=lambda x: x[1], reverse=True)[0][0]
        print(opt_params)
        
        self.max_df = opt_params[0]
        self.min_df = opt_params[1]
        
        return opt_params[0], opt_params[1], opt_params[2]
 

    # (Peripheral) Find top n words in each topic for each period
    def top_n_words_each_topic(self):
        
        # Declare the variables
        lda_df = self.lda_df
        n = self.top_n_words
        
        # Rank the lda_df
        lda_df = lda_df.T
        lda_df = lda_df[self.labels]
        lda_df.columns = list(range(lda_df.shape[1]))
        lda_df = lda_df.T
        
        # Create dictionary to store topic, words and probability
        d = {}
        for i in range(len(lda_df)):
            if n != None:
                row_topn = lda_df.iloc[i,:].sort_values(ascending = False)[:n]
            else:
                row_topn = lda_df.iloc[i,:].sort_values(ascending = False)
            word = row_topn.index.tolist()
            prob = row_topn.values.tolist()
            tup = list(tuple(zip(word,prob)))
            d[i] = tup

        self.dictionary = d
        return d
    
    def Topic_Score_Calculation(self):
        
        # Define the input variables
        doc_top_df = self.doc_top_df
        last_date = self.last_date
        
        if self.score == 'single':
            # (1) Calculate topic score on the last date - use documents in the last day only
            try:
                topic_scores = doc_top_df.loc[last_date].mean().to_frame().T
                topic_scores.index = [last_date]
            except:
                topic_scores = doc_top_df.loc[last_date].to_frame().T

        elif self.score == 'all':
            # (2) Calculate topic score on the last date - average of all documents in the window
            topic_scores = doc_top_df.mean().to_frame().T
            topic_scores.index = [last_date]
            
        return topic_scores

        
    
    # (Optional function) Creat graphs
    @staticmethod
    def LDA_graphs(summary_d, n_words = 10, kind = 'wordcloud'):
        
        """ The dictionary is the output of rolling_LDA().period_summary()"""
        """ 
        kind = 'wordcloud' (default) --> generate wordcloud
        kind = 'bar' --> generate horizontal barchart
        """
           
        dictionary = summary_d
        
        period_names = list(dictionary.keys())
        period_values = list(dictionary.values())
        
        # Declare the variables
        for p_name, p_values in zip(period_names,period_values):

            pic_height = 5*np.ceil(len(p_values)/4)
            fig = plt.figure(figsize = (20,pic_height))
            
            for i in range(len(p_values)):
                d = {}
                for wp in p_values[i]:
                    w = wp[0]
                    p = wp[1]
                    d[w] = p
                    if len(d) >= n_words:
                        break

                n = np.ceil(len(p_values)/4)
                ax = plt.subplot(n,4,i+1)
                ax.set_title("Topic #" + str(i))
                
                if kind == 'bar':
                    plt.barh(list(d.keys()),list(d.values()))
                    plt.xticks(ticks =[])
                    plt.yticks(fontsize = 8)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    for i in ax.patches:
                        ax.text(i.get_width() , i.get_y() + 0.25, str(round((i.get_width()), 4)), fontsize=8, color='black')
                else:
                    wordcloud = WordCloud(width = 500, height= 500, max_font_size=60,max_words=100, background_color='white')
                    wordcloud.generate_from_frequencies(frequencies=d)
                    plt.imshow(wordcloud,interpolation="bilinear")
                    plt.axis("off")
            
            plt.suptitle(str(p_name))
            plt.show()