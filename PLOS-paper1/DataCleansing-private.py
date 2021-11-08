import pandas as pd
import nltk
import re
import os
import numpy as np

df = pd.DataFrame()
files = os.listdir('./DataWithQuality')
for file in files:
    if file != '.DS_Store':
        data = pd.read_csv('./DataWithQuality/{}'.format(file))
        df = pd.concat([df, data])

df = df[df['lang']=='en']
df.sort_values(by=['story_date'], inplace=True)
df.reset_index(drop=True, inplace=True)
df.rename(columns={'story_date': 'date'}, inplace=True)
df.drop_duplicates('headline', inplace=True)

# Remove stop words except those for negations, [n, could, may] are not in the stopword list
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop = stop + ['could','may','might','would','also','said','says','say', 'rate', 'rates', 'rated', 'report', 'reports', 'reported', 'reuters',
               'monday','tuesday','wednesday','thursday','friday','saturday','sunday', 'business', 'dollar', 'dollars','ratings',
               'pts','points', 'hong kong', 'hk', 'index', 'market', 'markets', 'financial', 'fitch', 'rating', 'company', 'companies',
               'day', 'time', 'week', 'night', 'last', 'one', 'two', 'three', 'since', 'first', 'second', 'third', 'business', 'firms', 'firm']

# Try both Lemmatizing and Stemming
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

lemmatizer = WordNetLemmatizer()
# ps = PorterStemmer()
# content = content.apply(lambda x: " ".join(ps.stem((x.strip())) for x in x.split()))

def clean_content(x):
    x = re.sub('\[.*?\]|<.*?>|'
               'Co\s|Ltd\s|'
               'http.*?\s|'
               'HTTP.*?\s|'
               '\'s|n\'t|'
               '\spct\s|percent|Percent|\sper\s|'
               'N/A|'
               '\s\dQ\s|'
               '\w+\..*?\.net|'
               '\w+\..*?\.com|'
               'Source text in Chinese|Keywords:|'
               'Further company coverage|'
               '[A-Z].*?\d+.*?\(Reuters\).*?-\s|'
               'Reporting by.*?\n|'
               'Editing by.*?\n', '', x, flags=re.IGNORECASE)
    x = re.sub('\n|/|-', ' ', x)
    x = re.sub('For a richer, multimedia version of Top News visit.*|'
               '\(.*?\)|', '', x)
    x = re.sub('[^A-Za-z\s]|\d', '', x)
    x = re.sub('\s+', ' ', x)
    x = x.lower()
    
    # Remove stop words
    x = " ".join(lemmatizer.lemmatize(x.strip()) for x in nltk.word_tokenize(x) if x not in stop)
    
    

    return x

content = df['body']
content = content.apply(clean_content)

content.index = df['date']
content = content.to_frame()

cleaned_content = content[content['body'] != '']

content[content['body'].isnull() == False]

cleaned_content.to_csv('full_content_QD.csv')

