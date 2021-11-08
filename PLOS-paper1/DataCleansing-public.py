import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('full textual data.csv', index_col=0)

# Take the title into account
df['content'] = df['title'] + "' " + df['content']
content = df['content']

# Replace the header, e.g. HONG KONG March 25 (Reuters), By... x Min Read, Our Standards...Principles
content = content.str.replace(r"'[A-Z].+?Reuters\)","")
content = content.str.replace(r"By.*?,","")
content = content.str.replace(r"\d Min Read.*?,","")
content = content.str.replace(r"Our Standards.*?\.","")


# Delete names after Reporting by and Editing by
content = content.str.replace(r"Reporting by.+","")
content = content.str.replace(r"Editing by.+","")

# Other replacing
content = content.str.replace('-',' ')
content = content.str.replace('\n',' ')
content = content.str.replace('\t',' ')


# Separating n't from the verbs for better stopword removal
# Some words are not the same after separating n't, e.g. won't --> wo n't, so have to transform it 
# But for aren't --> are not then ok
content = content.str.replace(r"won't",' would not')
content = content.str.replace(r"shan't",' shall not')
content = content.str.replace(r"ain't",' am not')
content = content.str.replace(r"n't",' not')

# Can also delete 's...
content = content.str.replace("'s",'')

# Remove things inside ()
content = content.str.replace(r"\(.+?\)","")

# Remove things inside []
content = content.str.replace(r"/","")
# content = content.str.replace(r"\[[^\\']+?\]","")

# Remove all dollars e.g. HKD134, USD234
# content = content.apply(lambda x: re.sub(r"([A-Z]+).\d","",x))

# Remove all punctuations except ".", "," for negation
content = content.str.replace(r"[^\w\s\d\.,&]",' ')

# Remove stop words except those for negations, [n, could, may] are not in the stopword list
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop = list(set(stop) - set(['no','nor','not'])) + ['n','could','may','might','would','also','said','says','say',
                                                    'monday','tuesday','wednesday','thursday','friday','saturday','sunday','percent','pct',
                                                   'pts','points']
content = content.str.lower()
content = content.apply(lambda x: " ".join(x for x in nltk.word_tokenize(x) if x not in stop))
content = content.str.replace(r'[^\w\s]','')
content = content.str.replace(r'\d','')

# Lemmatizing the words
lemmatizer = WordNetLemmatizer()
content = content.apply(lambda x: " ".join(lemmatizer.lemmatize(x.strip()) for x in nltk.word_tokenize(x)))

# Reformat the dataframe
content.index = df['date']
content = content.to_frame()

content.to_csv('full_content.csv')