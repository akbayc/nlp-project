import pandas as pd
from sklearn.model_selection import train_test_split
# text processing libraries
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk import FreqDist

# keras nn libraries
from keras.models import Model
from keras.layers import LSTM, Input, Dense, Dropout, Activation, Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# other libraries
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from trustpilotdata import get_data


def remove_sw(x):
    return [w for w in x if w not in sw]


def remove_punct(x):
    return [w for w in x if w not in punct]


def stem_words(x):
    return [stemmer.stem(w) for w in x]


stemmer = PorterStemmer()

punct = list(string.punctuation)
punct.append('...')
punct.append('’')
punct.append("''")
punct.append('``')

sw = stopwords.words('english')
sw.append('ca')
sw.append('n\'t')
sw.append('\'s'),
sw.append('\'re')
sw.append('\'ve')
sw.append('\'m')
sw.append('\'d')
sw = set(sw)

# read df
df = get_data(url = 'https://www.trustpilot.com/review/www.hsbc.co.uk?page=',
                 page = 80)


# creating target columns -> rating
df['rating'] = df['comment_score'].map(lambda x: 0 if x < 3 else (1 if x == 3 else 2))
counts = df['rating'].value_counts()
class_weight = {0: 1, 1: 50, 2: 14}

# lower all letters
df['comment'] = df['comment'].map(lambda x: x if type(x) != str else x.lower())

# word tokenize words
df['comment'] = df['comment'].apply(word_tokenize)

# removing punctuation signs
df['comment'] = df['comment'].apply(remove_punct)

# remove stopwords
df['comment'] = df['comment'].apply(remove_sw)

alltokens = []
for i in df['comment']:
    for j in i:
        alltokens.append(j)

freq = FreqDist(alltokens)
freq.plot(30, cumulative=False)


# stemming words
df['comment'] = df['comment'].apply(stem_words)


""" ---------------------------LSTM MODEL--------------------------- """

X = df['comment']
Y = df['rating']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

max_words = 1000
max_len = 300
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

def RNN():
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(input_dim=max_words, output_dim=50)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256)(layer)
    layer = Activation('relu')(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


model = RNN()

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(sequences_matrix, Y_train, batch_size=128, epochs=10, validation_split=0.2,
          class_weight=class_weight)

