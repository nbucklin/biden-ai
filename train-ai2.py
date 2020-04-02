from __future__ import print_function
#import Keras library
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint


#import spacy, and spacy french model
# spacy is used to work on text
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
nlp.max_length = 30000000

#import other libraries
import numpy as np
import random
import sys
import os
import time
import codecs
import collections
from six.moves import cPickle

#define parameters used in the tutorial
data_dir = "C:\\Users\\paperspace\\Desktop\\ai-biden\\"
save_dir = data_dir # directory to store trained NN models
file_list = ["101","102","103","104","105","106","107","108","109","110","111","112","201","202","203","204","205","206","207","208","209","210","211","212","213","214","301","302","303","304","305","306","307","308","309","310","311","312","313","314","401","402","403","404","405","406","407","408","409","410","411","412"]
vocab_file = os.path.join(save_dir, "words_vocab.pkl")
sequences_step = 1 #step to create sequences

os.chdir("C:\\Users\\paperspace\\Desktop\\ai-biden\\")

def create_wordlist(doc):
    w1 = []
    for word in doc:
        if word not in ("\n","\n\n",'\u2009','\xa0'):
            w1.append(word.lower())
    return w1    
    
wordlist = []

with codecs.open('biden-speeches.txt','r',encoding='utf8') as t:
    data = t.read().lower()
        
#create sentences
doc = nlp(data)
wordlist = create_wordlist(data)

# count the number of words
word_counts = collections.Counter(wordlist)

# Mapping from index to word : that's the vocabulary
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]

#size of the vocabulary
vocab_size = len(words)
print("vocab size: ", vocab_size)

#save the words and vocabulary
with open(os.path.join(vocab_file), 'wb') as f:
    cPickle.dump((words, vocab, vocabulary_inv), f)

seq_length = 30
#create sequences
sequences = []
next_words = []
for i in range(0, len(wordlist) - seq_length, sequences_step):
    sequences.append(wordlist[i: i + seq_length])
    next_words.append(wordlist[i + seq_length])

print('nb sequences:', len(sequences))

X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence):
        X[i, t, vocab[word]] = 1
    y[i, vocab[next_words[i]]] = 1
    
def bidirectional_lstm_model(seq_length, vocab_size):
    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"),input_shape=(seq_length, vocab_size)))
    model.add(Dropout(0.6))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    
    optimizer = Adam(lr=learning_rate)
    callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    print("model built!")
    return model

rnn_size = 256 # size of RNN
seq_length = 30 # sequence length
learning_rate = 0.001 #learning rate

md = bidirectional_lstm_model(seq_length, vocab_size)
md.summary()    

batch_size = 32 # minibatch size
num_epochs = 50 # number of epochs

filepath="weights-improvement2-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(save_dir, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
#fit the model
history = md.fit(X, y,
                 batch_size=batch_size,
                 shuffle=True,
                 epochs=num_epochs,
                 validation_split=0.1,
                 callbacks = callbacks_list)
#save the model
#md.save(save_dir + "/" + 'my_model_generate_sentences.h5')





