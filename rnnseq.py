""" Testing a simple deep learning classifier using LSTM for modeling sequences """

import tools
import numpy as np
from keras.preprocessing.text import hashing_trick
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from hashlib import md5

""" Maximum sequence length (padded with zeros) """
MAX_SEQ_LENGTH = 1000
""" K-mer depth """
SEQDEPTH = 4
""" Hashed dictionary size """
TOKEN_SIZE = 20

set1 = 'data/1.4.1.21.fasta'
set2 = 'data/2.6.1.1.fasta'

fas1 = tools.readfasta(set1)
fas2 = tools.readfasta(set2)

n1 = len(fas1)
n2 = len(fas2)

TRAIN_BATCH_SIZE = n1 + n2

seqs = [[fas1[s] for s in sorted(fas1)], [fas2[s] for s in sorted(fas2)]]

X = np.zeros( (TRAIN_BATCH_SIZE, MAX_SEQ_LENGTH, SEQDEPTH) )
Xs = np.zeros( (TRAIN_BATCH_SIZE, MAX_SEQ_LENGTH, SEQDEPTH*TOKEN_SIZE) )
Xs2 = np.zeros( (TRAIN_BATCH_SIZE, MAX_SEQ_LENGTH, SEQDEPTH*TOKEN_SIZE) )
Y = np.transpose( np.array( [np.append( np.ones(n1), np.zeros(n2) ),
                             np.append( np.zeros(n1), np.ones(n2) )] ) )
# Y = Y[:,0]

# Use to_categorical to convert to one-hot each entry in the sequence
# hot1 = np_utils.to_categorical( input1 - 1 )

""" Architecture 3: I just realised that there is no need for hashing, 
    but concatenate one-hot encoding down to desired depth """
n = 0
for i in range(0, len(seqs)):
    for j in range(0, len(seqs[i])):
        aaix = tools.aaindex( str(seqs[i][j].seq) )
        for l in range(0, len(aaix)):
            for k in range(0, SEQDEPTH):
                try:
                    Xs2[n, l, aaix[l+k] + TOKEN_SIZE*k] = 1
                except:
                    continue
        n += 1

n = 0
for i in range(0, len(seqs)):
    for j in range(0, len(seqs[i])):
        for k in range(1, SEQDEPTH+1):
            kmers = tools.kmers( str(seqs[i][j].seq), k )
            sequence = str(' '.join( kmers ))
            input1 = np.array(hashing_trick(sequence, hash_function= 'md5', n=TOKEN_SIZE+1))
            for l in range(0, len(input1)):
                """ Architecture 1: use hashed kmer """
                X[n, l, k-1] = input1[l]
                """ Architecture 2: activate input corresponding to hashed kmer (one-hot) """
                Xs[n, l, (k-1)*TOKEN_SIZE + input1[l]-1] = 1
        n += 1

""" Flip sequences (zero-padding at the start) """
Xsr = np.flip( Xs, 1 )
Xsr = np.flip( Xs2, 1 )

model = Sequential()
model.add( LSTM(128, input_shape=(MAX_SEQ_LENGTH, SEQDEPTH*TOKEN_SIZE),
                dropout=0.2, recurrent_dropout=0.2) ) # , return_sequences=True) )
# model.add( LSTM(32, dropout=0.1, recurrent_dropout=0.1) )
# model.add( Dense(100, activation='tanh') )
model.add( Dense(32, activation='elu') )
model.add( Dense(Y.shape[1], activation='sigmoid') )

# Typically for LSTM, we use RMSprop(lr=0.01) optimizer

# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.optimizer.lr = 0.001

model.fit(Xsr, Y, epochs=10, batch_size=100, validation_split=0.1)
