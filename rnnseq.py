""" Testing a simple deep learning classifier using LSTM for modeling sequences """
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tools
import numpy as np
from keras.preprocessing.text import hashing_trick
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from hashlib import md5

from importlib import reload
reload(tools)

def tensorSeq(seqs, MAX_SEQ_LENGTH, SEQDEPTH, TOKEN_SIZE):
    Xs2 = np.zeros( (TRAIN_BATCH_SIZE, MAX_SEQ_LENGTH, SEQDEPTH*TOKEN_SIZE) )
    """ Architecture 3: I just realised that there is no need for hashing,
    but concatenate one-hot encoding down to desired depth """
    for i in range(0, len(seqs)):
        for j in range(0, len(seqs[i])):
            aaix = tools.aaindex( seqs[i][j]  )
            for l in range(0, len(aaix)):
                for k in range(0, SEQDEPTH):
                    try:
                        Xs2[i, l, aaix[l+k] + TOKEN_SIZE*k] = 1
                    except:
                        continue
    """ Flip sequences (zero-padding at the start) """
    Xsr = np.flip( Xs2, 1 )
    return Xsr


def tensorSeqHashed(seqs, MAX_SEQ_LENGTH, SEQDEPTH, TOKEN_SIZE, HASH_FUNCTION= 'md5'):
    TRAIN_BATCH_SIZE = len(seqs)
    X = np.zeros( (TRAIN_BATCH_SIZE, MAX_SEQ_LENGTH, SEQDEPTH) )
    Xs = np.zeros( (TRAIN_BATCH_SIZE, MAX_SEQ_LENGTH, SEQDEPTH*TOKEN_SIZE) )
    for i in range(0, len(seqs)):
        for k in range(1, SEQDEPTH+1):
            kmers = tools.kmers( seqs[i], k )
            sequence = str(' '.join( kmers ))
            input1 = np.array(hashing_trick(sequence, hash_function= HASH_FUNCTION, n=TOKEN_SIZE+1))
            for l in range(0, len(input1)):
                try:
                    """ Architecture 1: use hashed kmer """
                    X[i, l, k-1] = input1[l]
                    """ Architecture 2: activate input corresponding to hashed kmer (one-hot) """
                    Xs[i, l, (k-1)*TOKEN_SIZE + input1[l]-1] = 1
                except:
                    continue
    Xsr = np.flip( Xs, 1 )
    return Xsr

""" Maximum sequence length (padded with zeros) """
MAX_SEQ_LENGTH = 1000
""" K-mer depth """
SEQDEPTH = 4
""" Hashed dictionary size """
TOKEN_SIZE = 20

fasfile = os.path.join('/mnt/SBC1/data/METANETX2', 'seqs.fasta')
infofile = os.path.join('/mnt/SBC1/data/METANETX2', 'reac_seqs.tsv')

print("Building training set...")
seqdict = tools.dbfasta(fasfile)
seqinfo, ecinfo = tools.seqinfo(infofile)
data = tools.ecdataset(seqdict, ecinfo)

pattern = ''
eclist = set()
for ec in ecinfo:
    if ec.startswith(pattern):
        eclist.add(ec)

TEST = False
if TEST:
    ectest = set(['1.4.1.13','2.2.1.2','3.1.1.3','4.1.1.11','5.1.1.1','6.1.1.18'])
    eclist = ectest

seqs, seqids, Y, Yids = tools.dataset(data, eclist)

ix = [i for i in np.arange(0, len(seqs))]
np.random.shuffle( ix )
seqs = [seqs[i] for i in ix]
seqids = [seqids[i] for i in ix]
Y = [Y[i] for i in ix]

TRAIN_BATCH_SIZE = len(seqs)
Y =  np_utils.to_categorical(Y)

print("Training set [%s]: %d; Classes: %d" % (pattern, len(seqs), len(set(Yids))))

print("Allocating tensors...")

HASHED = True
if HASHED:
    Xsr =  tensorSeqHashed(seqs, MAX_SEQ_LENGTH, SEQDEPTH, TOKEN_SIZE, HASH_FUNCTION= 'md5')
else:
    Xsr = tensorSeq(seqs, MAX_SEQ_LENGTH, SEQDEPTH, TOKEN_SIZE)



print("Model definition...")

model = Sequential()
model.add( LSTM(128, input_shape=(MAX_SEQ_LENGTH, SEQDEPTH*TOKEN_SIZE),
                activation='tanh', recurrent_activation='sigmoid',
                dropout=0.2, recurrent_dropout=0.01) ) #, return_sequences=True) )
#model.add( LSTM(32, dropout=0.1, recurrent_dropout=0.1) )
# model.add( Dense(100, activation='tanh') )
model.add( Dense(256, activation='relu') )
model.add( Dense(Y.shape[1], activation='softmax') )

# Typically for LSTM, we use RMSprop(lr=0.01) optimizer

# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

#model.optimizer.lr = 0.001

model.fit(Xsr, Y, epochs=1000, batch_size=100, validation_split=0.1)
