""" Testing a simple deep learning classifier using LSTM for modeling sequences """
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tools
import numpy as np
from keras.preprocessing.text import hashing_trick
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Activation, Dropout, Merge, Input
from keras.layers.merge import Concatenate
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
    if 'flip' in dir(np): # np >= 1.12
        Xsr = np.flip( Xs, 1 )
    else:
        Xsr = np.fliplr( Xs )
    return Xsr

def tensorReact(reac):
    """ To do: provide a list of reaction SMILES, return a tensor """
    pass

""" Maximum sequence length (padded with zeros) """
MAX_SEQ_LENGTH = 1000
""" K-mer depth """
SEQDEPTH = 1 # 4
""" Hashed dictionary size """
TOKEN_SIZE = 20


print("Building training set...")

#DATASET = 'THERMO'
#DATASET = 'THERMO2'
#DATASET = 'EC'
DATASET = 'RFP'

if DATASET == 'EC':
    seqs, seqids, Y, Yids = tools.ecdataset()
    SEQDEPTH = 4
    LSTMDIM = 128
    HIDDENDIM = 256
    METRICS = 'categorical_accuracy'
    OUTACTIVATION = 'softmax'
    EPOCHS = 1000
    LOSS = 'categorical_crossentropy'
    OPTIMIZER='rmsprop'
    SUBSAMPLE = None
    BATCH_SIZE = 100
elif DATASET == 'RFP':
    seqs, seqids, Y, Yids = tools.seq2reacdataset(8)
    # TO DO: keep only non-zer bits
    SEQDEPTH = 4
    LSTMDIM = 128
    HIDDENDIM = 256
    METRICS = 'categorical_accuracy'
    OUTACTIVATION = 'sigmoid'
    EPOCHS = 10
    LOSS = 'binary_crossentropy'
    LOSS = 'mse'
    SUBSAMPLE = 50000
    OPTIMIZER='adamax'
    BATCH_SIZE = 5000
elif DATASET == 'THERMO':
    seqs, seqids, Y, Yids = tools.thermodataset(balanced=True)
    SEQDEPTH = 2
    LSTMDIM = 32
    HIDDENDIM = 16
    METRICS = 'categorical_accuracy'
    OUTACTIVATION = 'sigmoid'
    EPOCHS = 10
    LOSS = 'categorical_crossentropy'
    SUBSAMPLE = None
    OPTIMIZER='rmsprop'
    BATCH_SIZE = 100
elif DATASET == 'THERMO2':
    seqs, seqids, Y, Yids = tools.thermodataset2()
    SEQDEPTH = 8
    LSTMDIM = 256
    HIDDENDIM = 32
    METRICS = 'binary_accuracy'
    OUTACTIVATION = 'softmax'
    EPOCHS = 10
    TOKEN_SIZE = 40
    BATCH_SIZE = 100
    LOSS = 'categorical_crossentropy'
    OPTIMIZER='rmsprop'
    SUBSAMPLE = None
    vseqs, vseqids, vY, vYids = tools.thermodataset(balanced=True)
    vY =  np_utils.to_categorical(vY)
    vXsr =  tensorSeqHashed(vseqs, MAX_SEQ_LENGTH, SEQDEPTH, TOKEN_SIZE, HASH_FUNCTION= 'md5')

# Shuffle order to avoid biases
ix = [i for i in np.arange(0, len(seqs))]
np.random.shuffle( ix )
if SUBSAMPLE is not None:
    ix = ix[0:SUBSAMPLE]
seqs = [seqs[i] for i in ix]
seqids = [seqids[i] for i in ix]
Y = [Y[i] for i in ix]
Yids = [Yids[i] for i in ix]

TRAIN_BATCH_SIZE = len(seqs)
if DATASET != 'RFP':
    Y =  np_utils.to_categorical(Y)
else:
    Y = np.array( Y )

print("Training set [%s]: %d; Classes: %d" % (DATASET, len(seqs), len(set(Yids))))

print("Allocating tensors...")

HASHED = True
if HASHED:
    Xsr =  tensorSeqHashed(seqs, MAX_SEQ_LENGTH, SEQDEPTH, TOKEN_SIZE, HASH_FUNCTION= 'md5')
else:
    Xsr = tensorSeq(seqs, MAX_SEQ_LENGTH, SEQDEPTH, TOKEN_SIZE)



print("Model definition...")

model = Sequential()
model.add( LSTM(LSTMDIM, input_shape=(MAX_SEQ_LENGTH, SEQDEPTH*TOKEN_SIZE),
                activation='tanh', recurrent_activation='sigmoid',
                dropout=0.2, recurrent_dropout=0.01) ) #, return_sequences=True) )
#model.add( LSTM(32, dropout=0.1, recurrent_dropout=0.1) )
# model.add( Dense(100, activation='tanh') )
model.add( Dense(HIDDENDIM, activation='relu') )
model.add( Dense(Y.shape[1], activation=OUTACTIVATION) )

# A test example about merging layers: merge the output of the RNN with some other external input tensor
IMERGE = False
if IMERGE:
    # This is my sequence tensor
    input1 = Input(shape=(MAX_SEQ_LENGTH, SEQDEPTH*TOKEN_SIZE))
    x10 = LSTM(LSTMDIM, activation='tanh', recurrent_activation='sigmoid',
         dropout=0.2, recurrent_dropout=0.01)(input1)
    x1 = Dense(16)(x10)

    # A random input tensor just for testing purposes
    noise = np.random.normal( size=(len(seqs), 32) )
    input2 = Input(shape=(noise.shape,))
    x2 = Dense(8, activation='relu')(input2)

    # Concatenate
    concat = Concatenate()([x1, x2])
    out1 = Dense(18)(concat)
    out = Dense(Y.shape[1])(out1)

    # Define model from tensor graph
    bigmodel = Model(inputs=[input1, input2], outputs=out)
    bigmodel.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    bigmodel.fit([Xsr,noise], Y, epochs=1000, batch_size=100, validation_split=0.1)

# Typically for LSTM, we use RMSprop(lr=0.01) optimizer

#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=[METRICS])

#model.optimizer.lr = 0.1

model.fit(Xsr, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)
# This was a test for THERMO2 set
#model.fit(Xsr, Y, epochs=EPOCHS, batch_size=100, validation_data=(vXsr, vY)) #validation_split=0.1)
