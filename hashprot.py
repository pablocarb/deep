""" Compute protein sequence k-mers """
from recurrentshop import RecurrentModel   
from seq2seq.models import SimpleSeq2Seq, Seq2Seq
from keras.models import Sequential
from keras.layers import Dense, Activation, Input

import numpy as np
ta1 = 'MAPPSVFAEVPQAQPVLVFKLTADFREDPDPRKVNLGVGAYRTDDCHPWVLPVVKKVEQKIANDNSLNHEYLPILGLAEFRSCASRLALGDDSPALKEKRVGGVQSLGGTGALRIGADFLARWYNGTNNKNTPVYVSSPTWENHNAVFSAAGFKDIRSYRYWDAEKRGLDLQGFLNDLENAPEFSIVVLHACAHNPTGIDPTPEQWKQIASVMKHRFLFPFFDSAYQGFASGNLERDAWAIRYFVSEGFEFFCAQSFSKNFGLYNERVGNLTVVGKEPESILQVLSQMEKIVRITWSNPPAQGARIVASTLSNPELFEEWTGNVKTMADRILTMRSELRARLEALKTPGTWNHITDQIGMFSFTGLNPKQVEYLVNEKHIYLLPSGRINVSGLTTKNLDYVATSIHEAVTKIQ'
ta2 = 'MAGNGAIVESDPLNWGAAAAELAGSHLDEVKRMVAQARQPVVKIEGSTLRVGQVAAVASAKDASGVAVELDEEARPRVKASSEWILDCIAHGGDIYGVTTGFGGTSHRRTKDGPALQVELLRHLNAGIFGTGSDGHTLPSEVTRAAMLVRINTLLQGYSGIRFEILEAITKLLNTGVSPCLPLRGTITASGDLVPLSYIAGLITGRPNAQAVTVDGRKVDAAEAFKIAGIEGGFFKLNPKEGLAIVNGTSVGSALAATVMYDANVLAVLSEVLSAVFCEVMNGKPEYTDHLTHKLKHHPGSIEAAAIMEHILDGSSFMKQAKKVNELDPLLKPKQDRYALRTSPQWLGPQIEVIRAATKSIEREVNSVNDNPVIDVHRGKALHGGNFQGTPIGVSMDNARLAIANIGKLMFAQFSELVNEFYNNGLTSNLAGSRNPSLDYGFKGTEIAMASYCSELQYLGNPITNHVQSADEHNQDVNSLGLVSARKTAEAIDILKLMSSTYIVALCQAVDLRHLEENIKASVKNTVTQVAKKVLTMNPSGELSSARFSEKELISAIDREAVFTYAEDAASASLPLMQKLRAVLVDHALSSGERGAGALRVLQDHQVRGGAPRGAAPGGGGRPRGVAEGTAPVANRIADSRSFPLYRFVREELGCVFLTGERLKSPGEECNKVFVGISQGKLVDPMLECLKEWDGKPLPINIK'

def kmers(seq, n):
    kpos = []
    for i in range(0, len(seq)-n):
        kmer = seq[i:(i+n)]
        kpos.append(min(kmer, kmer[::-1]))
    return kpos

""" Map the kmer sequence into hashed one-hot """
from keras.preprocessing.text import one_hot

input1 = one_hot(' '.join(kmers(ta1, 3)), n=10)
input2 = one_hot(' '.join(kmers(ta2, 3)), n=10)

# """ Next thing to test: use a recurrent layer in order to have constant input size """
# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras.layers.recurrent import LSTM

# # keras.layers.recurrent.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)

# from keras.layers import Embedding


# """ This is still an empty model """
# model = Sequential()
# model.add(Embedding(500, output_dim=128))
# model.add(LSTM(units=100))
# model.add(Activation('relu'))
# model.add(Dense(1))
# model.add(Activation('softmax'))

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
# import numpy as np

# model.fit([np.array(input)], [np.array([1])])

TRAIN_BATCH_SIZE = 2
INPUT_SEQUENCE_LENGTH = 1000
TOKEN_REPRESENTATION_SIZE = 10
HIDDEN_LAYER_DIMENSION = 3
TOKEN_DICT_SIZE = TOKEN_REPRESENTATION_SIZE
ANSWER_MAX_TOKEN_LENGTH = INPUT_SEQUENCE_LENGTH
X = np.zeros((TRAIN_BATCH_SIZE, INPUT_SEQUENCE_LENGTH, TOKEN_REPRESENTATION_SIZE))
Y = np.zeros((TRAIN_BATCH_SIZE, INPUT_SEQUENCE_LENGTH, TOKEN_REPRESENTATION_SIZE))

sample = [input1, input2]

for i in range(0, len(sample)):
    for j in range(0, len(sample[i])):
        px = sample[i][len(sample[i])-j-1]
        py = sample[i][j]
        token = np.zeros(TOKEN_REPRESENTATION_SIZE)
        token[px] = 1
        X[i,j] = token
        token = np.zeros(TOKEN_REPRESENTATION_SIZE)
        token[py] = 1
        Y[i,j] = token
  
# model = SimpleSeq2Seq(input_dim=TOKEN_REPRESENTATION_SIZE,input_length=INPUT_SEQUENCE_LENGTH,
#                       hidden_dim=HIDDEN_LAYER_DIMENSION,output_dim=TOKEN_DICT_SIZE,
#                       output_length=ANSWER_MAX_TOKEN_LENGTH,depth=1)

print('Create model')
model = Seq2Seq(input_dim=TOKEN_REPRESENTATION_SIZE,input_length=INPUT_SEQUENCE_LENGTH,
                      hidden_dim=HIDDEN_LAYER_DIMENSION,output_dim=TOKEN_DICT_SIZE,
                      output_length=ANSWER_MAX_TOKEN_LENGTH,depth=4)
print('Compile model')
model.compile(loss='mse', optimizer='rmsprop')
print('Train')
#model.fit(X, Y, epochs=1000)
print('Predict model')
y = model.predict(X)

# Once the model is fitted, we extract the encoder:

input1 = Input(shape=(INPUT_SEQUENCE_LENGTH, TOKEN_REPRESENTATION_SIZE))
input2 = model.layers[0](input1)     
input3 = model.layers[1](input2)     
input4 = model.layers[2](input3)     
input5 = model.layers[3](input4)     
input6 = model.layers[4](input5)     

rnn = RecurrentModel(input=input1, output=input6)

mn = RecurrentSequential()                       
mn.add(rnn.get_cell()) 

""" In principle, we have here the encoder as a keras recurrent layer """

mn.add(model.layers[5].get_cell()) 

kmodel = 


# data = np.array([np.array(input1), np.array(input2)])

# ndata = np.expand_dims(data,2)
# ndata = np.expand_dims(ndata,3)

# X[1,] = input1
# """ It won't work. It looks like we need to batch the input into chunks of same length """
# y = model.predict(X)
