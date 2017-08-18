""" Compute protein sequence k-mers """
from seq2seq.models import SimpleSeq2Seq
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
INPUT_SEQUENCE_LENGTH = 20
TOKEN_REPRESENTATION_SIZE = 10
HIDDEN_LAYER_DIMENSION = 3
TOKEN_DICT_SIZE = 4
ANSWER_MAX_TOKEN_LENGTH = 8
X = np.zeros((TRAIN_BATCH_SIZE, INPUT_SEQUENCE_LENGTH, TOKEN_REPRESENTATION_SIZE))

X[1] = input1
X[2] = input2
  
model = SimpleSeq2seq(
        input_dim=TOKEN_REPRESENTATION_SIZE,
        input_length=INPUT_SEQUENCE_LENGTH,
        hidden_dim=HIDDEN_LAYER_DIMENSION,
        output_dim=TOKEN_DICT_SIZE,
        output_length=ANSWER_MAX_TOKEN_LENGTH,
        depth=1
    )

             
data = np.array([np.array(input1), np.array(input2)])

ndata = np.expand_dims(data,2)
ndata = np.expand_dims(ndata,3)

X[1,] = input1
""" It won't work. It looks like we need to batch the input into chunks of same length """
y = model.predict(X)
