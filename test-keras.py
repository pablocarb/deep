from keras.models import Sequential
from keras.layers import Dense, Activation

""" There are two types of models: Sequential and the Model class (functional API) """

""" This is still an empty model """
model = Sequential()

""" Core layers: Dense, Activation, Dropout (to avoid overfitting), Flatten, Reshape, Permute, RepeatVector, Lambda (arbitrary expression), ActivityRegularization (update to cost function based input activity), Masking """
""" Convolutional layers: Conv1D, Conv2D, SeparableConv2D, Conv2DTranspose, Conv3D, Cropping1D (crops along the time dimension (axis 1), Cropping2D, Cropping3D, UpSampling1D, UpSampling2D, UpSampling3D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D """
""" Pooling layers: MaxPooling1D, MaxPooling2D, MaxPooling3D, AveragePooling1D, AveragePooling2D, AveragePooling3D, GlobalMaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling2D, GlobalAveragePooling2D """
""" LocallyConnected1D, LocallyConnected2D """
""" Recurrent layers: SimpleRNN, GRU, LSTM """
""" Embedding layers: Embedding (only initial) """
""" Merge leyers: Add, Multiply, Average, Maximum, Concatenate, Dot """
""" Advanced activations: LeakyReLU, PReLU, ELU, ThresholdedReLU """
""" Normalization layers: BatchNormalization """
""" Noise layers: GaussianNoise, GaussianDropout, AlphaDropout """
""" Layer wrappers : TimeDistributed (applies a layer to every temporal slidce, Bidirectional: for RNNs """
