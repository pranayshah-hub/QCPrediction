# coding=utf-8

# Copyright [2019] [Pranay Shah].

# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Python 3

"""Auto-encoding Time-Delay Neural Network model."""

import tensorflow as tf
from keras.layers import Dense, Activation, LSTM, TimeDistributed, Input
from keras.models import Sequential, Model
from data_preprocessing import *


class AENN(object):
    """
    This class implements the Auto-Encoder Neural Network model.

    """

    def __init__(self, input_size, encode1_size, encode2_size):
        """
        Args:
          input_size: Tuple of the shape of the input.
          encode1_size: Integer value of the size of the first hidden layer.
          encode2_size: Integer value of the size of the latent vector.

        """
        self.input_size = input_size
        self.encode1_size = encode1_size
        self.encode2_size = encode2_size
        self.encode1 = Dense(self.encode1_size, activation='relu')
        self.encode2 = Dense(self.encode2_size, activation='relu')
        self.decode1 = Dense(self.encode1_size, activation='relu')
        self.decode2 = Dense(self.input_size, activation='relu')


    def forward(self):
        """
        This function implements the forward propagation of the Auto-Encoder Neural Network model.

            Returns:
              autoencoder_deep: The Keras Model instantiation of the entire AE.
              encoder_deep: The Keras Model instantiation of the Encoder.
              decoder_deep: The Keras Model instantiation of the Decoder.

        """
        # Specify input layer
        input_size = Input(shape=(self.input_size,))

        # Forward propagation (encode and decode)
        encode1 = self.encode1(input_size)
        encode2 = self.encode2(encode1)
        decode = self.decode1(encode2)
        output = self.decode2(decode)

        # Define models
        autoencoder_deep = Model(self.input_size, output)
        encoder_deep = Model(self.input_size, encode2)

        # Forward propagation (decode)
        latent_input = Input(shape=(self.encode2_size,))
        decoded_layer = autoencoder_deep.layers[-2](latent_input)
        output_layer = autoencoder_deep.layers[-1](decoded_layer)

        # Define models
        decoder_deep = Model(latent_input, output_layer)

        return autoencoder_deep, encoder_deep, decoder_deep


    def compile(self):
        """
        This function compiles each model.

            Returns:
              autoencoder: The compiled Keras AE.
              encoder: The compiled Keras Encoder.
              decoder: The compiled Keras Decoder.

        """
        autoencoder, encoder, decoder = self.forward()
        autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')
        return autoencoder, encoder, decoder

    @staticmethod
    def lrelu(x, alpha=0.1):
        """
        This function implements the leaky ReLU activation function.

        """
        return tf.maximum(alpha * x, x)

class TDNN(AENN):
    """
    This class implements the Time-Delay Neural Network model

        Args:
          observations:

        Returns:
          q_values:

    """

    def __init__(self, hidden_ts_size):
        """
            Args:
              hidden_ts_size: Integer value of the hidden layer size of the TDNN.

        """
        super(AENN, self).__init__()
        self.time_int = [3, 4, 5, 6, 7, 8, 9, 10]
        self.input_size_ts = self.encode2_size * (self.time_int[1])
        self.hidden_ts_size = hidden_ts_size
        self.output_size = self.encode2_size
        self.hidden_ts = Dense(self.hidden_ts_size, activation = 'relu')
        self.output = Dense(self.output_size, activation = 'relu')

    def forward(self):
        """
        This function implements the forward propagation of the TDNN.

            Returns:
              time_series: The Keras Model instantiation of the TDNN.

        """
        # Specify input layer
        input_shape_ts = Input(shape=(self.input_size_ts,))

        # Forward propagation (encode and decode)
        hidden_ts = self.hidden_ts(input_shape_ts)
        output = self.output(hidden_ts)

        # Define models
        time_series = Model(input_shape_ts, output)

        return time_series


    def compile(self):
        """
        This function compiles the model.

            Returns:
              tdnn: The compiled Keras TDNN.

        """
        tdnn = self.forward()
        tdnn.compile(optimizer='rmsprop', loss='mean_squared_error')
        return tdnn


def training_aenn(training_data_dividing_constant):
    """
    This function implements the Auto-encoding Time-Delay Neural Network model

        Args:
          training_data_dividing_constant: Integer scale factor for dividing up the
            data into mini-batches.

        Returns:
          encoder: The trained encoder.
          decoder: The trained decoder.
          data_dict: Dictionary of the file names of each data set.
          list_of_vals_and_feats: A List of strings of the names of features.

    """

    # Produce training data
    FB1060_data_QC21 = ['FB21_labled_traindata_balanced', 'FB21_labled_testdata']
    FB1060_data_QC22 = ['FB22_labled_traindata_balanced', 'FB22_labled_testdata']
    FB1060_data_QC23 = ['FB23_labled_traindata_balanced', 'FB23_labled_testdata']
    FB1060_data_QC23x = ['FB23x_labled_traindata_balanced', 'FB23x_labled_testdata']
    FB1060_data_QC24 = ['FB24_labled_traindata_balanced', 'FB24_labled_testdata']

    data_dict = {'21': FB1060_data_QC21,
                 '22': FB1060_data_QC22,
                 '23': FB1060_data_QC23,
                 '23x': FB1060_data_QC23x,
                 '24': FB1060_data_QC24}

    training_data, val_data, list_of_vals_and_feats = preprocessing_aenn(data_dict, qc='21')

    # Define model
    aenn = AENN(input_size=20,
                encode1_size=14,
                encode2_size=8)

    # Compile model
    autoencoder, encoder, decoder = aenn.compile()

    # Training
    ae_training_data_size = int(training_data.shape[0])
    b_size = int(ae_training_data_size / training_data_dividing_constant)
    autoencoder.fit(training_data, training_data,
                         epochs=100,
                         batch_size=b_size,
                         shuffle=True,
                         validation_data=(val_data, val_data))

    return encoder, decoder, data_dict, list_of_vals_and_feats

def training_tdnn(data_dict, bs_scaling_constant, encoder):
    """
    This function implements the Auto-encoding Time-Delay Neural Network model

        Args:
          data_dict: Dictionary of the file names of each data set.
          bs_scaling_constant: Integer scale factor to divide the training
            data into mini-batches for the TDNN.
          encoder: The trained encoder model.

        Returns:
          time_delay_nn: The trained TDNN.
          x_ts_test: The test data for the TDNN.
          target_ts_test: The target column of the test data for the TDNN.
          encoded_columns: List of column names for the encoded data.

    """

    # Define model
    tdnn = TDNN(hidden_ts_size=20)
    time_int = tdnn.time_int
    encoding_size = tdnn.encode2_size

    # Produce training data
    (x_ts_train,
     x_ts_test,
     target_ts_train,
     target_ts_test,
     encoded_columns) = preprocessing_ts(data_dict,
                                         encoder,
                                         time_int,
                                         encoding_size)

    # Compile model
    time_delay_nn = tdnn.compile()

    # Training
    b_size = int(x_ts_train.shape[0] / bs_scaling_constant)
    time_delay_nn.fit(x_ts_train, target_ts_train,
                         epochs=20,
                         batch_size=b_size,
                         shuffle=False,
                         validation_data=None)

    return time_delay_nn, x_ts_test, target_ts_test, encoded_columns











