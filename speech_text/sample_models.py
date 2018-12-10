from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, AtrousConv1D, MaxPooling1D, Dense, Input,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout,advanced_activations)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer

    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization
    #added batch Normalization
    bn_rnn =  BatchNormalization()(simp_rnn)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    #added Time Distributed Dense wrapper
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization

    #added Batch Normalization layer
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer

    #added Time Distributed Dense wrapper
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization

    cur_layer = input_data

    #Added GRU, BatchNormalization in a loop (depends on count of recur_layers)
    for layer in range(recur_layers):
        rnn_name = 'rnn_'+ str(layer)
        bn_name = 'bn_'+ str(layer)
        rnn = GRU(units, activation='relu',
                  return_sequences=True, implementation=2,
                  name=rnn_name)(cur_layer)
        cur_layer = BatchNormalization(name=bn_name)(rnn)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer

    #added Time Distributed Dense wrapper
    time_dense =  TimeDistributed(Dense(output_dim))(cur_layer)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer

    #Added a Bidirectional GRU Layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, name="bi_directional", implementation=2),merge_mode='concat')(input_data)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer

    #Added Time Distributed Dense wrapper
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, recur_layers , kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a deep network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network

    # Add convolution layers
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)

    #Add convolution with dilation
    # conv_1d = Conv1D(filters, kernel_size,
    #                  padding=conv_border_mode,
    #                  activation='relu',
    #                  dilation_rate = conv_stride,
    #                  name='conv1d')(input_data)

    # Bbatch normalization
    bn_cnn = BatchNormalization(name='bn_1')(conv_1d)

    # Dropout to avoid over fitting
    dropout1 = Dropout(0.2)(bn_cnn)

    #Maxpooling  to reduce the size and generalize the features
    #maxpool1 = MaxPooling1D(pool_size=2)(dropout1)

    rnn_data = dropout1

    # Add bidirectional recurrent layers with Batch Normalization in a loop
    for layer in range(recur_layers):
        # Create Names
        bi_rnn_name = 'bi_rnn_'+ str(layer)
        bn_name = 'bi_bn_'+ str(layer)

        # create BiDirectional RNN layer , try dropout_W and U
        rnn = Bidirectional(GRU(units, return_sequences=True,name=bi_rnn_name, implementation=2,dropout=0.2, recurrent_dropout=0.2,activation='tanh'),merge_mode='concat')(rnn_data)

        # create BiDirectional RNN layer , and use LeakyRelu activation
        #rnn = Bidirectional(GRU(units, return_sequences=True,name=bi_rnn_name, implementation=2),merge_mode='concat')(rnn_data)
        #rnn_leaky = advanced_activations.LeakyReLU(alpha=.01)(rnn)

        bnn = BatchNormalization(name=bn_name)(rnn)

        # Dropout to avoid overfitting
        rnn_data = Dropout(0.2)(bnn)

    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn_data)

    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)

    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)

    print(model.summary())
    return model
