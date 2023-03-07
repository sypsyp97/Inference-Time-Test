from tensorflow import keras
from keras import layers
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np


layer_type_space = ['conv_block', 'inverted_residual_block', 'mobilevit_block', 'None']
kernel_size_space = [1, 2, 3, 4]
stride_space = [1, 2]
filters_space = [8, 12, 16, 24, 32, 48, 64, 96]
expansion_factor_space = [1, 2]
residual_space = ['None', 'Add', 'StochasticDepth', 'Concatenate']
normalization_space = ['BatchNormalization', 'LayerNormalization']
activation_space = ['relu', 'relu6', 'silu', 'silu']
transformer_space = [1, 2, 3, 4]
head_space = [1, 2, 3, 4]


def check_model(model):
    contains_multi_head_attention = False
    for layer in model.layers:
        if 'multi_head_attention' in str(layer):
            contains_multi_head_attention = True
            break

    if contains_multi_head_attention:
        for layer in model.layers:
            if 'multi_head_attention' in str(layer):
                output_shape = layer.output.shape
                size = output_shape[1]
                if size > 1024:
                    return True
        return False

    else:
        return True


def conv_block(x, filters=16, kernel_size=3, strides=2, normalization='BatchNormalization', activation='silu6'):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)

    try:
        normalization_layer = {'BatchNormalization': layers.BatchNormalization(epsilon=1e-6),
                               'LayerNormalization': layers.LayerNormalization(epsilon=1e-6)}[normalization]
        x = normalization_layer(x)

    except KeyError:
        print(f"{normalization} not found in the list of normalization layers.")
        x = x

    try:
        activation_function = {'relu': tf.nn.relu,
                               'relu6': tf.nn.relu6,
                               'silu': tf.nn.silu,
                               'silu6': lambda x: tf.math.minimum(tf.nn.silu(x), 6)}[activation]
        x = activation_function(x)

    except KeyError:
        print(f"{activation} not found in the list of activation functions.")
        x = x

    return x


def inverted_residual_block(x, expansion_factor, output_channels, strides=1, kernel_size=3,
                            normalization='BatchNormalization', activation='silu6', residual='Concatenate'):

    m = layers.Conv2D(filters=expansion_factor * output_channels, kernel_size=1, padding="same")(x)
    try:
        normalization_layer = {'BatchNormalization': layers.BatchNormalization(epsilon=1e-6),
                               'LayerNormalization': layers.LayerNormalization(epsilon=1e-6)}[normalization]
        m = normalization_layer(m)

    except KeyError:
        print(f"{normalization} not found in the list of normalization layers.")
        m = m

    m = layers.DepthwiseConv2D(kernel_size, strides=strides, padding="same")(m)
    try:
        normalization_layer = {'BatchNormalization': layers.BatchNormalization(epsilon=1e-6),
                               'LayerNormalization': layers.LayerNormalization(epsilon=1e-6)}[normalization]
        m = normalization_layer(m)

    except KeyError:
        print(f"{normalization} not found in the list of normalization layers.")
        m = m

    try:
        activation_function = {'relu': tf.nn.relu,
                               'relu6': tf.nn.relu6,
                               'silu': tf.nn.silu,
                               'silu6': lambda x: tf.math.minimum(tf.nn.silu(x), 6)}[activation]
        m = activation_function(m)

    except KeyError:
        print(f"{activation} not found in the list of activation functions.")
        m = m

    m = layers.Conv2D(output_channels, 1, padding="same")(m)

    try:
        normalization_layer = {'BatchNormalization': layers.BatchNormalization(epsilon=1e-6),
                               'LayerNormalization': layers.LayerNormalization(epsilon=1e-6)}[normalization]
        m = normalization_layer(m)

    except KeyError:
        print(f"{normalization} not found in the list of normalization layers.")
        m = m

    if strides == 1 and residual == 'Concatenate':
        m = layers.Concatenate(axis=-1)([m, x])

    elif tf.math.equal(x.shape[-1], output_channels) and strides == 1:

        if residual == 'Concatenate':
            m = layers.Concatenate(axis=-1)([m, x])
        elif residual == 'StochasticDepth':
            m = tfa.layers.StochasticDepth(0.5)([m, x])
        elif residual == 'Add':
            m = layers.Add()([m, x])
        else:
            m = m

    elif strides == 2 and residual == 'Concatenate':
        x = layers.Conv2D(output_channels, kernel_size=(2, 2), strides=2, padding="same")(x)
        m = layers.Concatenate(axis=-1)([m, x])

    elif tf.math.equal(x.shape[-1], output_channels) and strides == 2:
        x = layers.Conv2D(output_channels, kernel_size=(2, 2), strides=2, padding="same")(x)
        try:
            normalization_layer = {'BatchNormalization': layers.BatchNormalization(epsilon=1e-6),
                                   'LayerNormalization': layers.LayerNormalization(epsilon=1e-6)}[normalization]
            x = normalization_layer(x)

        except KeyError:
            print(f"{normalization} not found in the list of normalization layers.")
            x = x

        if residual == 'Concatenate':
            m = layers.Concatenate(axis=-1)([m, x])
        elif residual == 'StochasticDepth':
            m = tfa.layers.StochasticDepth(0.5)([m, x])
        elif residual == 'Add':
            m = layers.Add()([m, x])
        else:
            m = m

    else:
        m = m

    return m


def ffn(x, hidden_units, dropout_rate, use_bias=False):
    a = tf.reshape(x, (-1, 1, x.shape[1], x.shape[-1]))

    for hidden_unit in hidden_units:
        a = tf.keras.layers.Conv2D(filters=hidden_unit, kernel_size=1, padding='same', use_bias=use_bias)(a)
        a = layers.LayerNormalization(epsilon=1e-6)(a)
        a = tf.math.minimum(tf.nn.silu(a), 6)
        a = layers.Dropout(dropout_rate)(a)

    x = tf.reshape(a, (-1, x.shape[1], x.shape[-1]))

    return x


def transformer_block(encoded_patches, transformer_layers, projection_dim, num_heads=2):
    for i in range(transformer_layers):
        # Layer normalization 1.
        t1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads,
                                                     key_dim=projection_dim)(t1, t1)
        # Skip connection 1.
        t2 = tfa.layers.StochasticDepth()([attention_output, encoded_patches])
        # Layer normalization 2.
        t3 = layers.LayerNormalization(epsilon=1e-6)(t2)
        t3 = ffn(t3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.1)

        # Skip connection 2.
        encoded_patches = tfa.layers.StochasticDepth()([t3, t2])

    encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    return encoded_patches


def mobilevit_block(x, num_blocks, projection_dim, strides=1, kernel_size=3, num_heads=2,
                    residual='Concatenate', activation='silu6', normalization='BatchNormalization'):
    local_features = conv_block(x, filters=projection_dim, kernel_size=kernel_size,
                                strides=strides, activation=activation, normalization=normalization)
    local_features = conv_block(local_features, filters=projection_dim, kernel_size=1,
                                strides=1, activation=activation, normalization=normalization)

    non_overlapping_patches = tf.reshape(local_features, (
        -1, tf.shape(local_features)[1] * tf.shape(local_features)[2], tf.shape(local_features)[-1]))

    global_features = transformer_block(non_overlapping_patches, num_blocks, projection_dim, num_heads=num_heads)

    folded_feature_map = tf.reshape(global_features, (
        -1, tf.shape(local_features)[1], tf.shape(local_features)[2], tf.shape(local_features)[-1]))

    folded_feature_map = conv_block(folded_feature_map, filters=x.shape[-1], kernel_size=kernel_size,
                                    strides=1, activation=activation, normalization=normalization)

    if strides == 1:

        if residual == 'Concatenate':
            local_global_features = layers.Concatenate(axis=-1)([folded_feature_map, x])
        elif residual == 'StochasticDepth':
            local_global_features = tfa.layers.StochasticDepth(0.5)([folded_feature_map, x])
        elif residual == 'Add':
            local_global_features = layers.Add()([folded_feature_map, x])
        else:
            local_global_features = folded_feature_map

    elif strides == 2:
        x = layers.Conv2D(x.shape[-1], kernel_size=(2, 2), strides=2, padding="same")(x)

        if residual == 'Concatenate':
            local_global_features = layers.Concatenate(axis=-1)([folded_feature_map, x])
        elif residual == 'StochasticDepth':
            local_global_features = tfa.layers.StochasticDepth(0.5)([folded_feature_map, x])
        elif residual == 'Add':
            local_global_features = layers.Add()([folded_feature_map, x])
        else:
            local_global_features = folded_feature_map

    else:
        local_global_features = folded_feature_map

    local_global_features = conv_block(local_global_features, filters=projection_dim, kernel_size=1,
                                       strides=1, activation=activation, normalization=normalization)

    return local_global_features


def decoded_block(x, layer_array):
    layer_type_index = int(str(layer_array[0]) + str(layer_array[1]), 2)
    kernel_size_index = int(str(layer_array[2]) + str(layer_array[3]), 2)
    stride_index = layer_array[4]
    filters_index = int(str(layer_array[5]) + str(layer_array[6]) + str(layer_array[7]), 2)
    expansion_factor_index = layer_array[8]
    residual_index = int(str(layer_array[9]) + str(layer_array[10]), 2)
    normalization_index = layer_array[11]
    activation_index = int(str(layer_array[12]) + str(layer_array[13]), 2)
    transformer_index = int(str(layer_array[14]) + str(layer_array[15]), 2)
    head_index = int(str(layer_array[16]) + str(layer_array[17]), 2)

    layer_type_dict = {

        0: lambda x: conv_block(x, filters=filters_space[filters_index],
                                kernel_size=kernel_size_space[kernel_size_index],
                                strides=stride_space[stride_index],
                                normalization=normalization_space[normalization_index],
                                activation=activation_space[activation_index]),

        1: lambda x: inverted_residual_block(x, expansion_factor=expansion_factor_space[expansion_factor_index],
                                             kernel_size=kernel_size_space[kernel_size_index],
                                             output_channels=filters_space[filters_index],
                                             strides=stride_space[stride_index],
                                             normalization=normalization_space[normalization_index],
                                             activation=activation_space[activation_index],
                                             residual=residual_space[residual_index]),

        2: lambda x: mobilevit_block(x, num_blocks=transformer_space[transformer_index],
                                     projection_dim=filters_space[filters_index],
                                     strides=stride_space[stride_index],
                                     normalization=normalization_space[normalization_index],
                                     kernel_size=kernel_size_space[kernel_size_index],
                                     num_heads=head_space[head_index],
                                     activation=activation_space[activation_index],
                                     residual=residual_space[residual_index]),

        3: lambda x: x
    }
    return layer_type_dict[layer_type_index](x)


def create_model(model_array, num_classes=5, input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = conv_block(x, kernel_size=2, filters=16, strides=2)

    for i in range(9):
        x = decoded_block(x, model_array[i])

    x = conv_block(x, filters=320, kernel_size=1, strides=1)
    x = layers.GlobalAvgPool2D()(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    return model


def create_first_population(population=20, num_classes=5):

    model_list = []
    first_population_array = np.random.randint(0, 2, (population, 9, 18))

    for i in range(population):
        model = create_model(first_population_array[i], num_classes=num_classes)
        while check_model(model):
            first_population_array[i] = np.random.randint(0, 2, (9, 18))
            model = create_model(first_population_array[i], num_classes=num_classes)
        model_list.append(model)

    return first_population_array, model_list





