import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l
from tensorflow.keras import backend as K

# Modified from implementation: https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py
def create_base_fcn(input_shape, n_feature_maps, model_name="fcn1", pad_val='valid'):
    input_layer = keras.layers.Input(input_shape, name="input")

    conv1 = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding=pad_val)(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding=pad_val)(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding=pad_val)(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
    return keras.Model(inputs=input_layer, outputs=gap_layer, name=model_name)




# Modified from implementation: https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py
def create_base_res1(input_shape, n_feature_maps, model_name="res1"):
    
    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    
    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    
    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    
    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    
    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)
    
    # BLOCK 2
    
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    
    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    
    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    
    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    
    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)
    
    # BLOCK 3
    
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 4, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    
    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 4, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    
    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 4, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    
    # no need to expand channels because they are equal
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 4, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    
    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)
    
    # FINAL
    
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
    return keras.Model(inputs=input_layer, outputs=gap_layer, name=model_name)

