from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten


def deep(features_shape, num_classes, act='relu'):
    # Input
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x

    # Flatten
    o = Flatten(name='flatten')(o)

    # Dense layer
    o = Dense(512, activation=act, name='dense1')(o)
    o = Dense(512, activation=act, name='dense2')(o)
    o = Dense(512, activation=act, name='dense3')(o)

    # Predictions
    pred_layer = Dense(num_classes, activation='softmax', name='pred')(o)

    # Print network summary
    Model(inputs=x, outputs=o).summary()

    return Model(inputs=x, outputs=o)


from keras.layers import Conv2D, MaxPooling2D, BatchNormalization


def deep_cnn(features_shape, num_classes, act='relu'):
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x

    # Block 1
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block1_conv', input_shape=features_shape)(o)
    o = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block1_pool')(o)
    o = BatchNormalization(name='block1_norm')(o)

    # Block 2
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block2_conv')(o)
    o = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(o)
    o = BatchNormalization(name='block2_norm')(o)

    # Block 3
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block3_conv')(o)
    o = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(o)
    o = BatchNormalization(name='block3_norm')(o)

    # Flatten
    o = Flatten(name='flatten')(o)

    # Dense layer
    a1 = Dense(64, activation=act, name='dense')(o)
    o = BatchNormalization(name='dense_norm')(a1)
    o = Dropout(0.5, name='dropout')(o)

    # Predictions
    pred_layer = Dense(num_classes, activation='softmax', name='pred')(o)

    # Print network summary
    Model(inputs=x, outputs=pred_layer).summary()

    return Model(inputs=x, outputs=pred_layer)


def deep_cnn2(features_shape, num_classes, act='relu'):
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x

    # Block 1
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block1_conv', input_shape=features_shape)(o)
    o = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block1_pool')(o)
    o = BatchNormalization(name='block1_norm')(o)

    # Block 2
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block2_conv')(o)
    o = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(o)
    o = BatchNormalization(name='block2_norm')(o)

    # Block 3
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block3_conv')(o)
    o = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(o)
    o = BatchNormalization(name='block3_norm')(o)

    # Flatten
    o = Flatten(name='flatten')(o)

    # Dense layer
    a1 = Dense(64, activation=act, name='dense')(o)
    o = BatchNormalization(name='dense_norm')(a1)
    o = Dropout(0.5, name='dropout')(o)

    # Predictions
    pred_layer = Dense(num_classes, activation='softmax', name='pred')(o)

    # Print network summary
    Model(inputs=x, outputs=[a1, pred_layer]).summary()

    return Model(inputs=x, outputs=[a1, pred_layer])


def AlexNet_cnn(features_shape, num_classes, act='relu'):
    # Layer 1
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x

    # Block 1
    o = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=features_shape)(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(o)

    # Block 2
    o = Conv2D(96, (3, 3), activation='relu', padding='same', strides=1, name='block2_conv')(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(o)

    # Block 3
    o = Conv2D(192, (3, 3), activation='relu', padding='same', strides=1, name='block3_conv')(o)

    # Block 4
    o = Conv2D(192, (3, 3), activation='relu', padding='same', strides=1, name='block4_conv')(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(o)

    # Flatten Block 5
    o = Flatten(name='flatten')(o)

    # Dense layer 1
    a1 = Dense(512, activation='tanh', name='dense_1')(o)
    o = Dropout(0.5, name='dropout_1')(a1)

    # Dense layer 2
    a2 = Dense(256, activation='tanh', name='dense_2')(o)
    o = Dropout(0.5, name='dropout_2')(a2)

    # Predictions last block
    pred_layer = Dense(num_classes, activation='softmax', name='pred')(o)

    # Print network summary
    Model(inputs=x, outputs=[a1, a2, pred_layer]).summary()

    return Model(inputs=x, outputs=[a1, a2, pred_layer])

def AlexNet_cnn2(features_shape, num_classes, act='relu'):
    # Layer 1
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x

    # Block 1
    o = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=features_shape)(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(o)

    # Block 2
    o = Conv2D(96, (3, 3), activation='relu', padding='same', strides=1, name='block2_conv')(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(o)

    # Block 3
    o = Conv2D(192, (3, 3), activation='relu', padding='same', strides=1, name='block3_conv')(o)

    # Block 4
    o = Conv2D(192, (3, 3), activation='relu', padding='same', strides=1, name='block4_conv')(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(o)

    # Flatten Block 5
    o = Flatten(name='flatten')(o)

    # Dense layer 1
    a1 = Dense(512, activation='tanh', name='dense_1')(o)
    o = Dropout(0.5, name='dropout_1')(a1)

    # Dense layer 2
    a2 = Dense(256, activation='tanh', name='dense_2')(o)
    o = Dropout(0.5, name='dropout_2')(a2)

    # Predictions last block
    pred_layer = Dense(num_classes, activation='softmax', name='pred')(o)

    # Print network summary
    Model(inputs=x, outputs=[pred_layer]).summary()

    return Model(inputs=x, outputs=[pred_layer])

def AlexNet_cnn3(features_shape, num_classes, act='relu'):
    # Layer 1
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x

    # Block 1
    con1 = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=features_shape)(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(con1)

    # Block 2
    con2 = Conv2D(96, (3, 3), activation='relu', padding='same', strides=1, name='block2_conv')(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(con2)

    # Block 3
    con3 = Conv2D(192, (3, 3), activation='relu', padding='same', strides=1, name='block3_conv')(o)

    # Block 4
    con4 = Conv2D(192, (3, 3), activation='relu', padding='same', strides=1, name='block4_conv')(con3)
    o = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(con4)

    # Flatten Block 5
    o = Flatten(name='flatten')(o)

    # Dense layer 1
    a1 = Dense(512, activation='tanh', name='dense_1')(o)
    o = Dropout(0.5, name='dropout_1')(a1)

    # Dense layer 2
    a2 = Dense(256, activation='tanh', name='dense_2')(o)
    o = Dropout(0.5, name='dropout_2')(a2)

    # Predictions last block
    pred_layer = Dense(num_classes, activation='softmax', name='pred')(o)

    # Print network summary
    Model(inputs=x, outputs=[con1,con2,con3,con4,a1, a2, pred_layer]).summary()

    return Model(inputs=x, outputs=[con1,con2,con3,con4,a1, a2, pred_layer])


def VGG19(features_shape, num_classes, act='relu'): # Block 1
    cifar_input = Input(name='inputs', shape=features_shape, dtype='float32')
    # cifar_input = x

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=features_shape)(cifar_input)#3

    x = Conv2D(64, (3, 3),#4
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)#5

    # Block 2
    x = Conv2D(128, (3, 3),#6
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),#7
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)#8

    # Block 3
    x = Conv2D(256, (3, 3),#9
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),#10
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),#11
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), #12
                      activation='relu',
                      padding='same',
                      name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)#13

    # Block 4
    x = Conv2D(512, (3, 3), #14
                       activation='relu',
                       padding='same',
                       name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block4_conv3')(x)
    x = Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x) #17

    # Block 5
    x = Conv2D(512, (3, 3), #18
                       activation='relu',
                       padding='same',
                       name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block5_conv3')(x)
    x = Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x) #22


        # Classification block
    x = Flatten(name='flatten')(x) #23
    x = Dense(4096, activation='relu', name='fc1')(x) #24
    x = Dense(4096, activation='relu', name='fc2')(x) #25
    pred_layer = Dense(num_classes, activation='softmax', name='pred')(x) #26

    # Print network summary
    Model(inputs = cifar_input, outputs =[pred_layer]).summary()

    return Model(inputs=cifar_input, outputs=[pred_layer])

def VGG19_2(features_shape, num_classes, act='relu'): # Block 1
    cifar_input = Input(name='inputs', shape=features_shape, dtype='float32')
    # cifar_input = x

    c1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=features_shape)(cifar_input)

    c2 = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(c1)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(c2)

    # Block 2
    c3 = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    c4 = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(c3)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(c4)

    # Block 3
    c5 = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    c6 = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(c5)
    c7 = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(c6)
    c8 = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv4')(c7)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(c8)

    # Block 4
    c9 = Conv2D(512, (3, 3), #14
                       activation='relu',
                       padding='same',
                       name='block4_conv1')(x)
    c10 = Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block4_conv2')(c9)
    c11 = Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block4_conv3')(c10)
    c12 = Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block4_conv4')(c11)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(c12) #17

    # Block 5
    c13 = Conv2D(512, (3, 3), #18
                       activation='relu',
                       padding='same',
                       name='block5_conv1')(x)
    c14 = Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block5_conv2')(c13)
    c15 = Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block5_conv3')(c14)
    c16 = Conv2D(512, (3, 3),
                       activation='relu',
                       padding='same',
                       name='block5_conv4')(c15)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(c16) #22


        # Classification block
    x = Flatten(name='flatten')(x) #23
    d1 = Dense(4096, activation='relu', name='fc1')(x) #24
    d2 = Dense(4096, activation='relu', name='fc2')(d1) #25
    pred_layer = Dense(num_classes, activation='softmax', name='pred')(d2) #26

    # Print network summary
    Model(inputs = cifar_input, outputs =[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,d1,d2,pred_layer]).summary()

    return Model(inputs=cifar_input, outputs=[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,d1,d2,pred_layer])