import tensorflow as tf
import keras
import math
from models import VGG19_2
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from kerassurgeon.operations import delete_channels


NUM_CLASSES = 10
EPOCHS = 75
BATCH = 32
INPUT_SHAPE = (32, 32, 3)
NUMBER_OF_SAMPLES = 200   # out of 10000
TRESH_HOLD = 50


# RGB
cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0  # To make the data between 0~1
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

x_train, x_validation = x_train[0:40000], x_train[40000:50000]
y_train, y_validation = y_train[0:40000], y_train[40000:50000]
x_test_light = x_test[0:NUMBER_OF_SAMPLES]
print(x_train.shape, 'train samples')
print(y_train.shape, 'train labels')
print(x_validation.shape, 'validation samples')
print(y_validation.shape, 'validation labels')
print(x_test.shape, 'test samples')
print(y_test.shape, 'train labels')
print(x_train[0].shape)


opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

model_origin = VGG19_2(INPUT_SHAPE, NUM_CLASSES)
model_origin.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model_origin.load_weights('weights_path.h5')
total_params = model_origin.count_params()
total_filters = 2*64 + 2*128 + 4*256 + 4*512 + 4*512

"""Pruning with sum sorting algorithm"""
after_prun_pararms_vec = []
precent_pruned_params_vec = []
precent_pruned_filters_vec = []
acc_score_vec = []

for precent in range(100, 55, -5):
    total_pruned_filters = 0
    model = model_origin
    block_precentage_pruning = [0, 0, 0, precent/100, precent/100]

    i = 0
    vec = []
    for layer in model.layers:
        name = str(layer)
        if 'Conv2D' in name:
            vec.append(i)
        i = i+1
    layer_number = iter(vec)

    #  The layer we prun in the weights matrix. - you need to check what layer to prun in weights = model3.get_weights()
    conv_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    Prun_1 = math.floor(block_precentage_pruning[0]*64)
    Prun_2 = math.floor(block_precentage_pruning[1]*128)
    Prun_3 = math.floor(block_precentage_pruning[2]*256)
    Prun_4 = math.floor(block_precentage_pruning[3]*512)
    Prun_5 = math.floor(block_precentage_pruning[4]*512)

    prun_vec = []
    for x in range(0, 2):
        prun_vec.append(Prun_1)
    for x in range(2, 4):
        prun_vec.append(Prun_2)
    for x in range(4, 8):
        prun_vec.append(Prun_3)
    for x in range(8, 12):
        prun_vec.append(Prun_4)
    for x in range(12, 16):
        prun_vec.append(Prun_5)


    prun_list = iter(prun_vec)

    weights = model.get_weights()

    idx = 0
    for x in conv_layers:
        curr_prun_number = next(prun_list)
        layer_number_iter = next(layer_number)
        if curr_prun_number == 0:
            idx = idx+1
            continue


        prediction = model.predict(x_test_light)
        layer_shape = weights[x].shape
        output_layer = []
        neuron_num = layer_shape[-1]
        identical_counter = np.zeros((neuron_num, neuron_num), dtype=int)

        for i in range(0, NUMBER_OF_SAMPLES):  # comparing all samples
            for j in range(0, neuron_num):
                for k in range(j+1, neuron_num):
                    if np.sum(abs(prediction[idx][i, :, :, j] - prediction[idx][i, :, :, k])) < TRESH_HOLD:
                        identical_counter[j][k] = identical_counter[j][k] + 1

        treshhold_pruning = (precent / 100) * NUMBER_OF_SAMPLES
        filters_to_prun = np.zeros((512 * 512, 2))
        PrunedNeurons = np.zeros((512 * 512, 1), dtype=int)
        idx_2 = 0
        for j in range(0, neuron_num):
            for k in range(j+1, neuron_num):
                if identical_counter[j][k] > treshhold_pruning:
                    filters_to_prun[idx_2] = (j, k)
                    PrunedNeurons[idx_2] = k
                    idx_2 = idx_2 + 1
        PrunedNeurons = np.unique(PrunedNeurons)
        PrunedNeurons = np.delete(PrunedNeurons, 0, 0)
        layer = model.layers[layer_number_iter]
        model = delete_channels(model, layer, PrunedNeurons)
        model_origin.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        weights = model.get_weights()
        layer_shape_after = weights[x].shape
        neurons_after = layer_shape_after[-1]
        total_pruned_filters = total_pruned_filters + (neuron_num - neurons_after)
        idx = idx + 1
    after_prun_pararms = model.count_params()
    after_prun_pararms_vec.append(after_prun_pararms)
    precent_pruned_params = (total_params - after_prun_pararms) / total_params * 100
    precent_pruned_params_vec.append(precent_pruned_params)
    precent_pruned_filters = total_pruned_filters / total_filters * 100
    precent_pruned_filters_vec.append(precent_pruned_filters)
    scores = model.predict(x_test)
    y_pred = np.argmax(scores[18], axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc_score = accuracy_score(y_true, y_pred)
    acc_score_vec.append(acc_score)
    print('Accuracy:')
    print(acc_score)
plt.plot(precent_pruned_params_vec, acc_score_vec)
plt.ylabel('Accuracy')
plt.xlabel('prunning of network params %')
plt.show()
plt.plot(precent_pruned_filters_vec, acc_score_vec)
plt.ylabel('Accuracy')
plt.xlabel('prunning of network filters %')
plt.show()

f = open('Results_path.txt', "a")
f.write("{} \n\n{} \n\n{} \n\n{} \n\n".format(after_prun_pararms_vec, precent_pruned_params_vec, precent_pruned_filters_vec, acc_score_vec))
f.close()




