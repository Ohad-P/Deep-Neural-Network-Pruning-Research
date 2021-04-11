import tensorflow as tf
import keras
import math
from models import AlexNet_cnn2 , AlexNet_cnn, AlexNet_cnn3, VGG19,VGG19_2
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from kerassurgeon.operations import  delete_channels
# The path of the stored model
filepath = 'model.h5'

NUM_CLASSES=10
EPOCHS = 75
BATCH = 32
# learningRate = 0.000001
INPUT_SHAPE = (32,32,3)






#RGB
cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0 #To make the data between 0~1
y_train=to_categorical(y_train,num_classes=10)
y_test=to_categorical(y_test,num_classes=10)

x_train, x_validation = x_train[0:40000],x_train[40000:50000]
y_train, y_validation = y_train[0:40000],y_train[40000:50000]

print(x_train.shape, 'train samples')
print(y_train.shape, 'train labels')
print(x_validation.shape, 'validation samples')
print(y_validation.shape, 'validation labels')
print(x_test.shape, 'test samples')
print(y_test.shape, 'train labels')
print(x_train[0].shape)

#reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, mode='auto')
early = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, patience=5, mode='auto')
#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
# opt = keras.optimizers.Adam(learning_rate=0.01)
# plot_model
#model = VGG19(INPUT_SHAPE, NUM_CLASSES)
#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#for layer in model.layers[14:22]: # setting blocks 4-5 to be untrainable
#    layer.trainable = False
#model.load_weights('Weights_Path.h5')
#model.fit(x_train,y_train,epochs=EPOCHS,callbacks=[early])
#model.save_weights('Weights_Path.h5')

#test_net=model.evaluate(x_test,y_test,verbose=1)
#score = model.evaluate(x_test, y_test)

#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#for layer in model.layers[14:22]: # setting blocks 4-5 to be untrainable
#    layer.trainable = True
#model.fit(x_train,y_train,epochs=EPOCHS,callbacks=[early])
#model.save_weights('Weights_Path.h5')

#test_net=model.evaluate(x_test,y_test,verbose=1)
#score = model.evaluate(x_test, y_test)

#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

model_origin = VGG19_2(INPUT_SHAPE, NUM_CLASSES)
model_origin.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model_origin.load_weights('Weights_Path.h5')
total_params = model_origin.count_params()
total_filters = 2*64 + 2*128 + 4*256 + 4*512 + 4*512

"""Pruning with sum sorting algorithm"""
after_prun_pararms_vec = []
precent_pruned_params_vec = []
precent_pruned_filters_vec = []
acc_score_vec = []

for precent in range(0, 70, 5):
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

    conv_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]  # The layer we prun in the weights matrix. - you need to check what layer to prun in  weights = model3.get_weights()
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

    total_pruned_filters = 0
    for prun_num in prun_vec:
        total_pruned_filters += prun_num

    prun_list = iter(prun_vec)

    #scores = model.predict(x_test)
    #y_pred = np.argmax(scores[18], axis=1)
    #y_true = np.argmax(y_test,axis=1)
    #acc_score = accuracy_score(y_true, y_pred)
    #print('Accuracy before pruning filters:')
    #print(acc_score)

    weights = model.get_weights()
    prediction = model.predict(x_test)
    for x in conv_layers:
        curr_prun_number = next(prun_list)
        layer_number_iter = next(layer_number)
        if curr_prun_number == 0:
            continue
        layer_shape = weights[x].shape
        kernels_amount=layer_shape[-1]  # the last element is the number of kernels
        layer_input_size = 1
        for i in layer_shape[:-1]:  # computing kernel size
            layer_input_size *= i

        curr_filter = np.zeros((kernels_amount, layer_input_size), dtype=float)
        for i in range(0, kernels_amount):
            flat = weights[x][:, :, :, i].flatten()
            curr_filter[i] = flat[:]
            curr_filter = abs(curr_filter)
        sum_filter = np.sum(curr_filter, axis=1)
        filter_argsort = np.argsort(sum_filter)
        delete_filters = np.zeros((curr_prun_number, 1), dtype=int)
        for i in range(0, curr_prun_number):
            delete_filters[i] = filter_argsort[i]

        layer = model.layers[layer_number_iter]
        delete_filters = np.unique(delete_filters)

        model = delete_channels(model, layer, delete_filters)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

        # Calculating accuracy after prunning

    after_prun_pararms = model.count_params()
    after_prun_pararms_vec.append(after_prun_pararms)
    precent_pruned_params = (total_params-after_prun_pararms)/total_params * 100
    precent_pruned_params_vec.append(precent_pruned_params)
    precent_pruned_filters = total_pruned_filters/total_filters * 100
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

f = open('Results_Path.txt', "a")
f.write("{} \n\n{} \n\n{} \n\n{} \n\n".format(after_prun_pararms_vec, precent_pruned_params_vec, precent_pruned_filters_vec ,acc_score_vec ))
f.close()


"""****************************************************"""
"""Random Pruning :"""
"""****************************************************"""

R_after_prun_pararms_vec = []
R_precent_pruned_params_vec = []
R_precent_pruned_filters_vec = []
R_acc_score_vec = []

for precent in range(0, 70, 5):
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

    conv_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]  # The layer we prun in the weights matrix. - you need to check what layer to prun in  weights = model3.get_weights()
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

    total_pruned_filters = 0
    for prun_num in prun_vec:
        total_pruned_filters += prun_num

    prun_list = iter(prun_vec)

    for x in conv_layers:
        curr_prun_number = next(prun_list)
        layer_number_iter = next(layer_number)
        if curr_prun_number == 0:
            continue
        layer = model.layers[layer_number_iter]
        RandomPrune = np.random.permutation(np.arange(512))[:curr_prun_number]
        model = delete_channels(model, layer, RandomPrune)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

    # Calculating accuracy after prunning

    after_prun_pararms = model.count_params()
    R_after_prun_pararms_vec.append(after_prun_pararms)
    precent_pruned_params = (total_params-after_prun_pararms)/total_params * 100
    R_precent_pruned_params_vec.append(precent_pruned_params)
    precent_pruned_filters = total_pruned_filters/total_filters * 100
    R_precent_pruned_filters_vec.append(precent_pruned_filters)
    scores = model.predict(x_test)
    y_pred = np.argmax(scores[18], axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc_score = accuracy_score(y_true, y_pred)
    R_acc_score_vec.append(acc_score)
    print('Accuracy:')
    print(acc_score)

plt.plot(precent_pruned_params_vec, acc_score_vec, 'b', R_precent_pruned_params_vec, R_acc_score_vec, 'r')
plt.ylabel('Accuracy')
plt.xlabel('prunning of network params %')
plt.show()
plt.plot(precent_pruned_filters_vec, acc_score_vec, 'b', R_precent_pruned_filters_vec, R_acc_score_vec, 'r')
plt.ylabel('Accuracy')
plt.xlabel('prunning of network filters %')
plt.show()

f = open('Results_Path.txt', "a")
f.write("{} \n\n{} \n\n{} \n\n{} \n\n".format(R_after_prun_pararms_vec, R_precent_pruned_params_vec, R_precent_pruned_filters_vec ,R_acc_score_vec ))
f.close()

