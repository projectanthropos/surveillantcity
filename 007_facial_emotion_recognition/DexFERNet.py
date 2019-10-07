# THIS SCRIPT CREATES THE NETWORK STRUCTURE FOR DEXPRESSION
# THE DETAIL CAN BE FOUND IN THIS PAPER:
# DeXpression: Deep Convolutional Neural Network for Expression Recognition
# https://arxiv.org/pdf/1509.05371v2.pdf

# IMPORT PACKAGES
import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Conv2D, ReLU, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras.layers.merge import concatenate
from keras.utils import plot_model
import matplotlib.pyplot as plt
from GenerateData import generate_data
import os


#==========================================
# Name: construct_network
# Purpose: DEFINE NETWORK ARCHITECTURE
# Input Parameter(s): NONE
# Return Value(s): model --- THE DEFINED NETWORK MODEL
#============================================
def construct_network(imagesize):
    # FIRST NET LAYER
    input = Input(shape=(imagesize,imagesize,1), dtype='float32')
    net = Conv2D(filters=128, kernel_size=7, strides=2, padding='same', activation=None)(input)
    net = ReLU()(net)
    net = MaxPooling2D(pool_size=3, strides=2, padding='same')(net)
    net = BatchNormalization()(net)

    # FEATEX #1
    net1 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(net)
    net1 = ReLU()(net1)

    net2 = MaxPooling2D(pool_size=3, strides=1, padding='same')(net)

    net3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(net1)
    net3 = ReLU()(net3)

    net4 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(net2)
    net4 = ReLU()(net4)

    # FEATEX #2
    added_first_net = concatenate([net3, net4])
    added_first_net = MaxPooling2D(pool_size=3, strides=2, padding='same')(added_first_net)

    net5 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(added_first_net)
    net5 = ReLU()(net5)

    net6 = MaxPooling2D(pool_size=3, strides=1, padding='same')(added_first_net)

    net7 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(net5)
    net7 = ReLU()(net7)

    net8 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(net6)
    net8 = ReLU()(net8)

    added_second_net = concatenate([net7, net8])
    added_second_net = MaxPooling2D(pool_size=3, strides=2, padding='same')(added_second_net)

    # FINAL NET LAYER
    final_net = Flatten()(added_second_net)
    final_net = Dropout(0.4)(final_net)
    final_net = Dense(7, activation='softmax')(final_net)

    # CREATE NETWORK MODEL
    model = Model(inputs=input, outputs=final_net)

    # PLOT THE NETWORK MODEL
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


#==========================================
# Name: optimizer
# Purpose: DEFINE DIFFERENT KINDS OF OPTIMIZERS
#          USE GRID SEARCH TO SELECT THE BEST ONE
# Input Parameter(s): learing_rate --- LEARNING RATE WITH DEFAULT VALUE AS 0.001
#                     method --- OPRIMIZER METHOD WITH DEFAULT VALUE AS 'Adam'
# Return Value(s): model_optimizer --- THE SELECTED OPTIMIZER
#============================================
def optimizer(learing_rate = 0.001, method='Adam'):
    if method == 'Adam':
        model_optimizer = optimizers.Adam(lr=learing_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if method == 'Adamax':
        model_optimizer = optimizers.Adamax(lr=learing_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

    if method == 'Nadam':
        model_optimizer = optimizers.Nadam(lr=learing_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    if method == 'SGD':
        model_optimizer = optimizers.SGD(lr=learing_rate, momentum=0.0, decay=0.0, nesterov=False)

    if method == 'RMSprop':
        model_optimizer = optimizers.RMSprop(lr=learing_rate, rho=0.9, epsilon=None, decay=0.0)

    print('Using:', method)
    return model_optimizer


#==========================================
# Name: compile_model
# Purpose: COMPILE MODEL AND SET UP THE TRAINING PROCESS
# Input Parameter(s): model --- THE BUILT MODEL
#                     selected_optm --- OPRIMIZER METHOD
#                     train_generator --- TRAINING DATA
#                     val_generator --- VALIDATION DATA
#                     eval_generator --- TEST DATA
#                     num_epoch --- NUMBER OF EPOCH WE WANT TO TRAIN
#                     batch_size --- BATCH SIZE
#                     callbacks_list --- CALLBACKS WE SET TO MONITOR TRANING PROCESS
#                     val_tag --- WHETHER WE WANT TO DO VALIDATION
# Return Value(s): NONE
#============================================
def compile_model (model, selected_optm, train_generator, val_generator, eval_generator, num_epoch, batch_size, callbacks_list, train_num, val_num, test_num, val_tag =True):

    # COMPILE MODEL
    model.compile(optimizer=selected_optm, loss='categorical_crossentropy', metrics=['accuracy'])

    # TRAIN MODEL
    if val_tag:
        history_fit = model.fit_generator(train_generator,
                        steps_per_epoch = int(train_num/batch_size),
                        epochs=num_epoch,
                        callbacks=callbacks_list,
                        validation_data= val_generator,
                        validation_steps = int(val_num/batch_size))
    else:
        if val_tag:
            history_fit = model.fit_generator(train_generator,
                                steps_per_epoch=int(train_num / batch_size),
                                epochs=num_epoch,
                               )

    # TEST MODEL
    history_predict = model.predict_generator(eval_generator, steps = int(test_num/batch_size))
    print('Model has been trained!\n')

    # WRITE TRAINING HISTORY LOG INTO DOCUMENT
    print('...... Writing to log ......\n')
    log_dir = 'dataset/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + 'model_fit_log', 'w') as f:
        f.write(str(history_fit.history))
    with open(log_dir + 'model_predict_log', 'w') as f:
        f.write(str(history_predict))
    print('...... Finish writing ......\n')

    #################################################
    ####### DRAW FIGURES FOR TRAINING PROCESS #######
    #################################################
    train_loss = history_fit.history['loss']
    val_loss = history_fit.history['val_loss']

    train_ac = history_fit.history['acc']
    val_ac = history_fit.history['val_acc']

    epochs = range(0, len(train_loss))
    if not os.path.exists('figure/'):
        os.makedirs('figure/')
    plt.figure()
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig('figure/loss.png')

    plt.figure()
    plt.plot(epochs, train_ac, label='Training Acc')
    plt.plot(epochs, val_ac, label='Validation Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig('figure/accuracy.png')

#==========================================
# Name: run_model
# Purpose: CALL FUNCTIONS DEFINED IN THIS SCRIPT
#          AND ACTUALLY TRAIN OUR MODEL
# Input Parameter(s): NONE
# Return Value(s): NONE
#============================================


if __name__ == '__main__':
    #run_model(64)
    pass











