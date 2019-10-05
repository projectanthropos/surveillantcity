# THIS SCRIPT WILL CREATE THE NEURAL NETWORK MODEL PROPSOED IN THIS PAPER:
# NELEC at SemEval-2019 Task 3: Think Twice Before Going Deep
# https://arxiv.org/pdf/1904.03223.pdf

# IMPORT PACKAGES
from keras.models import Model
from keras.layers import Bidirectional, LSTM, GRU, Embedding, Input, GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate,Dropout,Dense, Conv1D, GaussianNoise
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.regularizers import l1_l2
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from kutilities.callbacks import MetricsCallback
from sklearn.metrics import classification_report
import os
import PreProcessing as pp
import utils
import numpy as np


embedding_matrix, vocab_size, result_trn, result_val, result_tst = pp.get_embedding_matrix()

#==========================================
# Name: construct_network
# Purpose: DEFINE NETWORK ARCHITECTURE
# Input Parameter(s): NONE
# Return Value(s): model --- THE DEFINED NETWORK MODEL
#============================================
def construct_network():

    #  SET UP NETWORK INPUT
    trn1_input = Input(shape=(pp.MAX_SEQUENCE_LENGTH,), dtype='int32')
    trn2_input = Input(shape=(pp.MAX_SEQUENCE_LENGTH,), dtype='int32')
    trn3_input = Input(shape=(pp.MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_input_dim = embedding_matrix.shape[0]
    embedding_output_dim = embedding_matrix.shape[1]

    # EMBEDDING LAYER
    embeddingLayer1 = Embedding(embedding_input_dim,
                               embedding_output_dim,
                               weights=[embedding_matrix],
                               input_length=pp.MAX_SEQUENCE_LENGTH,
                               trainable=False)

    embeddingLayer2 = Embedding(embedding_input_dim,
                                embedding_output_dim,
                                weights=[embedding_matrix],
                                input_length=pp.MAX_SEQUENCE_LENGTH,
                                trainable=False)

    embeddingLayer3 = Embedding(embedding_input_dim,
                                embedding_output_dim,
                                weights=[embedding_matrix],
                                input_length=pp.MAX_SEQUENCE_LENGTH,
                                trainable=False)

    trn1_branch = embeddingLayer1(trn1_input)
    trn2_branch = embeddingLayer2(trn2_input)
    trn3_branch = embeddingLayer3(trn3_input)

    #trn1_branch = GaussianNoise(stddev=0.1)(trn1_branch)
    #trn2_branch = GaussianNoise(stddev=0.1)(trn2_branch)
    #trn3_branch = GaussianNoise(stddev=0.1)(trn3_branch)

    # LSTM LAYER
    lstm1 = Bidirectional(LSTM(256, return_sequences=True,
                           dropout=0.2, recurrent_dropout=0.2,
                           bias_regularizer=l1_l2(0.01, 0.01),
                           recurrent_regularizer=l1_l2(0.01, 0.01),
                           ))

    lstm2 = Bidirectional(LSTM(512, return_sequences=True,
                               dropout=0.2, recurrent_dropout=0.2,
                               bias_regularizer=l1_l2(0.01, 0.01),
                               recurrent_regularizer=l1_l2(0.01, 0.01),
                               ))

    lstm3 = Bidirectional(LSTM(256, return_sequences=True,
                               dropout=0.2, recurrent_dropout=0.2,
                               bias_regularizer=l1_l2(0.01, 0.01),
                               recurrent_regularizer=l1_l2(0.01, 0.01),
                               ))


    trn1_branch = lstm1(trn1_branch)
    trn2_branch = lstm2(trn2_branch)
    trn3_branch = lstm3(trn3_branch)

    # COMBINE LAYERS
    comb_layer = concatenate([trn1_branch, trn2_branch, trn3_branch])

    # LSTM LAYER
    lstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))

    comb_layer = lstm(comb_layer)

    # CNN LAYER
    #cnn_layer = Conv1D(256,kernel_size=1, strides=1, padding='same')(comb_layer)

    # POOLING LAYER
    avg_pool_1 = GlobalAveragePooling1D()(comb_layer)
    max_pool_1 = GlobalMaxPooling1D()(comb_layer)
    feature_comb_layer = concatenate([avg_pool_1, max_pool_1])
    feature_comb_layer = Dropout(0.2)(feature_comb_layer)

    feature_comb_layer = Dense(128, activation="relu")(feature_comb_layer)
    feature_comb_layer = Dropout(0.2)(feature_comb_layer)
    output = Dense(pp.NUM_CLASSES, activation='softmax')(feature_comb_layer)


    # CREATE NETWORK MODEL
    model = Model(inputs=[trn1_input, trn2_input, trn3_input], outputs=output)

    # PLOT THE NETWORK MODEL
    plot_model(model, to_file='model.png', show_shapes=True)

    print('Neural network model has been constructed!')
    return model


#==========================================
# Name: train_model
# Purpose: COMPILE MODEL AND ACTUALLY TRAIN MODEL
# Input Parameter(s): NONE
# Return Value(s): NONE
#============================================
def train_model():

    # COMPILE MODEL
    model = construct_network()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # Get DATASET
    t, result_trn, result_val, result_tst = pp.convert_text2sequences()

    # PREPARE TRAINING
    datasets = {}
    datasets["val"] = [[result_val[1], result_val[2], result_val[3]], np.array(result_val[4])]
    datasets["test"] = [[result_tst[1], result_tst[2], result_tst[3]], np.array(result_tst[4])]

    metrics_callback = MetricsCallback(datasets=datasets, metrics=utils.metrics)

    save_path = "model/EmotionContext_{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(save_path, monitor='val_acc', save_best_only=True, save_weights_only=False, mode='auto', period=1)
    tensorboardCallback = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

    # SET UP CLASS WEIGHT FOR EACH CLASS
    class_weight = {0: 0.25,
                    1: 0.25,
                    2: 0.25,
                    3: 0.25}
    total = 0
    for i in range(4):
        class_weight[i] = (len(result_trn[4]) + len(result_tst[4])) / (
                    np.sum(np.argmax(result_trn[4], 1) == i) + np.sum(np.argmax(result_tst[4], 1) == i))
        total += class_weight[i]

    for i in range(4):
        class_weight[i] /= total
    print(class_weight)


    # TRAIN MODEL
    history = model.fit([result_trn[1], result_trn[2], result_trn[3]], np.array(result_trn[4]),
                        class_weight=class_weight,
                        callbacks=[metrics_callback, checkpoint, tensorboardCallback],
                        validation_data=([result_val[1], result_val[2], result_val[3]],np.array(result_val[4])),
                        epochs=100,
                        batch_size=512)

    #################################################
    ####### DRAW FIGURES FOR TRAINING PROCESS #######
    #################################################
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_ac = history.history['acc']
    val_ac = history.history['val_acc']

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
# Name: check_performance
# Purpose: CALL TRAINED MODEL AND APPLY IT ON TEST DATA
# Input Parameter(s): NONE
# Return Value(s): NONE
#============================================
def check_performance():
    # Get DATASET
    t, result_trn, result_val, result_tst = pp.convert_text2sequences()

    # CALL TRAINED MODEL
    model_file = os.listdir('model/')
    model = load_model('model/'+ model_file[-1])

    # MAKE PREDICTION
    y_pred = model.predict([result_tst[1], result_tst[2], result_tst[3]])

    # PRINT THE PERFORMANCE METRICS
    for title, metric in utils.metrics.items():
        print(title, metric(result_tst[4].argmax(axis=1), y_pred.argmax(axis=1)))
    print(classification_report(result_tst[4].argmax(axis=1), y_pred.argmax(axis=1)))

if __name__ == '__main__':
    train_model()
    check_performance()
    pass














