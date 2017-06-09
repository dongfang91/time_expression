import numpy as np
import os
import math
import h5py
import json

from collections import OrderedDict
from collections import Iterable
import csv

from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, TimeDistributed, merge
from keras.layers import GRU, Dropout, Input
from keras.layers import Embedding
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.regularizers import l1, l2
from keras.models import Model

from keras.callbacks import ModelCheckpoint, Callback



def load_input(filename):
    with h5py.File('data/'+ filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        x1 = hf.get('char')
        x2 = hf.get('pos')
        x3 = hf.get('unic')
        x4 = hf.get('vocab')



        x_char = np.array(x1)
        x_pos = np.array(x2)
        x_unic = np.array(x3)
        x_vocab = np.array(x4)

        #n_patterns = x_data.shape[0]

        # print x_data[0][1000:1100]
        # print y_data[0][1000:1100]
        #y_data = y_data.reshape(y_data.shape+(1,))
    del x1,x2,x4,x3
    return x_char,x_pos,x_unic,x_vocab


def load_pos(filename):
    with h5py.File('data/' + filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        x = hf.get('input')

        x_data = np.array(x)
        # n_patterns = x_data.shape[0]

        # print x_data[0][1000:1100]
        # print y_data[0][1000:1100]
        # y_data = y_data.reshape(y_data.shape+(1,))
        print(x_data.shape)
    del x
    return x_data


def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def trainging(storage,exp,sampleweights,char_x,pos_x,unicate_x,vocab_x,trainy_interval,trainy_operator_ex,trainy_operator_im,
                                        char_x_cv,pos_x_cv,unicate_x_cv,vocab_x_cv,cv_y_interval,cv_y_operator_ex,cv_y_operator_im,batchsize,epoch_size,
                                        n_char,n_pos,n_unicate,n_vocab,reload = False,modelpath = None,embedding_size_char =64,
                                        embedding_size_pos = 48, embedding_size_unicate = 32,embedding_size_vocab =32,
                                        gru_size1 = 128,gru_size2 = 160,gru_size3 = 128,gru_size4 = 160,gru_size5 = 128,gru_size6 = 160):

    seq_length = char_x.shape[1]
    type_size_interval = trainy_interval.shape[-1]
    type_size_operator_ex = trainy_operator_ex.shape[-1]
    type_size_operator_im = trainy_operator_im.shape[-1]




    if not os.path.exists(storage):
        os.makedirs(storage)
    if reload ==False:

        char_input = Input(shape=(seq_length,), dtype='int8', name='character')
        char_em = Embedding(output_dim=embedding_size_char, input_dim=n_char, input_length=seq_length,
                            W_regularizer = l1(.01),mask_zero=True,dropout = 0.2)(char_input)

        pos_input = Input(shape=(seq_length,), dtype='int8', name='pos')
        pos_em = Embedding(output_dim=embedding_size_pos, input_dim=n_pos, input_length=seq_length,
                            W_regularizer=l1(.01), mask_zero=True, dropout=0.08)(pos_input)

        unicate_input = Input(shape=(seq_length,), dtype='int8', name='unicate')
        unicate_em = Embedding(output_dim=embedding_size_unicate, input_dim=n_unicate, input_length=seq_length,
                            W_regularizer=l1(.01), mask_zero=True, dropout=0.15)(unicate_input)

        vocab_input = Input(shape=(seq_length,), dtype='int8', name='vocab')
        vocab_em = Embedding(output_dim=embedding_size_vocab, input_dim=n_vocab, input_length=seq_length,
                            W_regularizer=l1(.01), mask_zero=True, dropout=0.1)(vocab_input)

        input_merge1 = merge([char_em,pos_em,unicate_em,vocab_em], mode='concat')

        gru_out_1 = Bidirectional(GRU(gru_size1, input_shape=(seq_length, embedding_size_char+embedding_size_pos+embedding_size_unicate+embedding_size_vocab),
                                      return_sequences=True))(input_merge1)

        gru_out_2 = GRU(gru_size2, return_sequences=True) (gru_out_1)

        interval_output = TimeDistributed(Dense(type_size_interval, activation='softmax',W_regularizer = l1(.01),name='timedistributed_1'))(gru_out_2)

        input_merge2 = merge([input_merge1,gru_out_2], mode='concat')

        gru_out_3 = Bidirectional(GRU(gru_size3, input_shape=(seq_length, embedding_size_char+embedding_size_pos+embedding_size_unicate+embedding_size_vocab+gru_size2),
                                      return_sequences=True))(input_merge2)

        gru_out_4 = GRU(gru_size4, return_sequences=True) (gru_out_3)

        explicit_operator = TimeDistributed(Dense(type_size_operator_ex, activation='softmax',W_regularizer = l1(.01),name='timedistributed_2'))(gru_out_4)

        input_merge3 = merge([input_merge1,gru_out_2,gru_out_4], mode='concat')
        

        gru_out_5 = Bidirectional(GRU(gru_size5, input_shape=(seq_length, embedding_size_char+embedding_size_pos+embedding_size_unicate+embedding_size_vocab+gru_size2+gru_size4),
                                      return_sequences=True))(input_merge3)

        gru_out_6 = GRU(gru_size6, return_sequences=True) (gru_out_5)

        implicit_operator = TimeDistributed(Dense(type_size_operator_im, activation='softmax', W_regularizer=l1(.01), name='timedistributed_3'))(gru_out_6)


        model = Model(input=[char_input, pos_input,unicate_input,vocab_input], output=[interval_output, explicit_operator,implicit_operator])

        model.compile(optimizer='rmsprop',
                      loss={'timedistributed_1': 'categorical_crossentropy', 'timedistributed_2': 'categorical_crossentropy','timedistributed_3': 'categorical_crossentropy'},
                      loss_weights={'timedistributed_1': 1., 'timedistributed_2': 0.7,'timedistributed_3': 0.5},metrics=['fmeasure', 'categorical_accuracy','recall','precision'],
                      sample_weight_mode="temporal")


    else:
        model = load_model(storage+modelpath)

    filepath = storage + "/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='fmeasure', verbose=1, save_best_only=False, mode='min')
    csv_logger = CSVLogger('training_%s.csv' % exp)
    callbacks_list = [checkpoint, csv_logger]


    hist = model.fit({'character': char_x, 'pos': pos_x,'unicate':unicate_x,'vocab':vocab_x},
                  {'timedistributed_1': trainy_interval, 'timedistributed_2': trainy_operator_ex,'timedistributed_3': trainy_operator_im}, nb_epoch=epoch_size,
                  batch_size=batchsize, callbacks=callbacks_list,validation_data =({'character': char_x_cv, 'pos': pos_x_cv,'unicate':unicate_x_cv,'vocab':vocab_x_cv},
                  {'timedistributed_1': cv_y_interval, 'timedistributed_2': cv_y_operator_ex,'timedistributed_3':cv_y_operator_im}),sample_weight=sampleweights)
    model.save(storage + '/model_result.hdf5')
    np.save(storage + '/epoch_history.npy', hist.history)





char_x,pos_x,unicate_x,vocab_x = load_input("training_allsentence_input_addmarks3")
trainy_interval = load_pos("training_allintervallabels3")
trainy_operator_ex = load_pos("training_allexplicitoperatorlabels3")
trainy_operator_im = load_pos("training_implicitlabels3")


char_x_cv,pos_x_cv,unicate_x_cv,vocab_x_cv = load_input("val_sentence_input_addmarks3")
cv_y_interval = load_pos("val_allintervallabels3")
cv_y_operator_ex = load_pos("val_explicitoperatorlabels3")
cv_y_operator_im = load_pos("val_implicitlabels3")


n_pos = 46
n_char = 83
n_unicate = 14
n_vocab = 16
epoch_size = 800
batchsize = 100
#path = "experiment/"
path = "/xdisk/dongfangxu9/time_expression/sentence_level/"
exp = "exp_commodel_multioutputs_joint_tuned1"

#sample_weights = get_sample_weights_multiclass(trainy_softmax)

sampleweights_interval = np.load("data/sampleweights_allintervallabels3.npy")
sampleweights_operator_ex = np.load("data/sampleweights_allexplicitoperatorlabels3.npy")
sampleweights_operator_im = np.load("data/sampleweights_implicitlabels3.npy")
sampleweights = [sampleweights_interval,sampleweights_operator_ex,sampleweights_operator_im]

#print sample_weights1.shape
#print sample_weights.shape

storage = path + exp



trainging(storage,exp,sampleweights,char_x,pos_x,unicate_x,vocab_x,trainy_interval,trainy_operator_ex,trainy_operator_im,
                                        char_x_cv,pos_x_cv,unicate_x_cv,vocab_x_cv,cv_y_interval,cv_y_operator_ex,cv_y_operator_im,batchsize,epoch_size,
                                        n_char,n_pos,n_unicate,n_vocab,reload = False,modelpath = None,embedding_size_char =64,
                                        embedding_size_pos = 32, embedding_size_unicate = 32,embedding_size_vocab =16,
                                        gru_size1 = 148,gru_size2 = 180,gru_size3 = 200,gru_size4 = 170,gru_size5 = 200,gru_size6 = 170)
