import numpy as np
import os
import math
import h5py

from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense,TimeDistributed,merge
from keras.layers import GRU,Dropout,Input
from keras.layers import Embedding
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.regularizers import l1, l2
from keras.models import Model

from keras.callbacks import ModelCheckpoint



def load_input(filename):
    with h5py.File('data/'+ filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        x = hf.get('input')
        y = hf.get('output')
        x_data = np.array(x)
        #n_patterns = x_data.shape[0]
        y_data = np.array(y)
        # print x_data[0][1000:1100]
        # print y_data[0][1000:1100]
        #y_data = y_data.reshape(y_data.shape+(1,))
        print(x_data.shape)
        print(y_data.shape)


    del x
    del y
    return x_data,y_data

def load_pos(filename):
    with h5py.File('data/'+ filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        x = hf.get('input')

        x_data = np.array(x)
        #n_patterns = x_data.shape[0]

        # print x_data[0][1000:1100]
        # print y_data[0][1000:1100]
        #y_data = y_data.reshape(y_data.shape+(1,))
        print(x_data.shape)
    del x
    return x_data

def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def get_sample_weights(weghtis, label):
    sample_weights = label.copy()
    for i in range(sample_weights.shape[0]):
        for j in range(sample_weights.shape[1]):
            if sample_weights[i][j] == 1:
                sample_weights[i][j] = weghtis
            else:
                sample_weights[i][j] = 1
    print("sample_weights:", sample_weights[0][0:30])
    return sample_weights


def trainging_4features_sigmoid_softmax(storage,classweights,exp,char_x,pos_x,unicate_x,vocab_x,trainy_sigmoid,char_x_cv,pos_x_cv,unicate_x_cv,vocab_x_cv,cv_y_sigmoid,batchsize,epoch_size,n_char,n_pos,n_unicate,n_vocab,reload = False,modelpath = None,embedding_size_char =64,embedding_size_pos = 32, embedding_size_unicate = 8,embedding_size_vocab =16, gru_size = 128):
    seq_length = char_x.shape[1]
    type_size_sigmoid = trainy_sigmoid.shape[-1]
    rmsprop = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)



    if not os.path.exists(storage):
        os.makedirs(storage)
    if reload ==False:

        char_input = Input(shape=(seq_length,), name='character')
        char_em = Embedding(output_dim=embedding_size_char, input_dim=n_char, input_length=seq_length,
                            mask_zero=True)(char_input)

        pos_input = Input(shape=(seq_length,), name='pos')
        pos_em = Embedding(output_dim=embedding_size_pos, input_dim=n_pos, input_length=seq_length,
                            mask_zero=True)(pos_input)



        # char_input = Input(shape=(seq_length,), dtype='float32', name='character')
        # char_em = Embedding(output_dim=embedding_size_char, input_dim=n_char, input_length=seq_length,
        #                     W_regularizer=l1(.01), mask_zero=True, dropout=0.12)(char_input)
        #
        # pos_input = Input(shape=(seq_length,), dtype='float32', name='pos')
        # pos_em = Embedding(output_dim=embedding_size_pos, input_dim=n_pos, input_length=seq_length,
        #                    W_regularizer=l1(.01), mask_zero=True, dropout=0.1)(pos_input)
        #
        # unicate_input = Input(shape=(seq_length,), dtype='float32', name='unicate')
        # unicate_em = Embedding(output_dim=embedding_size_unicate, input_dim=n_unicate, input_length=seq_length,
        #                     W_regularizer=l1(.01), mask_zero=True, dropout=0.1)(unicate_input)
        #
        # vocab_input = Input(shape=(seq_length,), dtype='float32', name='vocab')
        # vocab_em = Embedding(output_dim=embedding_size_vocab, input_dim=n_vocab, input_length=seq_length,
        #                     W_regularizer=l1(.01), mask_zero=True, dropout=0.05)(vocab_input)

        #input_merge = merge([char_em,pos_em], mode='concat')
        input_merge = merge([char_em, pos_em], mode='concat')

        gru_out_1 = Bidirectional(GRU(gru_size, input_shape=(seq_length, embedding_size_char+embedding_size_pos),
                                      return_sequences=True))(input_merge)

        gru_out_2 = GRU(gru_size, return_sequences=True) (gru_out_1)

        #gru_out_3 = GRU(gru_size, return_sequences=True)(gru_out_2)

        #softmax_output = TimeDistributed(Dense(type_size_softmax, activation='softmax',W_regularizer = l1(.01),name='timedistributed_1'))(gru_out_2)

        relu_size = 128
        relu_layer = TimeDistributed(Dense(relu_size,activation='relu', W_regularizer=l1(.01)))(
            gru_out_2)

        sigmoid_output = TimeDistributed(Dense(type_size_sigmoid, activation='sigmoid',W_regularizer = l1(.01)))(relu_layer)

        model = Model(input=[char_input, pos_input], output=sigmoid_output)

        model.compile(optimizer=rmsprop,
                      loss='binary_crossentropy',
                      metrics=['fmeasure', 'precision', 'recall','accuracy'],sample_weight_mode="temporal")

        # and trained it via:

    else:
        model = load_model(storage+modelpath)

    filepath = storage + "/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='fmeasure', verbose=1, save_best_only=False, mode='max')
    csv_logger = CSVLogger('training_%s.csv' % exp)
    callbacks_list = [checkpoint, csv_logger]#,lrate]


    hist = model.fit({'character': char_x, 'pos': pos_x},
                  trainy_sigmoid, nb_epoch=epoch_size,
                  batch_size=batchsize, callbacks=callbacks_list,validation_data =({'character': char_x_cv, 'pos': pos_x_cv},
                 cv_y_sigmoid),class_weight=None,sample_weight=None)#None)
    model.save(storage + '/model_result.hdf5')
    np.save(storage + '/epoch_history.npy', hist.history)





training_file = "traing_one_zero"
data_x,data_y = load_input(training_file)


char_x = data_x[10:]
char_x_cv = data_x[0:10]
trainy = data_y[10:]
# cv_y = data_y[0:10]

sample_weights = get_sample_weights(15,trainy)
#
# trainy_sigmoid = trainy.reshape(trainy.shape + (1,))
# cv_y_sigmoid = cv_y.reshape(cv_y.shape + (1,))



pos_x = load_pos("pos_training_norm")
trainx_pos = pos_x[10:]
pos_x_cv = pos_x[0:10]

unicate_x = load_pos("unicode_category_training")
trainx_unicate = unicate_x[10:]
cv_x_unicate = unicate_x[0:10]

vocab_x = load_pos("vocab_training")
trainx_vocab = vocab_x[10:]
cv_x_vocab = vocab_x[0:10]

sigmoid_labels = load_pos("labels_sigmoid")
#
trainy_sigmoid =sigmoid_labels[10:]
#
#
cv_y_sigmoid = sigmoid_labels[0:10]

n_pos = 46
n_char = 83
n_unicate = 14
n_vocab = 16
epoch_size = 800
batchsize = 20
path = "experiment/"
#path = "/gsfs1/xdisk/dongfangxu9/time_expression/"
exp = "exp_com_sig1"

storage = path + exp

trainging_4features_sigmoid_softmax(storage,sample_weights,exp,char_x,trainx_pos,trainx_unicate,trainx_vocab,trainy_sigmoid,
                                    char_x_cv,pos_x_cv,cv_x_unicate,cv_x_vocab,cv_y_sigmoid,batchsize,epoch_size,n_char,n_pos,n_unicate,n_vocab,
                                    reload = False,modelpath = None,embedding_size_char =64,embedding_size_pos = 32, embedding_size_unicate = 8,embedding_size_vocab =16, gru_size = 128)




