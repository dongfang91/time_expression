import numpy as np
import os
import math
import h5py

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

from keras.callbacks import ModelCheckpoint

def hot_vectors2class_index (labels):
    examples = list()
    for instance in labels:
        label_index = list()
        for label in instance:
            k = list(label).index(1)
            label_index.append(k)
        examples.append(label_index)
    return examples

def counterList2Dict (counter_list):
    dict_new = dict()
    for item in counter_list:
        dict_new[item[0]]=item[1]
    return dict_new

def create_class_weight(labels,mu=0.5):
    n_softmax = labels.shape[-1]
    class_index = hot_vectors2class_index(labels)
    counts = np.zeros(n_softmax, dtype='int32')
    for softmax_index in class_index:
        softmax_index = np.asarray(softmax_index)
        for i in range(n_softmax):
            counts[i] = counts[i] + np.count_nonzero(softmax_index==i)

    labels_dict = counterList2Dict(list(enumerate(counts, 0)))

    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        if not labels_dict[key] == 0:
            score = math.log(mu*total/float(labels_dict[key]))
            class_weight[key] = score if score > 1.0 else 1.0
        else:
            class_weight[key] = 1.0

    return class_weight

def get_sample_weights_multiclass(labels):
    class_weight = create_class_weight(labels,mu=0.5)
    class_index = np.asarray(hot_vectors2class_index(labels))
    samples_weights = list()
    for instance in class_index:
        sample_weights = [class_weight[category] for category in instance]
        samples_weights.append(sample_weights)
    return np.asarray(samples_weights)



def load_input(filename):
    with h5py.File('data/' + filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        x = hf.get('input')
        y = hf.get('output')
        x_data = np.array(x)
        # n_patterns = x_data.shape[0]
        y_data = np.array(y)
        # print x_data[0][1000:1100]
        # print y_data[0][1000:1100]
        # y_data = y_data.reshape(y_data.shape+(1,))
        print(x_data.shape)
        print(y_data.shape)

    del x
    del y
    return x_data, y_data


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


def get_sample_weights_binaryclass(weghtis, label):
    sample_weights = label.copy()
    for i in range(sample_weights.shape[0]):
        for j in range(sample_weights.shape[1]):
            if sample_weights[i][j] == 1:
                sample_weights[i][j] = weghtis
            else:
                sample_weights[i][j] = 1
    print("sample_weights:", sample_weights[0][0:30])
    return sample_weights


def trainging_4features_sigmoid_softmax(storage, classweights, exp, char_x, pos_x, unicate_x, vocab_x, trainy_sigmoid,
                                        char_x_cv, pos_x_cv, unicate_x_cv, vocab_x_cv, cv_y_softmax, batchsize,
                                        epoch_size, n_char, n_pos, n_unicate, n_vocab, reload=False, modelpath=None,
                                        embedding_size_char=64, embedding_size_pos=32, embedding_size_unicate=8,
                                        embedding_size_vocab=16, gru_size=128):
    seq_length = char_x.shape[1]
    type_size_sigmoid = trainy_sigmoid.shape[-1]
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    if not os.path.exists(storage):
        os.makedirs(storage)
    if reload == False:

        char_input = Input(shape=(seq_length,), dtype='float32', name='character')
        char_em = Embedding(output_dim=embedding_size_char, input_dim=n_char, input_length=seq_length,
                            W_regularizer=l1(.01), mask_zero=True, dropout=0.12)(char_input)

        pos_input = Input(shape=(seq_length,), dtype='float32', name='pos')
        pos_em = Embedding(output_dim=embedding_size_pos, input_dim=n_pos, input_length=seq_length,
                           W_regularizer=l1(.01), mask_zero=True, dropout=0.1)(pos_input)

        unicate_input = Input(shape=(seq_length,), dtype='float32', name='unicate')
        unicate_em = Embedding(output_dim=embedding_size_unicate, input_dim=n_unicate, input_length=seq_length,
                            W_regularizer=l1(.01), mask_zero=True, dropout=0.1)(unicate_input)

        vocab_input = Input(shape=(seq_length,), dtype='float32', name='vocab')
        vocab_em = Embedding(output_dim=embedding_size_vocab, input_dim=n_vocab, input_length=seq_length,
                            W_regularizer=l1(.01), mask_zero=True, dropout=0.05)(vocab_input)

        input_merge = merge([char_em,pos_em,unicate_em,vocab_em], mode='concat')

        gru_out_1 = Bidirectional(GRU(gru_size, input_shape=(seq_length, embedding_size_char + embedding_size_pos + embedding_size_unicate+ embedding_size_vocab),
                                      return_sequences=True))(input_merge)

        gru_out_2 = GRU(gru_size, return_sequences=True)(gru_out_1)


        relu_size = 256
        relu_layer = TimeDistributed(Dense(relu_size,activation='relu', W_regularizer=l1(.01)))(
            gru_out_2)

        sigmoid_output = TimeDistributed(Dense(type_size_sigmoid, activation='softmax', W_regularizer=l1(.01)))(
            relu_layer)

        model = Model(input=[char_input, pos_input,unicate_input,vocab_input], output=sigmoid_output)

        model.compile(optimizer=rmsprop,
                      loss='categorical_crossentropy',
                      metrics=['fmeasure', 'precision', 'recall', 'accuracy'],sample_weight_mode = "temporal")

        # and trained it via:

    else:
        model = load_model(storage + modelpath)

    filepath = storage + "/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='fmeasure', verbose=1, save_best_only=False, mode='max')
    csv_logger = CSVLogger('training_%s.csv' % exp)
    callbacks_list = [checkpoint, csv_logger]  # ,lrate]

    hist = model.fit({'character': char_x, 'pos': pos_x,'unicate': unicate_x , 'vocab': vocab_x },
                     trainy_sigmoid, nb_epoch=epoch_size,
                     batch_size=batchsize, callbacks=callbacks_list,
                     validation_data=({'character': char_x_cv, 'pos': pos_x_cv,'unicate': unicate_x_cv , 'vocab': vocab_x_cv },
                                      cv_y_softmax), class_weight=None, sample_weight=classweights)  # None)
    model.save(storage + '/model_result.hdf5')
    np.save(storage + '/epoch_history.npy', hist.history)


training_file = "traing_one_zero"
data_x, data_y = load_input(training_file)

char_x = data_x[10:]
char_x_cv = data_x[0:10]
trainy = data_y[10:]
# cv_y = data_y[0:10]

sample_weights1 = get_sample_weights_binaryclass(15, trainy)
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

softmax_labels, sigmoid_labels = load_input("labels_softmax_sigmoid_main")
#
trainy_softmax = softmax_labels[10:]
#
#
cv_y_softmax = softmax_labels[0:10]

n_pos = 46
n_char = 83
n_unicate = 14
n_vocab = 16
epoch_size = 800
batchsize = 10
# path = "experiment/"
path = "/gsfs1/xdisk/dongfangxu9/time_expression/"
exp = "exp_allfeatures_softmax1_non_operator"

sample_weights = get_sample_weights_multiclass(trainy_softmax)

print sample_weights1.shape
print sample_weights.shape

storage = path + exp

trainging_4features_sigmoid_softmax(storage, sample_weights, exp, char_x, trainx_pos, trainx_unicate, trainx_vocab,
                                    trainy_softmax,
                                    char_x_cv, pos_x_cv, cv_x_unicate, cv_x_vocab, cv_y_softmax, batchsize, epoch_size,
                                    n_char, n_pos, n_unicate, n_vocab,
                                    reload=False, modelpath=None, embedding_size_char=64, embedding_size_pos=32,
                                    embedding_size_unicate=8, embedding_size_vocab=16, gru_size=128)
