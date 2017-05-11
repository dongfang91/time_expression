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

def read_from_json(filename):
    with open(filename+".txt", 'r') as outfile:
        data = json.load(outfile)
    outfile.close()
    return data


def prob2classes(prediction):
    if prediction.shape[-1] > 1:
        return prediction.argmax(axis=-1)

def calculate_performance_restricted(gold,instance_length,prediction):
    instan = len(prediction)
    n_match = 0.0
    n_pre = 0.0
    n_rec = 0.0
    for i in range(instan):
        for k in range(instance_length[i]):
            if gold[i][k] ==  prediction[i][k] and not prediction[i][k] ==0:
                n_match +=1
            if not gold[i][k] == 0:
                n_rec +=1
            if not prediction[i][k] ==0:
                n_pre+=1

    if n_pre > 0:
        precision = n_match/n_pre
    else:
        precision = 0.0
    if n_rec >0:
        recall = n_match/n_rec
    else:
        recall = 0.0
    if n_pre == 0 and n_rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    print n_match,n_pre,n_rec
    print "presion: ",precision, "recall: ",recall, "F1 score: ",f1
    return n_match,n_pre,n_rec,precision,recall,f1



def hot_vectors2class_index (labels):
    examples = list()
    for instance in labels:
        label_index = list()
        for label in instance:
            k = list(label).index(1)
            label_index.append(k)
        examples.append(label_index)
    return examples


class Perofrmance_restricted_length(Callback):
    """Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def __init__(self, location, val_data,y_label,instance_length,filename,separator=',', append=False):
        self.filepath = location
        self.val_data = val_data
        self.y_label  = y_label
        self.instance_length = instance_length
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True


    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename) as f:
                    self.append_header = bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a')
        else:
            self.csv_file = open(self.filename, 'w')

    def on_epoch_end(self, epoch, logs=None):

        filepath = self.filepath.format(epoch=epoch, **logs)
        self.model = load_model(filepath)
        y_predict = self.model.predict(self.val_data)
        classes = prob2classes(y_predict)
        n_match, n_pre, n_rec, precision, recall, f1 = calculate_performance_restricted(self.y_label, self.instance_length, classes)
        performance = {"match":n_match, "tagged":n_pre,"gold":n_rec,"precision":precision,"recall":recall,"f1":f1}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(performance.keys())
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + self.keys)
            if self.append_header:
                self.writer.writeheader()


        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(performance[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

        def on_train_end(self, logs=None):
            self.csv_file.close()




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
        print x_char.shape, x_pos.shape,
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


def trainging_4features_multiclass(instance_length,storage, classweights, exp, char_x, pos_x, unicate_x, vocab_x, train_y,
                                        char_x_cv, pos_x_cv, unicate_x_cv, vocab_x_cv, cv_y, batchsize,
                                        epoch_size, n_char, n_pos, n_unicate, n_vocab, reload=False, modelpath=None,
                                        embedding_size_char=64, embedding_size_pos=32, embedding_size_unicate=8,
                                        embedding_size_vocab=16, gru_size=128):

    char_embedding = np.load("data/training_sentence/char_embedding.npy")
    pos_embedding = np.load("data/training_sentence/pos_embedding.npy")
    unic_embedding = np.load("data/training_sentence/unic_embedding.npy")

    seq_length = char_x.shape[1]
    type_size_sigmoid = train_y.shape[-1]
    rmsprop = RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.0)

    y_label = hot_vectors2class_index(cv_y)

    if not os.path.exists(storage):
        os.makedirs(storage)
    if reload == False:

        char_input = Input(shape=(seq_length,), dtype='float32', name='character')
        char_em = Embedding(output_dim=embedding_size_char,weights=[char_embedding], input_dim=n_char, input_length=seq_length,
                            W_regularizer=l1(.01), mask_zero=True, dropout=0.12)(char_input)

        pos_input = Input(shape=(seq_length,), dtype='float32', name='pos')
        pos_em = Embedding(output_dim=embedding_size_pos, weights=[pos_embedding],input_dim=n_pos, input_length=seq_length,
                           W_regularizer=l1(.01), mask_zero=True, dropout=0.1)(pos_input)

        unicate_input = Input(shape=(seq_length,), dtype='float32', name='unicate')
        unicate_em = Embedding(output_dim=embedding_size_unicate, weights=[unic_embedding],input_dim=n_unicate, input_length=seq_length,
                            W_regularizer=l1(.01), mask_zero=True, dropout=0.1)(unicate_input)

        vocab_input = Input(shape=(seq_length,), dtype='float32', name='vocab')
        vocab_em = Embedding(output_dim=embedding_size_vocab, input_dim=n_vocab, input_length=seq_length,
                            W_regularizer=l1(.01), mask_zero=True, dropout=0.05)(vocab_input)

        input_merge = merge([char_em,pos_em,unicate_em,vocab_em], mode='concat')

        gru_out_1 = Bidirectional(GRU(gru_size, input_shape=(seq_length, embedding_size_char + embedding_size_pos + embedding_size_unicate+ embedding_size_vocab),
                                      return_sequences=True))(input_merge)

        gru_out_2 = GRU(gru_size, return_sequences=True)(gru_out_1)


        # relu_size = 128
        # relu_layer = TimeDistributed(Dense(relu_size,activation='relu', W_regularizer=l1(.01)))(
        #     gru_out_2)

        sigmoid_output = TimeDistributed(Dense(type_size_sigmoid, activation='softmax', W_regularizer=l1(.01)))(
            gru_out_2)

        model = Model(input=[char_input, pos_input,unicate_input,vocab_input], output=sigmoid_output)

        model.compile(optimizer=rmsprop,
                      loss='categorical_crossentropy',
                      metrics=['fmeasure', 'precision', 'recall', 'categorical_accuracy'],sample_weight_mode = "temporal")

        # and trained it via:

    else:
        model = load_model(modelpath)

    filepath = storage + "/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='fmeasure', verbose=1, save_best_only=False, mode='max')
    csv_logger = CSVLogger('training_%s.csv' % exp)

    new_performance = Perofrmance_restricted_length(filepath, [x_char_val,x_pos_val,x_unic_val,x_vocab_val],y_label,instance_length,'training_real_%s.csv' % exp)

    callbacks_list = [checkpoint, csv_logger,new_performance]  # ,lrate]

    hist = model.fit({'character': char_x, 'pos': pos_x,'unicate': unicate_x , 'vocab': vocab_x },
                     train_y, nb_epoch=epoch_size,
                     batch_size=batchsize, callbacks=callbacks_list,
                     validation_data=({'character': char_x_cv, 'pos': pos_x_cv,'unicate': unicate_x_cv , 'vocab': vocab_x_cv },
                                      cv_y), class_weight=None, sample_weight=classweights)  # None)
    model.save(storage + '/model_result.hdf5')
    np.save(storage + '/epoch_history.npy', hist.history)



# exp1 = "experiment/sentence_level/exp2/exp_embedding_inti"
# epoch = "670"
# weights = exp1+"/weights-improvement-"+ epoch+".hdf5"


x_char,x_pos,x_unic,x_vocab = load_input("training_sentence/train_sentence_input")
y_label = load_pos("training_sentence/train_one_hot_sentence_labels")

x_char_val,x_pos_val,x_unic_val,x_vocab_val = load_input("training_sentence/val_sentence_input")
y_label_val = load_pos("training_sentence/val_one_hot_sentence_labels")
data_name = "val"
instance_length = read_from_json("data/training_sentence/"+data_name+"_instan_len")




n_pos = 46
n_char = 83
n_unicate = 14
n_vocab = 16
epoch_size = 5
batchsize = 100
path = "experiment/"
#path = "/gsfs1/xdisk/dongfangxu9/time_expression/"
exp = "exp_allfeatures_softmax1_non_operator_12labels"

#sample_weights = get_sample_weights_multiclass(trainy_softmax)

sample_weights = np.load("data/training_sentence/sample_weights_all.npy")

#print sample_weights1.shape
#print sample_weights.shape

storage = path + exp



trainging_4features_multiclass(instance_length,storage, sample_weights, exp, x_char,x_pos,x_unic,x_vocab,
                               y_label,x_char_val,x_pos_val,x_unic_val,x_vocab_val, y_label_val, batchsize, epoch_size,
                                    n_char, n_pos, n_unicate, n_vocab,
                                    reload=False, modelpath=None, embedding_size_char=64, embedding_size_pos=32,
                                    embedding_size_unicate=8, embedding_size_vocab=16, gru_size=128)