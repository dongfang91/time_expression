import numpy as np
import os
import math

from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense,TimeDistributed,Merge
from keras.layers import GRU
from keras.layers import Embedding
from keras.models import load_model
from keras.callbacks import CSVLogger

from keras.callbacks import ModelCheckpoint

from keras.callbacks import Callback

from keras.optimizers import RMSprop
from collections import Counter

import get_training_data as read

from keras.callbacks import LearningRateScheduler

from keras.utils.generic_utils import Progbar

import keras


class ProgbarLogger1(Callback):
    """Callback that prints metrics to stdout.
    """

    def on_train_begin(self, logs=None):
        self.verbose = self.params['verbose']
        self.nb_epoch = self.params['nb_epoch']

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
            self.progbar = Progbar(target=self.params['nb_sample'],
                                   verbose=self.verbose)
        self.seen = 0

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.params['nb_sample']:
            self.log_values = []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and self.seen < self.params['nb_sample']:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values, force=True)





def trainging_2features(classweights,exp,train_x,pos_x,trainy,cv_x,cv_x_pos,cvy,epoch_size,n_char,n_pos,activ = 'softmax',reload = False,modelpath = None,embedding_size_char =64,embedding_size_pos = 32, gru_size = 128):
    train_y = trainy.reshape(trainy.shape + (1,))
    cv_y = cvy.reshape(cvy.shape + (1,))
    print train_y.shape
    print cv_y.shape
    #print train_y[0][0:30]

    seq_length = train_x.shape[1]
    type_size = train_y.shape[-1]



    if not os.path.exists(exp):
        os.makedirs(exp)
    if reload ==False:

        model_char = Sequential()
        model_char.add(Embedding(n_char, embedding_size_char, input_length=seq_length, mask_zero=True))

        model_pos = Sequential()
        model_pos.add(Embedding(n_pos, embedding_size_pos, input_length=seq_length, mask_zero=True))

        model_final = Sequential()
        model_final.add(Merge([model_char, model_pos], mode='concat'))

        model_final.add(Bidirectional(GRU(gru_size, input_shape=(seq_length, embedding_size_char+embedding_size_pos), return_sequences=True)))
        model_final.add(GRU(gru_size, return_sequences=True))
        model_final.add(TimeDistributed(Dense(type_size, activation=activ)))

        model_final.compile(loss='binary_crossentropy', optimizer='rmsprop',
                      metrics=['fmeasure', 'precision', 'recall','accuracy'],sample_weight_mode="temporal")#
    else:
        model_final = load_model(modelpath)

    filepath = exp + "/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='fmeasure', verbose=1, save_best_only=False, mode='min')
    csv_logger = CSVLogger(exp + '/training_%s.log' % exp)

    a = ProgbarLogger1()

    callbacks_list = [checkpoint, csv_logger,a]#,lrate]

#(cv_x,cv_y)
    hist = model_final.fit([train_x,pos_x], train_y, nb_epoch=epoch_size, batch_size=20, callbacks=callbacks_list,validation_data =([cv_x,cv_x_pos],cv_y),class_weight=None,sample_weight=classweights)#None)
    model_final.save(exp + '/model_result.hdf5')
    np.save(exp + '/epoch_history.npy', hist.history)







#read.generate_training_data_one_zero(outputfilename = training_file)
#read.generate_training_data_multiple(outputfilename = training_file)


# cv_p_aquaint = np.loadtxt('data/data4training/cv_p_aquaint_binary.txt')
# cv_p_timebank = np.loadtxt('data/data4training/cv_p_timebank_binary.txt')
#
# print "index of aquaint:"
# for data in cv_p_aquaint:
#
#     print np.argmax(data)
#
# print "index of timebank:"
# for data in cv_p_timebank:
#     print np.argmax(data)+10



# import get_training_data as read
# raw_data_dir = read.read_from_json("raw_data_dir")
# raw_data_dir_dict = dict()
# for item in range(len(raw_data_dir)):
#     raw_data_dir_dict[item] = raw_data_dir[item]
# new = OrderedDict(sorted(raw_data_dir_dict.items(), key=lambda t: t[0]))
# read.save_in_json("raw_data_dir_dict",new)
# for key,item in new.items():
#     print key , item





def get_class_weights(y):
    weights ={0:0,1:0}
    for data in y:
        counter = Counter(data)
        zero = max(counter.values())
        one = min (counter.values())
        weights[0]+=zero
        weights[1]+=one
    return  {cls: float(weights[0]/count) for cls, count in weights.items()}
# weights = get_class_weights(data_y)
# print weights

char2int = read.read_from_json('char2int')
int2char = dict((int,char) for char,int in char2int.items())

raw_text_dir = read.read_from_json('raw_data_dir')


nchars = 83
npos = 46
epoch_size = 2
exp = "exp1_pos_char"



# file = 15
# start = 9990
# stop =  10000
training_file = "traing_one_zero_sc"
data_x,data_y = read.load_input(training_file)

pos_x = read.load_pos("pos_training_norm")
# unicate_x = read.load_pos("unicode_category_training")
# vocab_x = read.load_pos("vocab_training")

# print raw_text_dir[file]
# print ''.join([int2char[i] for i in data_x[file][start:stop]])
# print data_x[file][start:stop]
# print pos_x[file][start:stop]

# print unicate_x[file][start:stop]
# print vocab_x[file][start:stop]

# import unicodedata
# print unicodedata.category(':'.decode("utf-8"))
# print unicodedata.category('/'.decode("utf-8"))
# print unicodedata.category(' '.decode("utf-8"))



# fold_size =5

#
# fold_indices_aquaint = map(lambda x: int(x), np.linspace(0, 10, fold_size + 1))
#
# fold_indices_timebank = map(lambda x: int(x)+10, np.linspace(0, 53, fold_size + 1))
# fold = 1
# foldx1 = data_x[fold_indices_aquaint[fold]:fold_indices_aquaint[fold + 1]]
# foldx2 = data_x[fold_indices_timebank[fold]:fold_indices_timebank[fold + 1]]
# fold_1 = np.concatenate((np.arange(fold_indices_aquaint[fold], fold_indices_aquaint[fold + 1]),np.arange(fold_indices_timebank[fold], fold_indices_timebank[fold + 1])))
# print fold_1

# trainx = np.delete(data_x,fold_1, 0)
# cv_x = data_x[fold_1]
# trainy = np.delete(data_y,fold_1, 0)
# cv_y = data_y[fold_1]




trainx = data_x[10:11]
cv_x = data_x[0:1]
trainy = data_y[10:11]
cv_y = data_y[0:1]

trainx_pos = pos_x[10:11]
cv_x_pos = pos_x[0:1]


cost_weights = 12
sample_weights = trainy.copy()
for i in range(sample_weights.shape[0]):
    for j in range(sample_weights.shape[1]):
        if sample_weights[i][j] ==1:
            sample_weights[i][j] =cost_weights
        else:
            sample_weights[i][j] = 1



#print "sample_weights:",sample_weights[0][0:30]
#training(sample_weights,exp,trainx,trainy,cv_x,cv_y,epoch_size,nchars,activ ='sigmoid')#,reload = True,modelpath = "exp15_continue1/weights-improvement-35.hdf5")

#print trainx[0][1:100]

trainging_2features(sample_weights,exp,trainx,trainx_pos,trainy,cv_x,cv_x_pos,cv_y,epoch_size,nchars,npos,activ ='sigmoid')#,reload = True,modelpath = "exp15_continue1/weights-improvement-35.hdf5")