import numpy as np
import os
import pickle as cPickle
import math
import h5py

from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense,TimeDistributed,merge
from keras.layers import GRU,Dropout,Input
from keras.layers import Embedding
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.regularizers import l1, l2
from keras.models import Model

from keras.callbacks import ModelCheckpoint


from keras.optimizers import RMSprop



def training_model(storage,exp,input_path,batch_size = 200,epoch_size =800,word_embedding_size = 128,pos_tag_embedding_size=8, gru_size1=128, gru_size2=160,n_word = 5651,n_pos = 46,reload = False,modelpath = None):
    print("Load input . . .")
    x = cPickle.load(open(input_path, "rb"))#,encoding='latin1')
    train,train_pos,dev, dev_pos,train_tag, dev_tag, train_weights = x[0], x[1], x[2], x[3], x[4],x[5], x[6]
    print("Loaded!\n")

    rmsprop = RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.0)
    seq_length = train.shape[1]
    type_size = train_tag.shape[-1]



    if not os.path.exists(storage+exp+"/"):
        os.makedirs(storage+exp+"/")

    if reload == False:

        word_input = Input(shape=(seq_length,), dtype='float32', name='word')
        word_em = Embedding(output_dim=word_embedding_size,input_dim=n_word, input_length=seq_length,
                            W_regularizer=l1(.01), mask_zero=True, dropout=0.12)(word_input) #weights=[char_embedding],

        pos_input = Input(shape=(seq_length,), dtype='float32', name='pos')
        pos_em = Embedding(output_dim=pos_tag_embedding_size,input_dim=n_pos, input_length=seq_length,
                           W_regularizer=l1(.01), mask_zero=True, dropout=0.1)(pos_input) #weights=[pos_embedding],



        input_merge = merge([word_em,pos_em], mode='concat')

        gru_out_1 = Bidirectional(GRU(gru_size1, input_shape=(
        seq_length, word_embedding_size + pos_tag_embedding_size ),
                                      return_sequences=True))(input_merge)

        gru_out_2 = GRU(gru_size2, return_sequences=True)(gru_out_1)

        softmax_output = TimeDistributed(Dense(type_size, activation='softmax', W_regularizer=l1(.01)))(
            gru_out_2)




        model = Model(input=[word_input, pos_input], output=softmax_output)

        #outputs = model.predict_on_batch({'word': train, 'pos': train_fea2, 'chunk_tag': train_fea3}


        model.compile(optimizer=rmsprop,
                      loss='categorical_crossentropy',
                      metrics=['fmeasure', 'categorical_accuracy', 'recall', 'precision'],
                      sample_weight_mode="temporal")

    else:
        model = load_model(modelpath)

    print("Begin to train...")
    filepath = storage +exp+ "/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='fmeasure', verbose=1, save_best_only=False, mode='max')
    csv_logger = CSVLogger('training_%s.csv' % exp)

    # new_performance = Perofrmance_restricted_length(filepath, [x_char_val,x_pos_val,x_unic_val,x_vocab_val],y_label,instance_length,'training_real_%s.csv' % exp)

    callbacks_list = [checkpoint, csv_logger]  # ,new_performance]  # ,lrate]

    hist = model.fit({'word': train, 'pos': train_pos},
                     train_tag, nb_epoch=epoch_size,
                     batch_size=batch_size, callbacks=callbacks_list,
                     validation_data=(
                     {'word': dev, 'pos': dev_pos},
                     dev_tag), class_weight=None, sample_weight=np.asarray(train_weights))  # None)
    model.save(storage+exp + '/model_result.hdf5')
    np.save(storage + exp+'/epoch_history.npy', hist.history)

training_model(storage="data/experiment/sentence_level/word_level",exp ="exp_wordlevel_pad3",input_path="data/training_sentence/word_input_pad3")
