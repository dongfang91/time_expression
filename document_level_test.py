import numpy as np
import re
import h5py

from keras.models import load_model


from collections import OrderedDict

import os
import get_training_data as read
import test



# type2id=read.read_from_json("type2id")
# # for element in type2id:
# #     new = OrderedDict(sorted(element.items(), key=lambda t: int(t[1])))
# #     print new
# new = OrderedDict(sorted(type2id.items(), key=lambda t: int(t[1])))
# for key,value in new.items():
#     print key, value
def similify_prediction(data):
    prediction = list()
    for sample in data:
        position_dict = dict()
        i=0
        for time_step in sample:
            if not time_step == 0:
                position_dict[i] =time_step
            i += 1
        new = OrderedDict(sorted(position_dict.items(), key=lambda t: t[0]))
        #print new
        prediction.append(new)
    return prediction

def pro2label(datas):
    time_step,type_size = datas.shape
    labels = list()
    i=0
    for data in datas:
        k = [j for j in range(type_size) if data[j]>0.5]
        if len(k)>0:
            labels.append((i,k))
        i+=1
    return labels
# prob= np.load(exp+"/y_predict_proba.npy")
# k = prob[1][20:30]
# print k

def make_prediction(x_data,model,exp):


    # print x_data[0][0:30]
    # print x_data[2][0:30]
    model1 = load_model(model)
    #model1.compile()
    y_predict = model1.predict_classes(x_data)
    # if not os.path.exists(exp):
    #     os.makedirs(exp)
    # np.save(exp+"/y_predict_classes", y_predict)
    # y_prob = model1.predict_proba(x_data)
    # np.save(exp+'/y_predict_proba', y_prob)

def found_location(k):
    instance = list()
    for instan in k:
        loc = list()
        for iter in range(len(instan)):
            if not instan[iter] ==0:
                loc.append((iter,instan[iter]))
        instance.append(loc)
    return instance

def prob2classes ( prediction):
    if prediction.shape[-1] > 1:
        return prediction.argmax(axis=-1)


def make_prediction_function(x_data,model,exp):
    model1 = load_model(model)
    y_predict = model1.predict(x_data)
    classes = prob2classes(y_predict)
    if not os.path.exists(exp):
        os.makedirs(exp)
    np.save(exp + "/y_predict_classes", classes)
    np.save(exp + "/y_predict_proba", y_predict)






exp1 ="experiment/exp_softmax_non_operator"
epoch = "88"
exp=exp1+"/"+epoch

#input = "data4training/binary_classfication4cv"


input = "traing_one_zero"
x_data_char,y_data_char = read.load_input(input)
# x_data_pos = read.load_pos("pos_training")
# x_data_unicate = read.load_pos("unicode_category_training")
# x_data_vocab = read.load_pos("vocab_training")
#x_data_unicate = read.load_pos("unicode_category_training")

#make_prediction_function(x_data=[x_data_char,x_data_pos,x_data_unicate,x_data_vocab],model =exp1+"/weights-improvement-"+ epoch+".hdf5",exp=exp)
#make_prediction(x_data=x_data,model =exp1+"/weights-improvement-90.hdf5",exp=exp)

classes= np.load(exp+"/y_predict_classes.npy")

# print classes[0]

class_loc = found_location(classes)


softmax_labels, sigmoid_labels = read.load_input("labels_softmax_sigmoid_12")

#print softmax_labels.shape
k = test.hot_vectors2class_index(softmax_labels)

k_loc = found_location(k)

test.calculate_precision_multi_class(class_loc[0:10],k[0:10],k_loc[0:10])

#classes= np.load(exp+"/y_predict_classes.npy")
#

# print classes.shape
predict_class = similify_prediction(classes)
true_class =similify_prediction(k)
#
#
# print true_class

# read.save_json(exp+"/prediction_binary",predict_class)
# read.save_json(exp+"/true_class",true_class)
posi = (0,10)
# #posi=(10,63)
test.performance_measure(predict_class,true_class,posi)
test.prediction_debugging(predict_class,true_class,posi)





