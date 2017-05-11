from keras.models import load_model
import cPickle


import os
import sentence_level_process as read
import numpy as np
import test
import get_training_data as read1
from keras import models


def prob2classes_multiclasses( prediction):
    if prediction.shape[-1] > 1:
        return prediction.argmax(axis=-1)

def pro2classes_binaryclass(prediction):
    if prediction.shape[-1] <= 1:
        return (prediction > 0.5).astype('int32')

def found_location_with_constraint(k):
    instance = list()
    instan_index = 0
    for instan in k:
        loc = list()
        for iter in range(len(instan)):
            #if not instan[iter] ==0 and iter <= instance_length[instan_index]-1:   #### with instance_length set
            if not instan[iter] == 0 :  #### without instance_length set
                loc.append([iter,instan[iter]])
        instance.append(loc)
        instan_index +=1
    return instance

def make_prediction_function_multiclass(x_data,model,exp,data_name):
    model1 = load_model(model)
    y_predict = model1.predict(x_data)
    classes = prob2classes_multiclasses(y_predict)

    if not os.path.exists(exp):
        os.makedirs(exp)
    np.save(exp + "/y_predict_classes_"+data_name, classes)

    return classes

def performance_score_multiclass():
    labels = read1.textfile2list("data/label/multi-hot.txt")
    #labels = read1.textfile2list("data/label/multi-hot.txt")
    one_hot = read.counterList2Dict(list(enumerate(labels, 1)))
    one_hot = {y: x for x, y in one_hot.iteritems()}
    int2label = dict((int, char) for char, int in one_hot.items())

    epoch = "625"

    # tag ="all"
    exp1 ="experiment\\sentence_level\\word_level\\"  #"/multi_class"#


    exp=exp1+"\\"+epoch

    data_name = "val"




    print "Load input . . ."
    x = cPickle.load(open("data/training_sentence/word_input", "rb"))
    train, train_pos, dev, dev_pos, train_tag, dev_tag = x[0], x[1], x[2], x[3], x[4],x[5]
    print "Loaded!\n"

    #classes  = make_prediction_function_multiclass([dev,dev_pos],exp1+"/weights-improvement-"+ epoch+".hdf5",exp,data_name)
    classes = np.load(exp + "/y_predict_classes_" + data_name + ".npy")



    gold = test.hot_vectors2class_index(dev_tag)
    class_loc = found_location_with_constraint(classes)
    gold_loc = found_location_with_constraint(gold)
    test.calculate_precision_multi_class(class_loc, gold, gold_loc)

performance_score_multiclass()