from keras.models import load_model
import test

from collections import OrderedDict

import os
import sentence_level_process as read
import numpy as np
import get_training_data as read1


def similify_prediction(data,instance_length):
    prediction = list()
    index_sample = 0

    for sample in data:
        position_dict = dict()
        i=0
        #for time_step in sample[0:instance_length[index_sample]]:    #### with instance_length included
        for time_step in sample[0:instance_length]:  #### without instance length included
            if not time_step == 0:
                position_dict[i] = time_step
            i += 1
        new = OrderedDict(sorted(position_dict.items(), key=lambda t: t[0]))
        #print new
        prediction.append(new)
        index_sample +=1
    return prediction

def loc2span(loc):
    span_list = list()
    for loc_sen in loc:
        span =list()
        len_loc_sen = len(loc_sen)
        if len_loc_sen <1:
            span.append([])
        else:

            current_location = 0
            while current_location < len_loc_sen:
                [posi,label] = loc_sen[current_location]
                n_step_forward = 0
                while [posi+n_step_forward, label] in loc_sen:
                    n_step_forward +=1
                    if not [posi+n_step_forward, label]in loc_sen:
                        span.append([posi,posi+n_step_forward-1,label])
                        current_location += n_step_forward
        span_list.append(span)
    #print span_list
    return span_list




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

def prob2classes_multiclasses( prediction):
    if prediction.shape[-1] > 1:
        return prediction.argmax(axis=-1)

def prob2classes_multiclasses_multioutput( prediction):
    output = list()
    for single_predic in prediction:
        if single_predic.shape[-1] > 1:
            output.append(single_predic.argmax(axis=-1))
    return output

def pro2classes_binaryclass(prediction):
    if prediction.shape[-1] <= 1:
        return (prediction > 0.5).astype('int32')


def make_prediction_function_multiclass(x_data,model,exp,data_name):
    model1 = load_model(model)
    y_predict = model1.predict(x_data)
    classes = prob2classes_multiclasses_multioutput(y_predict)

    if not os.path.exists(exp):
        os.makedirs(exp)
    np.save(exp + "/y_predict_classes_"+data_name, classes)

    return classes
    #np.save(exp + "/y_predict_proba_"+data_name, y_predict)




def span2xmlfiles(exp,target):
    import anafora

    raw_dir_simple = read1.read_from_json('raw_dir_simple')
    for data_id in range(0,10):
        data_spans = read1.read_json(exp+"\\span_label_all"+target)[data_id]
        data = anafora.AnaforaData()
        id = 0
        for data_span in data_spans:
            e = anafora.AnaforaEntity()
            e.spans = ((int(data_span[0]), int(data_span[1])+1),)
            e.type = data_span[2]
            e.id = str(id)+"@e@" + raw_dir_simple[data_id]
            data.annotations.append(e)
            id+=1
        print data
        data.indent()

        outputfile = exp+ "\\"+raw_dir_simple[data_id]+"\\"
        if not os.path.exists(outputfile):
            os.makedirs(outputfile)
        data.to_file(outputfile+raw_dir_simple[data_id] +".TimeNorm.gold.completed.xml")
        #data.to_file(outputfile + raw_dir_simple[data_id] + ".xml")
#span2xmlfiles()

def performance_score_multiclass():
    char2int = read1.read_from_json('char2int')
    explicit_labels0 = read1.textfile2list("data/label/explicit_label1.txt")     #### in folder data/label/
    explicit_labels1 = read1.textfile2list("data/label/explicit_label2.txt")     #### in folder data/label/
    labels_index = [explicit_labels0,explicit_labels1,explicit_labels1]

    #labels = explicit_labels1+explicit_labels2


    n_marks = 3
    epoch = "514"
    target = "com_joint"
    # tag ="all"
    exp1 ="experiment\\sentence_level\\all\\"+target+"\\merge_gru"  #"/multi_class"#


    exp=exp1+"\\"+epoch

    data_name = "val"

    add_marks = "" + str(n_marks)

    #######  x_char,x_pos,x_unic,x_vocab  #### model input "training_allsentence_input_addmarks3" "val_sentence_input_addmarks3", in data/training_sentence folder
    x_char,x_pos,x_unic,x_vocab = read.load_input("training_sentence/"+data_name+"_explicitlabel_input"+add_marks)#("training_sentence/"+data_name+"_sentence_input_addmarks")#
    ###label = read.load_pos("training_sentence/"+data_name+"_positiveoperatorsentence_all"+target+"labels"+add_marks)
    intervallabels = read.load_pos("training_sentence/" + data_name + "_allintervallabels" + add_marks)   #### all non-operator labels
    explicitoperatorlabels = read.load_pos("training_sentence/" + data_name + "_explicitoperatorlabels" + add_marks)   ### all explicit operator labels
    implicitlabels = read.load_pos("training_sentence/" + data_name + "_implicitlabels" + add_marks)   ### all implicit labels
    labels = [intervallabels,explicitoperatorlabels,implicitlabels]
    labels_name = ["non-operator labels","explicit operator labels","implicit operator labels"]

    print x_char.shape, x_char[0]


    ####instance_length = test.read_from_json("data/training_sentence/"+data_name+"_instan_len_addmarks"+add_marks)#("data/training_sentence/"+data_name+"_instan_len_addmarks")#

    classes  = make_prediction_function_multiclass([x_char,x_pos,x_unic,x_vocab],exp1+"/weights-improvement-"+ epoch+".hdf5",exp,data_name) #,x_interval

    classes= np.load(exp+"/y_predict_classes_"+data_name+".npy")

    spans = list()
    int2labels = list()



###############################evaluate character level performance on all input labels ##############

    for index in range(len(labels)):
        print "Character-level performance for ", labels_name[index]
        gold = test.hot_vectors2class_index(labels[index])
        class_loc = found_location_with_constraint(classes[index])
        gold_loc = found_location_with_constraint(gold)
        test.calculate_precision_multi_class(class_loc, gold, gold_loc)

###########################################################################################################

        span = loc2span(class_loc)
        spans.append(span)

        one_hot = read.counterList2Dict(list(enumerate(labels_index[index], 1)))
        one_hot = {y: x for x, y in one_hot.iteritems()}
        int2label = dict((int, char) for char, int in one_hot.items())
        int2labels.append(int2label)


    # # # ##########################span level performance, get the real span and its label ######################



    raw_dir_simple = read1.read_from_json('raw_dir_simple')

    sent_index = 0
    data_spans = list()
    for data_id in range(0, 10):
        sent_spans = read1.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])
        data_span = list()
        for sent_span in sent_spans:
            for index in range(len(labels)):
            #index = 0
                span_list = spans[index][sent_index]
                if len(span_list[0]) <1:
                    pass
                else:
                    for [posi_start,posi_end,label] in span_list:
                        data_span.append([posi_start-n_marks+sent_span[1],posi_end-n_marks+ sent_span[1],int2labels[index][label]])

            sent_index += 1
        data_spans.append(data_span)
    read1.save_json(exp+"\\span_label_all"+target,data_spans)
    span2xmlfiles(exp,target)

##################   only focus the non-masking zero labels  #################### another evaluation method ##########
    # instance_length1 = 606 +2*n_marks
    # test.calculate_performance_restricted(gold, instance_length1, classes)
    # class_posi = similify_prediction(classes,instance_length1)
    # gold_posi =  similify_prediction(gold,instance_length1)
    # posi = (0, 251)
    #
    #
    #test.performance_measure(class_posi, gold_posi, posi)
    #test.prediction_debugging(x_char,class_posi, gold_posi, posi,instance_length,labels,char2int)


performance_score_multiclass()







# def combined_labels():
#     import anafora
#     target2 = "operator"
#     target1 = "interval"
#     epoch1 = "503"
#     epoch2 = "311"
#     exp1 = "experiment\\sentence_level\\all\\" + target1 +"\\"+epoch1
#     exp2 = "experiment\\sentence_level\\all\\" + target2 +"\\"+epoch2
#
#
#     raw_dir_simple = read1.read_from_json('raw_dir_simple')
#     for data_id in range(0, 10):
#         data_spans1 = read1.read_json(exp1 + "\\span_label_all" + target1)[data_id]
#         #data_spans1 = read1.read_json("data\\real_span_label")[data_id]
#         data_spans2 = read1.read_json(exp2 + "\\span_label_all" + target2)[data_id]
#         data = anafora.AnaforaData()
#         id = 0
#         for data_span in data_spans1:
#             e = anafora.AnaforaEntity()
#             e.spans = ((int(data_span[0]), int(data_span[1]) + 1),)
#             e.type = data_span[2]
#             e.id = str(id) + "@e@" + raw_dir_simple[data_id]
#             data.annotations.append(e)
#             id += 1
#
#         for data_span in data_spans2:
#             e = anafora.AnaforaEntity()
#             e.spans = ((int(data_span[0]), int(data_span[1]) + 1),)
#             e.type = data_span[2]
#             e.id = str(id) + "@e@" + raw_dir_simple[data_id]
#             data.annotations.append(e)
#             id += 1
#         data.indent()
#         outputfile = "experiment\\sentence_level\\all\\"+"combined_labels\\5" + "\\" + raw_dir_simple[data_id] + "\\"
#         if not os.path.exists(outputfile):
#             os.makedirs(outputfile)
#         data.to_file(outputfile + raw_dir_simple[data_id] + ".TimeNorm.gold.completed.xml")


#combined_labels()
# def model_getweights():
#
#     model = list()
#     data = list()
#
#     for n_marks in range(1,2):
#
#
#         exp1 = "experiment/sentence_level/addmarks_noembedding/"+str(n_marks)+"/multi_class/" + str(n_marks)+"_best.hdf5"
#         model_training = models.load_model(exp1)
#
#         x_char, x_pos, x_unic, x_vocab = read.load_input("training_sentence/val_sentence_input" + "_addmarks" + str(n_marks))
#         data.append([x_char, x_pos, x_unic, x_vocab])
#
#
#         intermediate_layer_model = models.Model(input=model_training.input,
#                                                 output=model_training.layers[9].output)
#
#         model.append(intermediate_layer_model)
#
#         bigru_x1 = intermediate_layer_model.predict([x_char, x_pos, x_unic, x_vocab])
#
#         print bigru_x1.shape
#         print bigru_x1[135]
#         print "asd"




        # t1 = model_training.weights
    #
    # np.save(exp1+"/char_embedding_marks3",t1[0].get_value())
    # np.save(exp1+"/pos_embedding_marks3", t1[1].get_value())
    # np.save(exp1+"/unic_embedding_marks3", t1[2].get_value())
    #np.save(exp1 + "/vocab_embedding_marks2", t1[3].get_value())
    #
    # k = np.load("asd.npy")
    #
    # print k



#model_getweights()
# char_embedding = np.load("experiment/sentence_level/experiment_binary/char_embedding.npy")
# pos_embedding = np.load("experiment/sentence_level/experiment_binary/pos_embedding.npy")
# unicate_embedding = np.load("experiment/sentence_level/experiment_binary/unic_embedding.npy")
#
# print char_embedding.shape,pos_embedding.shape,unicate_embedding.shape

# def make_prediction_function_binaryclass(x_data,model,exp,data_name):
#     model1 = load_model(model)
#     y_predict = model1.predict(x_data)
#     classes = pro2classes_binaryclass(y_predict)    #### the reason to design the function is that there is no model.predict_classes using functional api
#     if not os.path.exists(exp):          ####  please see https://github.com/fchollet/keras/issues/2524
#         os.makedirs(exp)
#
#     np.save(exp + "/y_predict_classes_"+data_name, classes)
#
#     return classes
#
# def performance_score_binaryclass ():
#     ########################   binary_score is the same with keras score systems ##############################
#     exp1 ="experiment/sentence_level/addmarks_noembedding/1/binary"
#     epoch = "156"
#     exp=exp1+"/"+epoch
#
#     data_name = "val"
#
#     add_marks = "_addmarks"
#     x_char,x_pos,x_unic,x_vocab = read.load_input("training_sentence/"+data_name+"_sentence_input"+add_marks)#("training_sentence/"+data_name+"_sentence_input_addmarks")#
#     label = read.load_pos("training_sentence/one_"+data_name+"_sentence_labels"+add_marks)
#
#     instance_length = test.read_from_json("data/training_sentence/" + data_name + "_instan_len" + add_marks)
#
#     #classes  = make_prediction_function_binaryclass([x_char,x_pos,x_unic,x_vocab],exp1+"/weights-improvement-"+ epoch+".hdf5",exp,data_name)
#     classes = np.load(exp + "/y_predict_classes_" + data_name + ".npy")
#
#
#     label_index = label.reshape((label.shape[0], label.shape[1]))
#     class_index = classes.reshape((classes.shape[0], classes.shape[1]))
#
#     class_posi = similify_prediction(class_index,instance_length)
#     gold_posi =  similify_prediction(label_index,instance_length)
#     posi = (0, 251)
#     #posi=(10,63)
#     test.performance_measure(class_posi, gold_posi, posi)
#     test.prediction_debugging(x_char,class_posi, gold_posi, posi,instance_length)


#performance_score_binaryclass()