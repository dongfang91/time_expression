import numpy as np
import json
from collections import OrderedDict
import get_training_data as read



def save_in_json(filename, array):
    with open(filename+'.txt', 'w') as outfile:
        json.dump(array, outfile)
    outfile.close()

def read_from_json(filename):
    with open(filename+".txt", 'r') as outfile:
        data = json.load(outfile)
    outfile.close()
    return data

def read_from_dir(path):
    with open (path, "r") as myfile:
        data=myfile.read()
    return data




def xmltag2char (filename):
    xmltags = read_from_json(filename)
    gold_characters = list()
    for xmltag in xmltags:
        gold_dict = dict()
        for posi,info in xmltag.items():
            start = int(posi)
            end = int(info[0])
            for i in range(start,end):
                gold_dict[i] = 1
        new = OrderedDict(sorted(gold_dict.items(), key=lambda t: t[0]))
        gold_characters.append(new)
        print new
    save_in_json("data/gold_characters",gold_characters)

# xmltag2char("data/xmltags.txt")


def performance_measure_gold(exp,posi):
    gold_characters = read_from_json("data/gold_characters")
    predictions = read_from_json(exp + "/prediction_binary")
    pred = 0.0
    gold = 0.0
    match = 0.0
    start,end = posi
    for iter in range(start,end):
        gold_character = gold_characters[iter]
        prediction = predictions[iter]
        pred +=len(prediction)
        gold +=len(gold_character)
        for key in prediction.keys():
            if gold_character.has_key(key):
                match +=1
    print match,pred,gold
    print "precision: ",match/pred
    print "recall: ",match/gold


def performance_measure(predict,true,posi):
    pred = 0.0
    gold = 0.0
    match = 0.0
    start, end = posi
    for iter in range(start, end):
        gold_character = true[iter]
        prediction = predict[iter]
        pred += len(prediction)
        gold += len(gold_character)
        for key in prediction.keys():
            if gold_character.has_key(key) and gold_character[key] == prediction[key]:
                match += 1
    print match, pred, gold
    print "precision: ", match / pred
    print "recall: ", match / gold

def prediction_debugging(char_x,predict,true,posi,instance_length,labels,char2int):

    #labels = read.textfile2list("data/label/one-hot_all.txt")
    one_hot = read.counterList2Dict(list(enumerate(labels, 1)))
    one_hot = {y:x for x,y in one_hot.iteritems()}

    int2char = dict((int, char) for char, int in char2int.items())
    int2label = dict((int, char) for char, int in one_hot.items())


    start, end = posi
    for iter in range(start, end):
        raw_text = [ int2char[int] for int in char_x[iter][0:instance_length[iter]]]
        gold_character = true[iter]
        prediction = predict[iter]
        imprecise = dict()
        imprecise_gold = dict()
        nonrecall = dict()
        imprecise_term = ""

        nonrecall_term = ""

        #######################precision#############################
        for key in prediction.keys():
            if not gold_character.has_key(key) or prediction[key] != gold_character[key]:
                imprecise[key] = int2label[prediction[key]]  ######multiclass_debugging
                #imprecise[key] = prediction[key]             ######binaryclass_debugging
                if gold_character.has_key(key) and prediction[key] != gold_character[key]:
                    imprecise_gold[key] = int2label[gold_character[key]]
                if imprecise.has_key(key-1):
                    if key <len(raw_text):
                        imprecise_term+=raw_text[key]
                    else:
                        imprecise_term = imprecise_term
                else:
                    imprecise_term +=" " +str(key)+": "+ raw_text[key]


        ############################recall##########################
        for key in gold_character.keys():
            if not prediction.has_key(key):
                nonrecall[key] = int2label[gold_character[key]]  ######multiclass_debugging
                #nonrecall[key] = gold_character[key]             ######binaryclass_debugging
                if nonrecall.has_key(key-1):
                    nonrecall_term+=raw_text[key]
                else:
                    nonrecall_term +=" " +str(key)+": "+ raw_text[key]

        if len(imprecise) > 0 or len(nonrecall)>0:
            print "sentence",iter, ''.join(raw_text)
        if len(imprecise)>0:
            new1 = OrderedDict(sorted(imprecise.items(), key=lambda t: t[0]))
            new3 = OrderedDict(sorted(imprecise_gold.items(), key=lambda t: t[0]))

            print "imprecise: ", new1, "    gold:",new3
            print "imprecise: ", imprecise_term

        if len(nonrecall)>0:
            new2 = OrderedDict(sorted(nonrecall.items(), key=lambda t: t[0]))
            print "non_recall: ", new2
            print "non_recall: ", nonrecall_term
        if len(imprecise) > 0 or len(nonrecall) > 0:
            print "\n"

def hot_vectors2class_index (labels):
    examples = list()
    for instance in labels:
        label_index = list()
        for label in instance:
            k = list(label).index(1)
            label_index.append(k)
        examples.append(label_index)
    return examples


def calculate_precision_multi_class(result,gold,gold_loc):
    instan = len(result)
    n_chars = len(result[0])
    n_match = 0.0
    n_pre = 0.0
    n_rec = 0.0



    for i in range(instan):
        n_pre = n_pre + len(result[i])
        n_rec = n_rec + len(gold_loc[i])
        for pre in result[i]:
            if pre[1] == gold[i][int(pre[0])]:
                n_match +=1.0
    precision = n_match/n_pre
    recall = n_match/n_rec
    f1 = 2*precision*recall/(precision + recall)
    print n_pre,n_match,n_rec
    print "presion: ",precision, "recall: ",recall, "F1 score: ",f1


def locations_no_zeros(k,len_constraint):
    index_no_zeros = list()
    for i in range(len(k)):
        if not k[i] ==0 and i<=len_constraint-1:
            index_no_zeros.append((i,k[i]))



def calculate_performance_restricted(gold,instance_length,prediction,debug = True):
    instan = len(prediction)

    n_match = 0.0
    n_pre = 0.0
    n_rec = 0.0

    for i in range(instan):

        for k in range(instance_length):    ###### without setting instance length
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

    print n_pre,n_match,n_rec
    print "presion: ",precision, "recall: ",recall, "F1 score: ",f1



