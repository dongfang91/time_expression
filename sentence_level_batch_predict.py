import h5py
import numpy as np
import json
import csv
from collections import Iterable,OrderedDict


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
        RECALL = 0.0
    if n_pre == 0 and n_rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    print(n_match,n_pre,n_rec)
    print("presion: ",precision, "recall: ",recall, "F1 score: ",f1)
    return n_match, n_pre, n_rec, precision, recall, f1

def load_input(filename):
    with h5py.File('data/training_sentence/'+ filename + '.hdf5', 'r') as hf:
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
        #print x_char.shape, x_pos.shape,
    del x1,x2,x4,x3
    return x_char,x_pos,x_unic,x_vocab

def load_pos(filename):
    with h5py.File('data/training_sentence/'+ filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        x = hf.get('input')

        x_data = np.array(x)

        #print x_data.shape
    del x
    return x_data

def read_from_json(filename):
    with open("data/training_sentence/"+filename+".txt", 'r') as outfile:
        data = json.load(outfile)
    outfile.close()
    return data

def hot_vectors2class_index (labels):
    examples = list()
    for instance in labels:
        label_index = list()
        for label in instance:
            k = list(label).index(1)
            label_index.append(k)
        examples.append(label_index)
    return examples

def prob2classes(prediction):
    if prediction.shape[-1] > 1:
        return prediction.argmax(axis=-1)


def handle_value(k):
    is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
    if isinstance(k, Iterable) and not is_zero_dim_ndarray:
        return '"[%s]"' % (', '.join(map(str, k)))
    else:
        return k

def pre(exp,filename):
    from keras import models
    x_char_val, x_pos_val, x_unic_val, x_vocab_val = load_input("val_sentence_input")
    val_data = [x_char_val, x_pos_val, x_unic_val, x_vocab_val]

    y_label_val = load_pos("val_sentence_labels")

    y_label = hot_vectors2class_index(y_label_val)

    data_name = "val"
    instance_length = read_from_json( data_name + "_instan_len")

    path = "/gsfs1/xdisk/dongfangxu9/time_expression/sentence_level/"
    #path = "experiment/sentence_level/exp2/"



    csv_writer = open(filename+".csv", 'w')



    writer = csv.DictWriter(csv_writer,
                                 fieldnames=['1epoch'] + ["2match", "3tagged","4gold","5precision","6recall","7f1"])
    writer.writeheader()
    keys = ["2match", "3tagged","4gold","5precision","6recall","7f1"]

    csvfile  = open("training_"+exp+'.csv')
    reader = csv.DictReader(csvfile)
    for row in reader:
        epoch = row['epoch']
        filepath = path + exp + "/weights-improvement-" + str(epoch) + ".hdf5"
        model_1 = models.load_model(filepath)
        y_predict = model_1.predict(val_data)
        classes = prob2classes(y_predict)
        n_match, n_pre, n_rec, precision, recall, f1 = calculate_performance_restricted(y_label, instance_length,classes)
        performance = {"2match": n_match, "3tagged": n_pre, "4gold": n_rec, "5precision": precision, "6recall": recall, "7f1": f1}

        row_dict = OrderedDict({'1epoch': epoch})
        row_dict.update((key, handle_value(performance[key])) for key in keys)
        keys = sorted(performance.keys())
        writer.writerow(row_dict)
        csv_writer.flush()
    csv_writer.close()

pre("exp_onehot_12labels_embedding_inti","real_exp_onehot_12labels_embedding_inti")



