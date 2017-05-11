import anafora.timeml as timeml
import os
import json
import anafora
import h5py
import numpy as np

#import cv

import math
from collections import OrderedDict

#############################xml into raw data ############################################
# timeml._timeml_dir_to_anafora_dir("data/TBAQ-cleaned/AQUAINT/","data/TBAQ-cleaned/AQUAINT/")


############################  read xml from file  #########################################
# data = anafora.AnaforaData.from_file("ABC19980108.1830.0711/ABC19980108.1830.0711.TimeNorm.gold.completed.xml")
# for annotation in data.annotations:
#     annotation.spans
#     annotation.type


def save_in_json(filename, array):
    with open('data/'+filename+'.txt', 'w') as outfile:
        json.dump(array, outfile)
    outfile.close()

def read_from_json(filename):
    with open('data/'+filename+'.txt', 'r') as outfile:
        data = json.load(outfile)
    outfile.close()
    return data

def read_json(filename):
    with open(filename+'.txt', 'r') as outfile:
        data = json.load(outfile)
    outfile.close()
    return data

def save_json(filename, array):
    with open(filename+'.txt', 'w') as outfile:
        json.dump(array, outfile)
    outfile.close()

def read_from_dir(path):
    data =open(path).read()
    return data

def textfile2list(path):
    data = read_from_dir(path)
    list_new =list()
    for line in data.splitlines():
        list_new.append(line)
    return list_new

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
        print x_data.shape
        print y_data.shape


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
        print x_data.shape
    del x
    return x_data

def get_rawdata_dir():
    '''
    get the directory for whole raw data and xml data, using the same root dir raw_text_dir
    :param raw_text_dir: root directory
    :return: both raw_data directory and xml_data directory
    '''
    dirname = "data/TempEval-2013-Train/"
    xml_file_dir = list()
    raw_data_dir = list()
    roots = os.listdir(dirname)
    root_folder = list()
    for root in roots:
        root_com =os.path.join(dirname,root)
        root_folder += [os.path.join(root_com,f) for f in os.listdir(root_com) if os.path.isdir(os.path.join(root_com, f))]
    for dir in root_folder:
        xml_file_dir+= [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith(".TimeNorm.gold.completed.xml")]

    for file in xml_file_dir:
        text_file = file.replace("data/TempEval-2013-Train","data/TBAQ-cleaned")
        text_file = text_file.replace(".TimeNorm.gold.completed.xml","")
        raw_data_dir.append(text_file)

    return raw_data_dir,xml_file_dir

def xml_to_label(posi_info_dict,type2id,max_review_length):
    type_size = max(type2id.values())+1
    k = []
    for i in range(max_review_length):
        t = np.zeros(type_size)
        t[0] = 1
        k.append(t)

    for posi, info in posi_info_dict.items():
        position = int(posi)
        span_length = info[0] - position
        info.pop(0)
        info.pop(0)
        for i in range(span_length):
            t = np.zeros(type_size)
            # for type in info:
            #     t[type2id[type]] = 1
            type = info[0]
            t[type2id[type]] = 1

            k[position + i] = t
    return k


def index2vector(y, nb_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    y1 = np.copy(y)
    y = [0 if x==-1 else x for x in y1]

    categorical = np.eye(nb_classes)[y]
    for item in range(n):
        if y1[item]==-1:
            categorical[item]  = np.zeros(nb_classes)
    return categorical

#print index2vector([1,2,3,4,5,-1,2],10)


def xml_to_label_binary(posi_info_dict,type2id,real_length):
    type_size = max(type2id.values()) + 1
    k= np.zeros(real_length)

    for posi, info in posi_info_dict.items():
        position = int(posi)
        span_length = info[0] - position
        info.pop(0)
        for i in range(span_length):
            if "Event" not in info:
                k[position + i] = 1
    return k

def get_char2id_dict(raw_data_dir,char_dict=False,datalen = False):
    text = set()
    max_len = 0
    texts_length = list()
    for dir in raw_data_dir:
        raw_text = read_from_dir(dir)
        text.update(set(raw_text))
        length = len(raw_text)
        texts_length.append(length)

        if max_len < length:
            max_len = length
    if char_dict ==True:
        chars = sorted(list(text))
        char2int = dict((c, i + 1) for i, c in enumerate(chars))
        save_in_json('char2int',char2int)
    if datalen == True:
        save_in_json('texts_length', texts_length)
    return max_len

def extrac_xmltag ():
    delete_annotation = ["Event","Modifier","PreAnnotation","NotNormalizable"]
    raw_text_dir = read_from_json('raw_data_dir')
    xmltags = list()
    data_id = 0
    for dir in raw_text_dir:
        raw_text = read_from_dir(raw_text_dir[data_id])
        xml_data_dir = dir.replace("TBAQ-cleaned",
                                                     "TempEval-2013-Train") + ".TimeNorm.gold.completed.xml"
        data = anafora.AnaforaData.from_file(xml_data_dir)
        posi_info_dict = dict()
        for annotation in data.annotations:
            if posi_info_dict.has_key(annotation.spans[0][0]):
                # posi_info_dict[annotation.spans[0][0]].append(annotation.spans[0][1])
                if annotation.type not in delete_annotation:
                    posi_info_dict[annotation.spans[0][0]].append(annotation.type)

            else:
                anna_info = []
                terms = raw_text[annotation.spans[0][0]:annotation.spans[0][1]]
                anna_info.append(annotation.spans[0][1])
                anna_info.append(terms)
                anna_info.append(annotation.type)
                if annotation.type not in delete_annotation:
                    posi_info_dict[annotation.spans[0][0]] = anna_info
        k = OrderedDict(sorted(posi_info_dict.items(), key=lambda t: t[0]))
        print k
        xmltags.append(posi_info_dict)
        data_id+=1
    save_in_json("xmltags_deleted_others",xmltags)

#extrac_xmltag()



##################################  read file from txt, save it as vocabulary ########################
def type2id(input_type_file="data/entity_type.txt",binary = False):
    with open(input_type_file,'r') as file:
        lines = file.read()
    type2id = {}
    if binary == True:
        i = 2
        for line in lines.splitlines():
            type2id[line] = i
            # i+=1
        filename = "binary"
    else:
        i = 0
        for line in lines.splitlines():
            type2id[line] = i
            i+=1
        filename = "multiple"

    save_in_json(filename+"type2id",type2id)

# raw_text_dir = read_from_json('data/raw_data_dir.txt')
# get_char2id_dict(raw_text_dir,char_dict=False,datalen = True)


######################################################generates the directory for raw_data and xml_data###############
# raw_data_dir,xml_file_dir = get_rawdata_dir()
# save_in_json('raw_data_dir',raw_data_dir)
# save_in_json('xml_file_dir',xml_file_dir)

# raw_text_dir = read_from_json('data/raw_data_dir.txt')
# type2id = read_from_json('data/type2id.txt')
# id2type = dict((id,type) for type,id in type2id.items())
# max_len_text = get_char2id_dict(raw_text_dir)
# char2int = read_from_json('data/char2int.txt')
# data_size = len(raw_text_dir)
# type_size = max(type2id.values())+1
#
#
#
# print max(char2int.values())+1
# print type_size
# print data_size
# print max_len_text
########################################################################################################################

def generate_training_data_multiple(outputfilename):
    from keras.preprocessing.sequence import pad_sequences
    raw_text_dir = read_from_json('raw_data_dir')
    char2int = read_from_json('char2int')

    #data_size = len(raw_text_dir)
    max_len_text = get_char2id_dict(raw_text_dir)

    type2id = read_from_json('type2id')
    type_size = max(type2id.values())+1

    xmltags = read_from_json('xmltags')
    data_size =1

    f = h5py.File("data/"+outputfilename+".hdf5", "w")
    dset = f.create_dataset("input", (data_size,max_len_text), dtype='int8')
    dset2 = f.create_dataset("output", (data_size,max_len_text,type_size), dtype='int8')

    for data_id in range(data_size):
    # for data_id in range(1):

        raw_text = read_from_dir(raw_text_dir[data_id])

        labels = xml_to_label(xmltags[data_id],type2id, max_len_text)
        text_inputs = [[char2int[char] for char in raw_text]]
        print text_inputs
        data_x = pad_sequences(text_inputs, dtype='int8', maxlen=max_len_text,padding = "post")
        dset[data_id,:] = data_x[0]
        for i in range(max_len_text):
            dset2[data_id,i,:] = labels[i]

def get_doc_name():
    raw_text_dir = read_from_json('raw_data_dir')
    data_size = len(raw_text_dir)
    for data_id in range(data_size):
        raw_text = read_from_dir(raw_text_dir[data_id])
        if "-year" in raw_text:
            print raw_text_dir[data_id]

#get_doc_name

def generate_training_data_one_zero(outputfilename):
    from keras.preprocessing.sequence import pad_sequences
    raw_text_dir = read_from_json('raw_data_dir')
    char2int = read_from_json('char2int')
    data_size = len(raw_text_dir)
    max_len_text = get_char2id_dict(raw_text_dir)

    type2id = read_from_json('binarytype2id')
    type_size = 1

    texts_length = read_from_json('texts_length')

    xmltags = read_from_json('xmltags')

    f = h5py.File("data/"+outputfilename+".hdf5", "w")
    dset = f.create_dataset("input", (data_size,max_len_text), dtype='int8')
    dset2 = f.create_dataset("output", (data_size,max_len_text), dtype='int8')

    for data_id in range(data_size):


        raw_text = read_from_dir(raw_text_dir[data_id])
        # print raw_text_dir[data_id]
        # print raw_text
        labels = [xml_to_label_binary(xmltags[data_id], type2id,texts_length[data_id])]

        text_inputs = [[char2int[char] for char in raw_text]]
        #print text_inputs
        #print labels
        data_x = pad_sequences(text_inputs, dtype='int8', maxlen=max_len_text,padding = "post")
        data_y = pad_sequences(labels, dtype='int8', maxlen=max_len_text, padding="post")

        dset[data_id,:] = data_x[0]
        dset2[data_id,:] = data_y[0]



# def generate_cv_data(filename,outputfilename):
#     with h5py.File('data/'+ filename + '.hdf5', 'r') as hf:
#         print("List of arrays in this file: \n", hf.keys())
#         x = hf.get('input')
#         y = hf.get('output')
#         x_data = np.array(x)
#         #n_patterns = x_data.shape[0]
#         y_data = np.array(y)
#         # print x_data[0][1000:1100]
#         # print y_data[0][1000:1100]
#         #y_data = y_data.reshape(y_data.shape+(1,)
#     del x
#     del y
#     data_x_aquaint = x_data[0:10]
#     data_x_timebank = x_data[10:]
#
#     data_y_aquaint = y_data[0:10]
#     data_y_timebank = y_data[10:]
#
#     # cv_p_aquaint = cv.random_permutation_matrix(data_x_aquaint.shape[0])
#     # cv_p_timebank = cv.random_permutation_matrix(data_x_timebank.shape[0])
#     #
#     # np.savetxt('data/data4training/cv_p_aquaint_binary.txt', cv_p_aquaint)
#     # np.savetxt('data/data4training/cv_p_timebank_binary.txt', cv_p_timebank)
#     cv_p_aquaint = np.loadtxt('data/data4training/cv_p_aquaint_binary.txt')
#     cv_p_timebank = np.loadtxt('data/data4training/cv_p_timebank_binary.txt')
#     # print cv_p_aquaint.shape
#     # print cv_p_timebank.shape
#
#     data_x_aquaint, data_y_aquaint = cv.get_newdata(data_x_aquaint, data_y_aquaint, cv_p_aquaint)
#     data_x_timebank, data_y_timebank = cv.get_newdata(data_x_timebank, data_y_timebank, cv_p_timebank)
#     #print data_y_aquaint.shape
#     data_x = np.concatenate((data_x_aquaint,data_x_timebank),axis=0)
#     data_y = np.concatenate((data_y_aquaint,data_y_timebank),axis=0)
#     f = h5py.File("data/data4training/" + outputfilename + ".hdf5", "w")
#     raw_text_dir = read_from_json('raw_data_dir')
#     data_size = len(raw_text_dir)
#     max_len_text = get_char2id_dict(raw_text_dir)
#     dset = f.create_dataset("input", (data_size, max_len_text), dtype='int8')
#     dset2 = f.create_dataset("output", (data_size, max_len_text), dtype='int8')
#     for inter in range(len(data_x)):
#         dset[inter, :] = data_x[inter]
#         dset2[inter, :] = data_y[inter]

#generate_cv_data("traing_one_zero","binary_classfication4cv")
#generate_training_data_one_zero("traing_one_zero")

def counterList2Dict (counter_list):
    dict_new = dict()
    for item in counter_list:
        dict_new[item[0]]=item[1]
    return dict_new

def get_labels_sigmoid(outputfilename):
    raw_text_dir = read_from_json('raw_data_dir')
    data_size = len(raw_text_dir)
    text_length = read_from_json("texts_length")
    max_len_text = get_char2id_dict(raw_text_dir)



    labels = textfile2list("data/label/one-hot.txt")
    multi_labels = textfile2list("data/label/multi-hot.txt")
    multi_labels = labels+multi_labels
    multi_hot = counterList2Dict(list(enumerate(multi_labels, 1)))
    multi_hot = {y:x for x,y in multi_hot.iteritems()}

    n_sigmoid = len(multi_labels)+1

    f = h5py.File("data/"+outputfilename+".hdf5", "w")
    dset = f.create_dataset("input", (data_size,max_len_text,n_sigmoid), dtype='int8')

    xmltags = read_from_json('xmltags_deleted_others')

    for data_id in range(data_size):
        print data_id
        sigmoid_labels = np.zeros((max_len_text, n_sigmoid))
        sigmoid_labels[:text_length[data_id],0] = 1

        posi_info_dict = xmltags[data_id]
        for posi, info in posi_info_dict.items():
            position = int(posi)
            posi_end = int(info[0])
            info.pop(0)
            info.pop(0)
            info_new = list(set(info))
            index = 0
            sigmoid_index = list()
            for label in info_new:
                sigmoid_index.append(multi_hot[label])
            if len(sigmoid_index) !=0:
                k = np.sum(np.eye(n_sigmoid)[sigmoid_index],axis = 0)
                sigmoid_labels[position:posi_end,:] = np.repeat([k],posi_end-position, axis=0)

        #softmax_labels  = np.eye(n_softmax) [softmax_index]
        #dset[data_id] = softmax_labels
        dset[data_id] = sigmoid_labels
#get_labels_sigmoid(outputfilename = "labels_sigmoid")


def get_labels_softmax_sigmoid(outputfilename):
    raw_text_dir = read_from_json('raw_data_dir')
    data_size = len(raw_text_dir)
    max_len_text = get_char2id_dict(raw_text_dir)

    labels = textfile2list("data/label/one-hot_12.txt")
    one_hot = counterList2Dict(list(enumerate(labels, 1)))
    one_hot = {y:x for x,y in one_hot.iteritems()}
    n_softmax = len(labels) +1

    multi_labels = textfile2list("data/label/multi-hot.txt")
    multi_hot = counterList2Dict(list(enumerate(multi_labels, 0)))
    multi_hot = {y:x for x,y in multi_hot.iteritems()}
    n_sigmoid = len(multi_labels)

    f = h5py.File("data/"+outputfilename+".hdf5", "w")
    dset = f.create_dataset("input", (data_size,max_len_text,n_softmax), dtype='int8')
    dset2 = f.create_dataset("output", (data_size,max_len_text,n_sigmoid), dtype='int8')

    xmltags = read_from_json('xmltags_deleted_others')
    for data_id in range(data_size):
        print data_id
        sigmoid_labels = np.zeros((max_len_text, n_sigmoid))
        softmax_index = np.zeros(max_len_text,dtype = np.int8)

        posi_info_dict = xmltags[data_id]
        for posi, info in posi_info_dict.items():
            position = int(posi)
            posi_end = int(info[0])
            info.pop(0)
            info.pop(0)
            info_new = list(set(info))
            index = 0
            sigmoid_index = list()
            for label in info_new:
                if label in labels:
                    index = one_hot[label]
                elif label in multi_labels:
                    sigmoid_index.append(multi_hot[label])
            if len(sigmoid_index) !=0:
                k = np.sum(np.eye(n_sigmoid)[sigmoid_index],axis = 0)
                sigmoid_labels[position:posi_end,:] = np.repeat([k],posi_end-position, axis=0)
            softmax_index[position:posi_end] = np.repeat(index,posi_end-position)
        softmax_labels  = np.eye(n_softmax) [softmax_index]
        dset[data_id] = softmax_labels
        #dset2[data_id] = sigmoid_labels

# get_labels_softmax_sigmoid("")

def count_instances_categories(start,end):
    raw_text_dir = read_from_json('raw_data_dir')
    data_size = len(raw_text_dir)

    max_len_text = get_char2id_dict(raw_text_dir)

    labels = textfile2list("data/label/one-hot.txt")
    one_hot = counterList2Dict(list(enumerate(labels, 1)))
    one_hot = {y: x for x, y in one_hot.iteritems()}
    n_softmax = len(labels) + 1

    xmltags = read_from_json('xmltags_deleted_others')
    counts = np.zeros(n_softmax,dtype='int32')
    for data_id in range(start,end):
        softmax_index = np.zeros(max_len_text, dtype=np.int8)
        posi_info_dict = xmltags[data_id]
        for posi, info in posi_info_dict.items():
            position = int(posi)
            posi_end = int(info[0])
            info.pop(0)
            info.pop(0)
            info_new = list(set(info))
            index = 0

            for label in info_new:
                if label in labels:
                    index = one_hot[label]


            softmax_index[position:posi_end] = np.repeat(index, posi_end - position)
        for i in range (n_softmax):
            counts[i] = counts[i] + np.count_nonzero(softmax_index==i)
    return counts

def create_class_weight(labels_dict,mu=0.15):
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

# counts  = count_instances_categories(10,63)
# counts_dict = counterList2Dict(list(enumerate(counts, 0)))
# print counts_dict
#
# print create_class_weight(counts_dict)




def hot_vectors2class_index (labels):
    examples = list()
    for instance in labels:
        label_index = list()
        for label in instance:
            k = list(label).index(1)
            label_index.append(k)
        examples.append(label_index)
    return examples



#get_labels_softmax_sigmoid(outputfilename = "labels_softmax_sigmoid_12")
# softmax_labels,sigmoid_labels = load_input("labels_softmax_sigmoid_12")
# print softmax_labels[0][20:24]

def get_val_input_with_timex(n_marks,outputfilename1):
    from keras.preprocessing.sequence import pad_sequences

    max_len_text = 10802 + 2 * n_marks
    raw_dir_simple = read_from_json('raw_dir_simple')[:10]
    raw_text_dir = read_from_json('raw_data_dir')[:10]

    data_size = len(raw_dir_simple)

    char2int = read_from_json('char2int')
    pos2int = read_from_json('pos_tag_dict')


    text_pos_dict = read_json("data/pos/text_pos_text_dict_normalized")

    text_unicate_dict = read_from_json("text_unicate_dict")

    text_vocab_dict =read_from_json("text_vocab_dict")

    f = h5py.File("data/" + outputfilename1 + str(n_marks) + ".hdf5", "w")
    dset_char = f.create_dataset("char", (data_size, max_len_text), dtype='int8')
    dset_pos = f.create_dataset("pos", (data_size, max_len_text), dtype='int8')
    dset_unic = f.create_dataset("unic", (data_size, max_len_text), dtype='int8')
    dset_vocab = f.create_dataset("vocab", (data_size, max_len_text), dtype='int8')

    for data_id in range(data_size):
        print raw_dir_simple[data_id]
        raw_text = read_from_dir(raw_text_dir[data_id])
        # print raw_text_dir[data_id]
        # print raw_text

        doc_char = [[char2int[char] for char in raw_text]]

        pos_list = text_pos_dict[raw_dir_simple[data_id]]
        doc_pos = [[pos2int[pos] for pos in pos_list]]

        doc_unic = [text_unicate_dict[raw_text_dir[data_id]]]

        doc_vocab = [text_vocab_dict[raw_dir_simple[data_id]]]

        char_input = pad_sequences(doc_char, dtype='int8', maxlen=max_len_text, padding="post")
        pos_input = pad_sequences(doc_pos, dtype='int8', maxlen=max_len_text, padding="post")

        unic_input = pad_sequences(doc_unic, dtype='int8', maxlen=max_len_text, padding="post")
        vocab_input = pad_sequences(doc_vocab, dtype='int8', maxlen=max_len_text, padding="post")

        dset_char[data_id, :] = char_input[0]
        dset_pos[data_id, :] = pos_input[0]
        dset_unic[data_id, :] = unic_input[0]
        dset_vocab[data_id, :] = vocab_input[0]
#get_val_input_with_timex(3,"training_sentence/val_sentence_doc_input_addmarks")

def get_one_hot_labels_with_timex(n_marks,outputfilename1,outputfilename2):    ###contain onehot and multi-hot labels
    raw_dir_simple = read_from_json('raw_dir_simple')[:10]

    data_size = len(raw_dir_simple)


    #data_size = 464  # total traininf sentence with time ex     #0-63 print total,total_with_timex     0:63 witout time ex 1422; with time ex 558;    10:63 with time ex 464; without time ex 1171
    #data_size = 1171 #### total training sentence
    #max_len_text = 606 +2*n_marks  #with marks 606: without marks            NYT19980206.0466       document_length = 10802
    max_len_text = 10802 + 2 * n_marks

    labels = textfile2list("data/label/one-hot_12.txt")
    ############### multiclass classification #############
    one_hot = counterList2Dict(list(enumerate(labels, 1)))
    one_hot = {y:x for x,y in one_hot.iteritems()}
    n_softmax = len(labels) +1
    #####################binary_classification ############
    # one_hot = {label: 1 for label in labels}
    # n_softmax = 1
    #######################################################

    f = h5py.File("data/" + outputfilename1 +str(n_marks)+ ".hdf5", "w")
    dset = f.create_dataset("input", (data_size, max_len_text, n_softmax), dtype='int8')
    total_with_timex = 0

    for data_id in range(0,10):
        xmltags = read_from_json('xmltags_deleted_others')


        softmax_index = np.zeros(max_len_text, dtype=np.int8)

        posi_info_dict = xmltags[data_id]
        for posi, info in posi_info_dict.items():
            position = int(posi)
            posi_end = int(info[0])
            info.pop(0)
            info.pop(0)
            info_new = list(set(info))
            index = 0
            for label in info_new:
                if label in labels:
                    index = one_hot[label]
            softmax_index[position:posi_end] = np.repeat(index, posi_end - position)
        softmax_labels = np.eye(n_softmax)[softmax_index]
        dset[data_id] = softmax_labels

#get_one_hot_labels_with_timex(3,"training_sentence/val_sentence_doc_labels_addmarks","")