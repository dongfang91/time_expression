import get_training_data as read
import nltk
import h5py
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.tag.stanford import StanfordPOSTagger



#def simplify():
#     raw_text_dir = read.read_from_json('raw_data_dir')
#     raw_dir_simple = list()
#     for text_dir in raw_text_dir:
#         raw_dir_simple.append(text_dir.rsplit('\\',1)[1])
#     read.save_in_json("raw_dir_simple",raw_dir_simple)
#
# simplify()
# start =0
# end =63
def generate_pos(start=0,end=63):
    english_postagger = StanfordPOSTagger(
        'C:/Users/dongfangxu9/PycharmProjects/pos_tagger/models/english-left3words-distsim.tagger',
        'C:/Users/dongfangxu9/PycharmProjects/pos_tagger/stanford-postagger.jar')
    english_postagger.java_options = '-mx4096m'
    raw_text_dir = read.read_from_json('raw_data_dir')
    data_size = len(raw_text_dir)
    pos = list()


    for data_id in range(start,end):

        raw_text = read.read_from_dir(raw_text_dir[data_id])
        print raw_text_dir[data_id]
        contents = list()
        for line in raw_text.splitlines():
            print line
            text = nltk.word_tokenize(line)
            print text
            if len(text) ==0:
                k = []
            else:
                k = english_postagger.tag(text)
                index = 0
                for token in k:
                    if (text[index] != token[0]) and (token[0] =='``' or token[0] =="''"):   ######### deal with the double quotes, in nltk.tokenize treebank.py change the tokenizer for double quotes. Reasons: (double quotes (") are changed to doubled single forward- and backward- quotes (`` and ''))
                        k[index] =["\"","\'\'"]
                    if token[1] not in pos:
                        pos.append(token[1])
                    index +=1
            contents.append(k)

        read.save_json("data/pos/"+raw_text_dir[data_id].rsplit('\\',1)[1],contents)
    read.save_in_json("pos_tag",pos)

#generate_pos()





start = 0
end = 63
raw_text_dir = read.read_from_json('raw_data_dir')
raw_dir_simple = read.read_from_json('raw_dir_simple')

# data_size = len(raw_text_dir)
max_len_text = read.get_char2id_dict(raw_text_dir)
char2int = read.read_from_json('char2int')
int2char = dict((int,char) for char,int in char2int.items())





text_pos_text_dict = dict()
for data_id in range(start,end):
    print raw_dir_simple[data_id]
    pos = read.read_json("data/pos/"+raw_dir_simple[data_id])
    raw_text = read.read_from_dir(raw_text_dir[data_id])

    text_inputs = [[char2int[char] for char in raw_text]]
    postag = list()
    index = 0
    for line in raw_text.splitlines():
        if len(line) ==0:
            postag.append('\n')
            index +=1
        else:
            token_index = 0
            term = ""
            for char in line:
                # if term =="leade":
                #     print "ok"
                if char ==' ':
                    term =""
                    postag.append("null")
                else:
                    term += char
                    if term in pos[index][token_index][0] and len(term) <len(pos[index][token_index][0]):
                        if bool(re.compile(r'[/\:\-]').match(char)):
                            if len(term) ==1:
                                postag.append(pos[index][token_index][1])
                            else:
                                postag.append('Sep')
                        else:
                            postag.append(pos[index][token_index][1])
                    elif term in pos[index][token_index][0] and len(term) ==len(pos[index][token_index][0]):
                        # if pos[index][token_index][1] =="CD" and bool(re.compile(r'[/\:\-]').match(char)):
                        #     postag.append('Sep')
                        # else:
                        postag.append(pos[index][token_index][1])
                        token_index +=1
                        term = ""
                        if token_index ==len(pos[index]):
                            index +=1
                            postag.append('\n')
    text_pos_text_dict[raw_dir_simple[data_id]] = postag
#read.save_json("data/pos/text_pos_text_dict_normalized",text_pos_text_dict)





def get_list_cd():
    start = 0
    end = 63

    raw_dir_simple = read.read_from_json('raw_dir_simple')
    cd_list = list()
    for data_id in range(start,end):
        #print raw_dir_simple[data_id]
        pos = read.read_json("data/pos/"+raw_dir_simple[data_id])
        for pos_sen in pos:
            if len(pos_sen)>0:
                for pos_token in pos_sen:
                    if pos_token[1] == "CD":
                        if pos_token not in cd_list:
                            cd_list.append(pos_token)
                            print pos_token
    read.save_json("data/pos/cd_list",cd_list)

#get_list_cd()

def get_list_punctuation():
    start = 0
    end = 63
    p_list = ["/", ":", "-"]

    raw_dir_simple = read.read_from_json('raw_dir_simple')
    punctuation_list = list()
    for data_id in range(start,end):
        #print raw_dir_simple[data_id]
        pos = read.read_json("data/pos/"+raw_dir_simple[data_id])
        for pos_sen in pos:
            if len(pos_sen)>0:
                for pos_token in pos_sen:
                    if any( e in pos_token[0] for e in p_list):
                        if pos_token not in punctuation_list:
                            punctuation_list.append(pos_token)
                            print pos_token
    read.save_json("data/pos/punctuation_list",punctuation_list)
#get_list_punctuation()





#
# pos_tags = read.read_from_json("pos_tag")
# pos_tag_dict = dict()
# index = 1
# for tag in pos_tags:
#     pos_tag_dict[tag] = index
#     index +=1
# pos_tag_dict['null'] = index
# pos_tag_dict['\n'] = index+1
# pos_tag_dict['Sep'] = index+2
#
# read.save_in_json("pos_tag_dict",pos_tag_dict)

# text_pos_text_dict =  read.read_json("data/pos/text_pos_text_dict")
# file = 2
# start =1110
# stop =1120
# training_file = "traing_one_zero"
# data_x,data_y = read.load_input(training_file)
#
# print raw_dir_simple[file]
# print ''.join([int2char[i] for i in data_x[file][start:stop]])
# #print data_y[file][start:stop]
# print data_x[file][start:stop]
# print text_pos_text_dict[raw_dir_simple[file]][start:stop]

def generate_pos_training(pos_training_file):
    raw_text_dir = read.read_from_json('raw_data_dir')
    max_len_text = read.get_char2id_dict(raw_text_dir)
    text_pos_text_dict = read.read_json("data/pos/text_pos_text_dict_normalized")
    raw_dir_simple = read.read_from_json('raw_dir_simple')
    pos_tag_dict = read.read_from_json("pos_tag_dict")
    data_size = len(raw_text_dir)

    f = h5py.File("data/" + pos_training_file + ".hdf5", "w")
    dset = f.create_dataset("input", (data_size, max_len_text), dtype='int8')
    #dset2 = f.create_dataset("output", (data_size, max_len_text), dtype='int8')

    for data_id in range(data_size):
        pos_list = text_pos_text_dict[raw_dir_simple[data_id]]
        print raw_dir_simple[data_id]



        text_inputs = [[pos_tag_dict[pos] for pos in pos_list]]
        # print text_inputs
        # print labels
        data_x = pad_sequences(text_inputs, dtype='int8', maxlen=max_len_text, padding="post")
        dset[data_id, :] = data_x[0]


generate_pos_training("pos_training_norm")

