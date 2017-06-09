import get_training_data as read
from nltk.tokenize import sent_tokenize
from nltk.tokenize.util import regexp_span_tokenize
from nltk.tag.stanford import StanfordPOSTagger
import nltk
import re
import unicodedata
import h5py

import numpy as np



def spans(sents,txt):
    sentence_chunkings = list()
    offset = 0
    for sent in sents:
        offset = txt.find(sent, offset)
        item = (sent, offset, offset + len(sent))
        offset += len(sent)
        sentence_chunkings.append(item)
    return sentence_chunkings


def split_by_sentence(start=0,end=63):
    """
    Split the document into sentence.    (needed to build end2end system)
    :param start:
    :param end:
    :return:
    """
    raw_text_dir = read.read_from_json('raw_data_dir')   #### in folder data/
    raw_dir_simple = read.read_from_json('raw_dir_simple') #### in folder data/
    for data_id in range(start,end):
        raw_text = read.read_from_dir(raw_text_dir[data_id])
        sent_tokenize_list = sent_tokenize(raw_text)
        sent_tokenize_span_list = spans(sent_tokenize_list,raw_text)

        sent_span_list = list()
        for sent_tokenize_span in sent_tokenize_span_list:
            sent_spans = list(regexp_span_tokenize(sent_tokenize_span[0], r'\n'))
            for sent_span in sent_spans:
                sent_span = (sent_span[0]+sent_tokenize_span[1],sent_span[1]+sent_tokenize_span[1])
                sent_span_list.append((raw_text[sent_span[0]:sent_span[1]],sent_span[0],sent_span[1]))
        read.save_in_json("training_sentence/sentences/"+raw_dir_simple[data_id],sent_span_list)

#split_by_sentence()

def sentence_labeling(start=0,end=63):
    """
    Transform the document-level label into sentence label.
    :param start:
    :param end:
    :return:
    """

    raw_dir_simple = read.read_from_json('raw_dir_simple')
    xmltags = read.read_from_json('xmltags_deleted_others')

    for data_id in range(start, end):
        tag_list = list()

        tag_span = xmltags[data_id].keys()
        tag_span  = sorted(tag_span,key = int)
        print tag_span
        print raw_dir_simple[data_id]
        sentences = read.read_from_json("training_sentence/sentences/"+raw_dir_simple[data_id])
        i=0
        for sent in sentences:
            tag = list()
            if i < len(tag_span):
                if sent[2] < int(tag_span[i]) :
                    tag_list.append(tag)
                elif sent[1]<= int(tag_span[i]) and sent[2]> int(tag_span[i]):
                    while True:
                        tag.append((tag_span[i],xmltags[data_id][tag_span[i]]))
                        i=i+1
                        if i < len(tag_span):
                            if int(tag_span[i]) >sent[2]:
                                tag_list.append(tag)
                                break
                        else:
                            tag_list.append(tag)
                            break
            else:
                tag_list.append(tag)

        read.save_in_json("training_sentence/xml_tags/"+raw_dir_simple[data_id],tag_list)

#sentence_labeling()

def pos_sentence(start=0,end=63):
    """
    Get POS tags for each sentence. (needed to build end2end system)
    :param start:
    :param end:
    :return:
    """
    raw_dir_simple = read.read_from_json('raw_dir_simple')   #### in folder data/
    english_postagger = StanfordPOSTagger(
        'C:/Users/dongfangxu9/PycharmProjects/pos_tagger/models/english-left3words-distsim.tagger',    #### in folder data/
        'C:/Users/dongfangxu9/PycharmProjects/pos_tagger/stanford-postagger.jar') #### in folder data/
    english_postagger.java_options = '-mx4096m'

    pos = list()

    for data_id in range(start, end):
        sentences_spans = read.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])
        print raw_dir_simple[data_id]
        pos_sentences = list()
        for sent_span in sentences_spans:
            print sent_span[0]
            text = nltk.word_tokenize(sent_span[0])
            k = english_postagger.tag(text)   #####StanfordPnOSTagger failed to tag the underscore, see ttps://github.com/nltk/nltk/issues/1632  if use nltk 3.2.2, please change the code "word_tags = tagged_word.strip().split(self._SEPARATOR)" in function "parse_outputcode" of nltk.standford.py into "word_tags = tagged_word.strip().rsplit(self._SEPARATOR,1)" to handle undersocre issues
            index = 0

            for token in k:
                if (text[index] != token[0]) and (token[0] == '``' or token[
                    0] == "''"):  ######### deal with the double quotes, in nltk.tokenize treebank.py change the tokenizer for double quotes. Reasons: (double quotes (") are changed to doubled single forward- and backward- quotes (`` and ''))
                    k[index] = ["\"", "\'\'"]
                if token[1] not in pos:
                    pos.append(token[1])
                index += 1
            pos_sentences.append(k)

        read.save_in_json("training_sentence/pos/" + raw_dir_simple[data_id], pos_sentences)
    read.save_in_json("training_sentence/pos/pos_tag", pos)

#pos_sentence(start=0,end=63)

def generate_character_pos():
    """
    Transofrom word-level POS tag to Character-level POS tag . (needed to build end2end system)
    :return:
    """

    start = 0
    end = 63
    raw_dir_simple = read.read_from_json('raw_dir_simple')
    text_pos_text_dict = dict()

    for data_id in range(start, end):
        sentences_spans = read.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])
        pos_lists = read.read_from_json("training_sentence/pos/" + raw_dir_simple[data_id])
        pos_sentences = list()
        for sent_index in range(len(pos_lists)):
            postag = list()
            token_index = 0
            term = ""
            for char in sentences_spans[sent_index][0]:
                # if term =="leade":
                #     print "ok"
                if char == ' ':
                    term = ""
                    postag.append("null")
                else:
                    term += char
                    if term in pos_lists[sent_index][token_index][0] and len(term) < len(pos_lists[sent_index][token_index][0]):
                        if bool(re.compile(r'[/\:\-]').match(char)):
                            if len(term) == 1:
                                postag.append(pos_lists[sent_index][token_index][1])
                            else:
                                postag.append('Sep')
                        else:
                            postag.append(pos_lists[sent_index][token_index][1])
                    elif term in pos_lists[sent_index][token_index][0] and len(term) == len(pos_lists[sent_index][token_index][0]):
                        # if pos[index][token_index][1] =="CD" and bool(re.compile(r'[/\:\-]').match(char)):
                        #     postag.append('Sep')
                        # else:
                        postag.append(pos_lists[sent_index][token_index][1])
                        token_index += 1
                        term = ""
                        if token_index == len(pos_lists[sent_index]):
                            print postag
                            pos_sentences.append(postag)
        text_pos_text_dict[raw_dir_simple[data_id]] = pos_sentences
    read.save_in_json("training_sentence/pos/text_pos_text_dict_normalized",text_pos_text_dict)

#generate_character_pos()

def generate_unicodecate(start=0,end=63):
    """
    generate unicode category for each character in sentences.  (needed to build end2end system)
    :param start:
    :param end:
    :return:
    """
    raw_dir_simple = read.read_from_json('raw_dir_simple')
    unicatedict = read.read_from_json("unicatedict")  #### in folder data/

    text_unicode_dict = dict()

    for data_id in range(start, end):
        sentences_spans = read.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])
        unicate_sentences = list()
        for sent in sentences_spans:
            unicate_sentences.append([unicatedict[unicodedata.category(char.decode("utf-8"))] for char in sent[0]])
        print unicate_sentences
        text_unicode_dict[raw_dir_simple[data_id]] = unicate_sentences
    read.save_in_json("training_sentence/unicode_category/text_unicode_category_dict_normalized", text_unicode_dict)

#generate_unicodecate(start=0,end=63)

def generate_vocabulary(start=0,end=1):
    """
    Using pre-defined gazetteer to label each character in sentences. (needed to build end2end system)
    :param start:
    :param end:
    :return:
    """
    import vocabulary

    vocab_dict = vocabulary.get_vocab_dict()
    n_vocab = max(map(int, vocab_dict.keys())) - 1
    raw_dir_simple = read.read_from_json('raw_dir_simple')
    text_vocab_dict = dict()

    for data_id in range(start, end):
        sentences_spans = read.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])
        vocab_sentences = list()
        for sent in sentences_spans:
            a = np.ones(len(sent[0]))
            a = a.tolist()

            for index in range(n_vocab):
                vocab = vocab_dict[str(index + 2)]
                time_terms = re.compile('|'.join(vocab), re.IGNORECASE)
                for m in time_terms.finditer(sent[0]):
                    print len(sent[0])
                    print m.span()
                    for posi in range(m.span()[0],m.span()[1]):
                        a[posi] = index + 2
            vocab_sentences.append(a)
        text_vocab_dict[raw_dir_simple[data_id]] = vocab_sentences
    read.save_in_json("training_sentence/vocab/text_vocab_dict_normalized", text_vocab_dict)
#generate_vocabulary(start=0,end=63)


def load_input(filename):
    """
    Load input files. (needed to build end2end system)
    :param filename:
    :return:
    """
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
    """
    (needed to build end2end system)
    :param filename:
    :return:
    """

    with h5py.File('data/'+ filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        x = hf.get('input')

        x_data = np.array(x)

        print x_data.shape
    del x
    return x_data

def extract_tag(tags):
    """
    :param tags: xml_tags input
    :return: a list of tag
    """
    result = list()
    for item in tags:
        tag = item[1]
        tag_part = list()
        k = len(tag)
        for i in range(k-2):
            tag_part.append(tag[i+2])
        result +=tag_part

    # if len(result) >1:
    #     intersection1 = [x for x in result if x in explicit_labels1]
    #     intersection2 = [x for x in result if x in explicit_labels2]
    #
    #     if result[0] == result[1]:
    #         result = result [0]
    #     elif len(intersection1) ==1:
    #         result = intersection1
    #
    #     elif len(intersection2) ==1:
    #         result = intersection2

    return result


def extract_tag1(tags,explicit_labels1,explicit_labels2):
    """
    :param tags: xml_tags input
    :return: a list of explicit_tag
    """
    result = list()
    for item in tags:
        tag = item[1]
        tag_part = list()
        k = len(tag)
        for i in range(k-2):
            tag_part.append(tag[i+2])
        tag_part = get_explict_label(tag_part,explicit_labels1,explicit_labels2)
        result.append(tag_part)
    return result

def get_explict_label(result,explicit_labels1,explicit_labels2):

    if len(result) >1:
        intersection1 = [x for x in result if x in explicit_labels1]
        intersection2 = [x for x in result if x in explicit_labels2]

        if result[0] == result[1]:
            result = result [0]
        elif len(intersection1) ==1:
            result = intersection1

        elif len(intersection2) ==1:
            result = intersection2
    return result[0]

def get_implict_label(result,explicit_labels1,explicit_labels2):

    if len(result) >1:
        intersection1 = [x for x in result if x in explicit_labels1]
        intersection2 = [x for x in result if x in explicit_labels2]

        if result[0] == result[1]:
            result = ["null"]
        elif len(intersection2) ==1:
            result = intersection2

        return result[0]
    else:
        return "null"




def get_rnn_input(n_marks,outputfilename1,outputfilename2):
    """
    process sentence-level features into RNN input.  (needed to build end2end system)
    :param n_marks:
    :param outputfilename1:
    :param outputfilename2:
    :return:
    """
    from keras.preprocessing.sequence import pad_sequences

    raw_dir_simple = read.read_from_json('raw_dir_simple')  #### in folder data/
    char2int = read.read_from_json('char2int')  #### in folder data/
    pos2int = read.read_from_json('pos_tag_dict')   #### in folder data/


    pos_dict = read.read_from_json("training_sentence/pos/text_pos_text_dict_normalized")
    unicate_dict = read.read_from_json("training_sentence/unicode_category/text_unicode_category_dict_normalized")
    vocab_dcit = read.read_from_json("training_sentence/vocab/text_vocab_dict_normalized")
    max_len_text = 606 + 2 * n_marks  #with marks   606: without marks            NYT19980206.0466      document_length = 10802
    #max_len_text = 10802 + 2 * n_marks

    data_size = 1171 ## overall
    #data_size = 161  ## all_explict_operator
    #data_size = 464  # total traininf sentence with time ex     #0-63 print total,total_with_timex     0:63 witout time ex 1422; with time ex 558;    10:63 with time ex 464; without time ex 1171
    #data_size = 395 #### total training sentence with positive operators ######
    f = h5py.File("data/"+outputfilename1+str(n_marks)+".hdf5", "w")
    dset_char = f.create_dataset("char", (data_size,max_len_text), dtype='int8')
    dset_pos = f.create_dataset("pos", (data_size, max_len_text), dtype='int8')
    dset_unic = f.create_dataset("unic", (data_size, max_len_text), dtype='int8')
    dset_vocab = f.create_dataset("vocab", (data_size, max_len_text), dtype='int8')

    #explicit_labels1 = read.textfile2list("data/label/explicit_label1.txt")
    #explicit_labels2 = read.textfile2list("data/label/explicit_label2.txt")
    #explicit_labels =  explicit_labels2

    # explicit_labels = ["Last"]



    #total = 0
    total_with_timex = 0

    # train_senten_count = list()
    #
    # val_senten_count = list()

    #train_instan_len = list()
    val_instan_len = list()

    j=0
    for data_id in range(10,63):
       xml_tags =  read.read_from_json("training_sentence/xml_tags/" + raw_dir_simple[data_id])
       sent_spans = read.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])
       n_sent = len(sent_spans)
       k = 0
       print raw_dir_simple[data_id]
       for index in range(n_sent):

           # if not len(xml_tags[index]) == 0:       ####### using this line to exclude sentence without time ex
           #     xml_tag = extract_tag1 (xml_tags[index],explicit_labels1,explicit_labels2)
           #     intersection = [x for x in xml_tag if x in explicit_labels]
           #     if len(intersection)>0:

                   k+=1
                   ###################################  add end/start of sentence #####################################
                   # sent = "\n\n"+sent_spans[index][0]+"\n\n"
                   # pos_sent = ["\n","\n"] + pos_dict[raw_dir_simple[data_id]][index] + ["\n","\n"]
                   # unic_sent = [3,3]+unicate_dict[raw_dir_simple[data_id]][index]+[3,3]
                   # vocab_sent = [1,1] + vocab_dcit[raw_dir_simple[data_id]][index] + [1,1]
                   #print sent_spans[index][0],xml_tags[index]
                   sent = "\n\n\n"+sent_spans[index][0]+"\n\n\n"
                   pos_sent = ["\n","\n","\n"] + pos_dict[raw_dir_simple[data_id]][index] + ["\n","\n","\n"]
                   unic_sent = [3,3,3]+unicate_dict[raw_dir_simple[data_id]][index]+[3,3,3]
                   vocab_sent = [1,1,1] + vocab_dcit[raw_dir_simple[data_id]][index] + [1,1,1]

                   # sent = "\n\n\n\n" + sent_spans[index][0] + "\n\n\n\n"
                   # pos_sent = ["\n", "\n", "\n", "\n"] + pos_dict[raw_dir_simple[data_id]][index] + ["\n", "\n", "\n", "\n"]
                   # unic_sent = [3, 3, 3, 3] + unicate_dict[raw_dir_simple[data_id]][index] + [3, 3, 3, 3]
                   # vocab_sent = [1, 1, 1, 1] + vocab_dcit[raw_dir_simple[data_id]][index] + [1, 1, 1, 1]

                   # sent = "\n\n"+sent_spans[index][0]+"\n\n"
                   # pos_sent = ["\n","\n"] + pos_dict[raw_dir_simple[data_id]][index] + ["\n","\n"]
                   # unic_sent = [3,3]+unicate_dict[raw_dir_simple[data_id]][index]+[3,3]
                   # vocab_sent = [1,1] + vocab_dcit[raw_dir_simple[data_id]][index] + [1,1]

                   char_input = [[char2int[char] for char in sent]]
                   pos_input = [[pos2int[pos]for pos in pos_sent]]
                   unic_input = [unic_sent]
                   vocab_input = [vocab_sent]
                   ###################################  add end/start of sentence #####################################

                   #train_instan_len.append(len(sent))

                   ####################### no marks for start/end of sentences      ###################################
                   # char_input = [[char2int[char] for char in sent]]
                   # pos_input = [[pos2int[pos]for pos in pos_dict[raw_dir_simple[data_id]][index]]]
                   # unic_input = [unicate_dict[raw_dir_simple[data_id]][index]]
                   # vocab_input = [vocab_dcit[raw_dir_simple[data_id]][index]]
                   ####################### no marks for start/end of sentences      ###################################

                   char_x = pad_sequences(char_input, dtype='int8', maxlen=max_len_text, padding="post")
                   pos_x = pad_sequences(pos_input, dtype='int8', maxlen=max_len_text, padding="post")
                   unic_x = pad_sequences(unic_input, dtype='int8', maxlen=max_len_text, padding="post")
                   vocab_x = pad_sequences(vocab_input, dtype='int8', maxlen=max_len_text, padding="post")
                   dset_char[total_with_timex, :] = char_x[0]
                   dset_pos[total_with_timex, :] = pos_x[0]
                   dset_unic[total_with_timex, :] = unic_x[0]
                   dset_vocab[total_with_timex, :] = vocab_x[0]
                   total_with_timex += 1
       j=j+k
       # train_senten_count.append(j)
    print total_with_timex

    # read.save_in_json("training_sentence/train_sent_len",train_senten_len)

    data_size =251
    f = h5py.File("data/"+outputfilename2+str(n_marks)+".hdf5", "w")
    dset_char_val = f.create_dataset("char", (data_size,max_len_text), dtype='int8')
    dset_pos_val = f.create_dataset("pos", (data_size, max_len_text), dtype='int8')
    dset_unic_val = f.create_dataset("unic", (data_size, max_len_text), dtype='int8')
    dset_vocab_val = f.create_dataset("vocab", (data_size, max_len_text), dtype='int8')
    total_val = 0

    #val_senten_count = list()
    k=0
    for data_id in range(0,10):
       print raw_dir_simple[data_id]
       #xmltags =  read.read_from_json("training_sentence/xml_tags/" + raw_dir_simple[data_id])
       sent_spans = read.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])
       n_sent = len(sent_spans)
       k+= n_sent
       #val_senten_count.append(k)

       for index in range(n_sent):
           # if not len(xmltags[index]) == 0:       ####### using this line to exclude sentence without time ex
           #     xml_tag = extract_tag (xmltags[index])
           #     intersection = [x for x in xml_tag if x in explicit_labels]
           #     if len(intersection)>0:
           #        print sent_spans[index][0], xmltags[index]


           ###################################  add end/start of sentence #####################################
           # sent = "\n" + sent_spans[index][0] + "\n"
           # pos_sent = ["\n"] + pos_dict[raw_dir_simple[data_id]][index] + ["\n"]
           # unic_sent = [3] + unicate_dict[raw_dir_simple[data_id]][index] + [3]
           # vocab_sent = [1] + vocab_dcit[raw_dir_simple[data_id]][index] + [1]

           # sent = "\n\n" + sent_spans[index][0] + "\n\n"
           # pos_sent = ["\n", "\n"] + pos_dict[raw_dir_simple[data_id]][index] + ["\n", "\n"]
           # unic_sent = [3, 3] + unicate_dict[raw_dir_simple[data_id]][index] + [3, 3]
           # vocab_sent = [1, 1] + vocab_dcit[raw_dir_simple[data_id]][index] + [1, 1]

           # sent = "\n\n\n\n" + sent_spans[index][0] + "\n\n\n\n"
           # pos_sent = ["\n", "\n", "\n","\n"] + pos_dict[raw_dir_simple[data_id]][index] + ["\n", "\n", "\n","\n"]
           # unic_sent = [3, 3, 3,3] + unicate_dict[raw_dir_simple[data_id]][index] + [3, 3, 3,3]
           # vocab_sent = [1, 1, 1,1] + vocab_dcit[raw_dir_simple[data_id]][index] + [1,1, 1, 1]

               sent = "\n\n\n" + sent_spans[index][0] + "\n\n\n"
               pos_sent = ["\n", "\n", "\n"] + pos_dict[raw_dir_simple[data_id]][index] + ["\n", "\n", "\n"]
               unic_sent = [3, 3, 3] + unicate_dict[raw_dir_simple[data_id]][index] + [3, 3, 3]
               vocab_sent = [1, 1, 1] + vocab_dcit[raw_dir_simple[data_id]][index] + [1, 1, 1]
               char_input = [[char2int[char] for char in sent]]
               pos_input = [[pos2int[pos] for pos in pos_sent]]
               unic_input = [unic_sent]
               vocab_input = [vocab_sent]

               ###################################  add end/start of sentence #####################################

               #val_instan_len.append(len(sent))

               ####################### no marks for start/end of sentences      ###################################
               # char_input = [[char2int[char] for char in sent]]
               # pos_input = [[pos2int[pos]for pos in pos_dict[raw_dir_simple[data_id]][index]]]
               # unic_input = [unicate_dict[raw_dir_simple[data_id]][index]]
               # vocab_input = [vocab_dcit[raw_dir_simple[data_id]][index]]
               ####################### no marks for start/end of sentences      ###################################

               char_x = pad_sequences(char_input, dtype='int8', maxlen=max_len_text, padding="post")
               pos_x = pad_sequences(pos_input, dtype='int8', maxlen=max_len_text, padding="post")
               unic_x = pad_sequences(unic_input, dtype='int8', maxlen=max_len_text, padding="post")
               vocab_x = pad_sequences(vocab_input, dtype='int8', maxlen=max_len_text, padding="post")
               dset_char_val[total_val, :] = char_x[0]
               dset_pos_val[total_val, :] = pos_x[0]
               dset_unic_val[total_val, :] = unic_x[0]
               dset_vocab_val[total_val, :] = vocab_x[0]
               total_val += 1
    print  total_val
    # #
    # #read.save_in_json("training_sentence/train_instan_len",train_instan_len)
    # # #
    # read.save_in_json("training_sentence/val_instant_len_addmarks"+ str(n_marks), val_instan_len)
    #
    #
    # read.save_in_json("training_sentence/val_sent_count_addmarks"+ str(n_marks), val_senten_count)

#get_training_input_with_timex(2,"training_sentence/training_sentence_input_addmarks2","training_sentence/val_sentence_input_addmarks2")

get_rnn_input(3,"training_sentence/1","training_sentence/2")






def get_one_hot_labels_with_timex(n_marks,outputfilename1,outputfilename2):    ###contain onehot and binary labels
    raw_dir_simple = read.read_from_json('raw_dir_simple')


    data_size = 1171  # total traininf sentence with time ex     #0-63 print total,total_with_timex     0:63 witout time ex 1422; with time ex 558;    10:63 with time ex 464; without time ex 1171
    #data_size = 395 #### total training sentence
    max_len_text = 606 +2*n_marks  #with marks 606: without marks            NYT19980206.0466       document_length = 10802
    #max_len_text = 10802 + 2 * n_marks
    explicit_labels1 = read.textfile2list("data/label/explicit_label1_new.txt")
    explicit_labels2 = read.textfile2list("data/label/explicit_label2.txt")

    labels = explicit_labels2

    ############### multiclass classification #############
    one_hot = read.counterList2Dict(list(enumerate(labels, 1)))
    one_hot = {y:x for x,y in one_hot.iteritems()}
    n_softmax = len(labels) +1
    #####################binary_classification ############
    # one_hot = {label: 1 for label in labels}
    # n_softmax = 1
    #######################################################

    f = h5py.File("data/" + outputfilename1 +str(n_marks)+ ".hdf5", "w")
    dset = f.create_dataset("input", (data_size, max_len_text, n_softmax), dtype='int8')
    total_with_timex = 0

    for data_id in range(10,63):
        xmltags =  read.read_from_json("training_sentence/xml_tags/" + raw_dir_simple[data_id])
        sent_spans = read.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])

        n_sent = len(xmltags)
        for index in range(n_sent):
            softmax_index = np.zeros(max_len_text, dtype=np.int8)
            sentence_start = sent_spans[index][1]

                                                ############## not using this line would include all sentences, and please also change the layout of the sub below  "Shift +Tab"
            # if not len(xmltags[index]) == 0:  ####### using this line to exclude sentence without time ex
            #     xml_tag = extract_tag1(xmltags[index],explicit_labels1,explicit_labels2)
            #     intersection = [x for x in xml_tag if x in labels]
            #     if len(intersection) > 0:
            for label in xmltags[index]:
                posi, info = label
                position = int(posi) - sentence_start
                posi_end = int(info[0]) -sentence_start
                info.pop(0)
                info.pop(0)
                info_new = list(set(info))

                explicit_label = get_implict_label(info_new,explicit_labels1,explicit_labels2)

                ########################   to check whether explicit_label is part of operator #####
                if explicit_label in explicit_labels2:
                    label2int = one_hot[explicit_label]
                    ##################### add marks ########################################
                    softmax_index[position + n_marks:posi_end + n_marks] = np.repeat(label2int, posi_end - position)
                ##################### without marks ########################################
                # softmax_index[position :posi_end ] = np.repeat(index, posi_end - position)

            ############################  multiclass ####################################
            softmax_labels = np.eye(n_softmax)[softmax_index]
            ############################  binaryclass ###################################
            #softmax_labels = softmax_index.reshape(softmax_index.shape + (1,))
            ##########################################################################
            dset[total_with_timex] = softmax_labels
            total_with_timex +=1
    print total_with_timex

    data_size = 251
    f1 = h5py.File("data/" + outputfilename2 +str(n_marks)+ ".hdf5", "w")
    dset1 = f1.create_dataset("input", (data_size, max_len_text, n_softmax), dtype='int8')

    total_with_timex = 0

    for data_id in range(0, 10):
        xmltags = read.read_from_json("training_sentence/xml_tags/" + raw_dir_simple[data_id])
        sent_spans = read.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])

        n_sent = len(xmltags)
        for index in range(n_sent):
            softmax_index = np.zeros(max_len_text, dtype=np.int8)
            sentence_start = sent_spans[index][1]
            if not len(xmltags[index]) == 0:
                for label in xmltags[index]:
                    posi, info = label
                    position = int(posi) - sentence_start
                    posi_end = int(info[0]) - sentence_start
                    info.pop(0)
                    info.pop(0)
                    info_new = list(set(info))
                    index = 0
                    explicit_label = get_implict_label(info_new, explicit_labels1, explicit_labels2)
                    ########################   to check whether explicit_label is part of operator #####
                    if explicit_label in explicit_labels2:

                        label2int = one_hot[explicit_label]
                    ##################### add marks ########################################
                        softmax_index[position+n_marks:posi_end+n_marks] = np.repeat(label2int, posi_end - position)
                    ##################### without marks ########################################
                    #softmax_index[position :posi_end ] = np.repeat(index, posi_end - position)

            ############################  multiclass ####################################
            softmax_labels = np.eye(n_softmax)[softmax_index]
            ############################  binaryclass ###################################
            #softmax_labels = softmax_index.reshape(softmax_index.shape + (1,))
            ##########################################################################
            dset1[total_with_timex] = softmax_labels
            total_with_timex += 1
    print total_with_timex

# get_one_hot_labels_with_timex(2,"training_sentence/one_training_sentence_labels_addmarks", "training_sentence/one_val_sentence_labels_addmarks")
#get_one_hot_labels_with_timex(3,"training_sentence/training_allsentence_newallintervallabels_addmarks", "training_sentence/val_allsentence_newallintervallabels_addmarks")
#get_one_hot_labels_with_timex(3,"training_sentence/training_implicitlabels", "training_sentence/val_implicitlabels")



##############   for operator tagging #######################################################
def get_multi_hot_labels_with_timex(n_marks,outputfilename1,outputfilename2):    ###contain onehot and binary labels
    from copy import deepcopy
    raw_dir_simple = read.read_from_json('raw_dir_simple')


    #data_size = 464  # total traininf sentence with time ex     #0-63 print total,total_with_timex     0:63 witout time ex 1422; with time ex 558;    10:63 with time ex 464; without time ex 1171
    #data_size = 1171 #### total training sentence
    data_size = 278  #### total training sentence with positive operators ######

    max_len_text = 606 +2*n_marks  #with marks 606: without marks            NYT19980206.0466       document_length = 10802
    #max_len_text = 10802 + 2 * n_marks

    multi_labels = read.textfile2list("data/label/multi-hot.txt")
    ############### multiclass classification #############
    multi_hot = read.counterList2Dict(list(enumerate(multi_labels, 1)))
    multi_hot = {y:x for x,y in multi_hot.iteritems()}
    n_sigmoid = len(multi_labels) +1
    #####################binary_classification ############
    # one_hot = {label: 1 for label in labels}
    # n_softmax = 1
    #######################################################

    f = h5py.File("data/" + outputfilename1 +str(n_marks)+ ".hdf5", "w")
    dset = f.create_dataset("input", (data_size, max_len_text, n_sigmoid), dtype='int8')
    total_with_timex = 0
    #n_sents = list()
    for data_id in range(10,63):
       xmltags =  read.read_from_json("training_sentence/xml_tags/" + raw_dir_simple[data_id])
       sent_spans = read.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])

       n_sent = len(xmltags)
       #print n_sent
       #n_sents.append(n_sent)
       for index in range(n_sent):
           sigmoid_labels = np.zeros((max_len_text, n_sigmoid), dtype = np.int8)
           sigmoid_labels [:,0] = 1
           sentence_start = sent_spans[index][1]
           if not len(xmltags[index]) == 0:    ############## using this line to exclude sentence without time ex,
               a = deepcopy(xmltags[index])                                  ############## not using this line would include all sentences, and please also change the layout of the sub below  "Shift +Tab"
               xml_tag = extract_tag(a)
               intersection = [x for x in xml_tag if x in multi_labels]
               if len(intersection) > 0:
                   for label in xmltags[index]:
                       posi, info = label
                       position = int(posi) - sentence_start
                       posi_end = int(info[0]) -sentence_start
                       info.pop(0)
                       info.pop(0)
                       info_new = list(set(info))

                       sigmoid_index = list()
                       for label in info_new:
                           if label in multi_labels:
                               sigmoid_index.append(multi_hot[label])
                       if len(sigmoid_index) != 0:
                           k = np.sum(np.eye(n_sigmoid)[sigmoid_index], axis=0)
                           sigmoid_labels[position+n_marks:posi_end+n_marks, :] = np.repeat([k], posi_end - position, axis=0)
                       ##################### add marks ########################################
                       # softmax_index[position + n_marks:posi_end + n_marks] = np.repeat(index, posi_end - position)
                       ##################### without marks ########################################
                       # softmax_index[position :posi_end ] = np.repeat(index, posi_end - position)

                   ############################  multiclass ####################################
                   #softmax_labels = np.eye(n_softmax)[softmax_index]
                   ############################  binaryclass ###################################
                   #softmax_labels = softmax_index.reshape(softmax_index.shape + (1,))
                   ##########################################################################
                   dset[total_with_timex] = sigmoid_labels
                   total_with_timex +=1
    print total_with_timex

    data_size = 251
    f1 = h5py.File("data/" + outputfilename2 +str(n_marks)+ ".hdf5", "w")
    dset1 = f1.create_dataset("input", (data_size, max_len_text, n_sigmoid), dtype='int8')

    total_with_timex = 0

    for data_id in range(0, 10):
        xmltags = read.read_from_json("training_sentence/xml_tags/" + raw_dir_simple[data_id])
        sent_spans = read.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])

        n_sent = len(xmltags)
        for index in range(n_sent):
            sigmoid_labels = np.zeros((max_len_text, n_sigmoid), dtype=np.int8)
            sigmoid_labels[:, 0] = 1
            sentence_start = sent_spans[index][1]

            for label in xmltags[index]:
               posi, info = label
               position = int(posi) - sentence_start
               posi_end = int(info[0]) -sentence_start
               info.pop(0)
               info.pop(0)
               info_new = list(set(info))

               sigmoid_index = list()
               for label in info_new:
                   if label in multi_labels:
                       sigmoid_index.append(multi_hot[label])
               if len(sigmoid_index) != 0:
                   k = np.sum(np.eye(n_sigmoid)[sigmoid_index], axis=0)
                   sigmoid_labels[position+n_marks:posi_end+n_marks, :] = np.repeat([k], posi_end - position, axis=0)

            dset1[total_with_timex] = sigmoid_labels
            total_with_timex +=1
    #print total_with_timex

#get_multi_hot_labels_with_timex(3,"training_sentence/training_positiveoperatorsentence_alloperatorlabels_addmarks", "training_sentence/val_positiveoperatorsentence_alloperatorlabels_addmarks")



def get_interval_inputs(n_marks,outputfilename1,outputfilename2):
    raw_dir_simple = read.read_from_json('raw_dir_simple')
    data_size = 1171  ## overall
    j = 0
    max_len_text = 606 + 2 * n_marks
    explicit_labels1 = read.textfile2list("data/label/explicit_label1.txt")
    explicit_labels2 = read.textfile2list("data/label/explicit_label2.txt")

    labels = explicit_labels1 + explicit_labels2


    one_hot = read.counterList2Dict(list(enumerate(labels, 1)))
    one_hot = {y: x for x, y in one_hot.iteritems()}
    n_softmax = len(labels) + 2

    f = h5py.File("data/" + outputfilename1 + str(n_marks) + ".hdf5", "w")
    dset = f.create_dataset("input", (data_size, max_len_text), dtype='int8')
    total_with_timex = 0

    for data_id in range(10, 63):
        sent_spans = read.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])
        xmltags = read.read_from_json("training_sentence/xml_tags/" + raw_dir_simple[data_id])
        n_sent = len(sent_spans)
        k = 0
        print raw_dir_simple[data_id]
        for index in range(n_sent):
            softmax_index = np.zeros(max_len_text, dtype=np.int8)
            softmax_index[0:3] = n_softmax-1
            sentence_start = sent_spans[index][1]
            sentence_stop = sent_spans[index][2]
            len_sentence = sentence_stop-sentence_start
            softmax_index[3:3 + len_sentence] = n_softmax
            softmax_index[3+len_sentence:len_sentence+6] = n_softmax - 1
            for label in xmltags[index]:
                posi, info = label
                position = int(posi) - sentence_start
                posi_end = int(info[0]) - sentence_start
                info.pop(0)
                info.pop(0)
                info_new = list(set(info))

                explicit_label = get_explict_label(info_new, explicit_labels1, explicit_labels2)

                ########################   to check whether explicit_label is part of operator #####
                if explicit_label in labels:
                    label2int = one_hot[explicit_label]
                    ##################### add marks ########################################
                    softmax_index[position + n_marks:posi_end + n_marks] = np.repeat(label2int, posi_end - position)

            dset[total_with_timex] = softmax_index
            total_with_timex +=1
    print total_with_timex


    data_size = 251
    f2 = h5py.File("data/" + outputfilename2 + str(n_marks) + ".hdf5", "w")
    dset2 = f2.create_dataset("input", (data_size, max_len_text), dtype='int8')
    total_with_timex = 0


    for data_id in range(0, 10):
        sent_spans = read.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])
        xmltags = read.read_from_json("training_sentence/xml_tags/" + raw_dir_simple[data_id])
        n_sent = len(sent_spans)
        k = 0
        print raw_dir_simple[data_id]
        for index in range(n_sent):
            softmax_index = np.zeros(max_len_text, dtype=np.int8)
            softmax_index[0:3] = n_softmax - 1
            sentence_start = sent_spans[index][1]
            sentence_stop = sent_spans[index][2]
            len_sentence = sentence_stop - sentence_start
            softmax_index[3:3 + len_sentence] = n_softmax
            softmax_index[3 + len_sentence:len_sentence + 6] = n_softmax - 1
            for label in xmltags[index]:
                posi, info = label
                position = int(posi) - sentence_start
                posi_end = int(info[0]) - sentence_start
                info.pop(0)
                info.pop(0)
                info_new = list(set(info))

                explicit_label = get_explict_label(info_new, explicit_labels1, explicit_labels2)

                ########################   to check whether explicit_label is part of operator #####
                if explicit_label in labels:
                    label2int = one_hot[explicit_label]
                    ##################### add marks ########################################
                    softmax_index[position + n_marks:posi_end + n_marks] = np.repeat(label2int,
                                                                                     posi_end - position)
            dset2[total_with_timex] = softmax_index
            total_with_timex +=1
    print total_with_timex

#get_interval_inputs(3,"training_sentence/training_explicit_fea", "training_sentence/val_explicit_fea")


####label with only one hot training data

def hot_vectors2class_index (instance):
    label_index = list()
    for label in instance:
        k = list(label).index(1)
        label_index.append(k)
    return label_index

def hot_vectors2class_index_forweights (labels):
    examples = list()
    n_sen = 0
    for instance in labels:
        n_lable = 0
        label_index = list()
        for label in instance:
            # if list.count(1)==1:
            #     k = list(label).index(1)
            # else:
            #     k = indices = [i for i, x in enumerate(my_list) if x == "whatever"]
            k = list(label).index(1)
            label_index.append(k)
            n_lable +=1

        examples.append(label_index)
        n_sen +=1
    return examples

def locations_no_zeros(k,len_constraint):
    index_no_zeros = list()
    for i in range(len(k)):
        if not k[i] ==0 and i<=len_constraint-1:
            index_no_zeros.append((i,k[i]))
    return index_no_zeros

def found_location_with_constraint(k,instance_length):
    instance = list()
    instan_index = 0
    for instan in k:
        loc = list()
        for iter in range(len(instan)):
            if not instan[iter] ==0 and iter <= instance_length[instan_index]-1:
                loc.append((iter,instan[iter]))
        instance.append(loc)
        instan_index +=1
    return instance

def counterList2Dict (counter_list):
    dict_new = dict()
    for item in counter_list:
        dict_new[item[0]]=item[1]
    return dict_new

def create_class_weight(labels,mu):
    n_softmax = labels.shape[-1]
    class_index = hot_vectors2class_index_forweights(labels)
    counts = np.zeros(n_softmax, dtype='int32')
    for softmax_index in class_index:
        softmax_index = np.asarray(softmax_index)
        for i in range(n_softmax):
            counts[i] = counts[i] + np.count_nonzero(softmax_index==i)

    labels_dict = counterList2Dict(list(enumerate(counts, 0)))

    total = np.sum(labels_dict.values())
    class_weight = dict()

    for key, item in labels_dict.items():
        if not item == 0:
            score = mu * total/float(item)
            class_weight[key] = score if score > 1.0 else 1.0
        else:
            class_weight[key] = 10.0

    return class_weight

def get_sample_weights_multiclass(labels,mu1):
    class_weight = create_class_weight(labels,mu=mu1)
    class_index = np.asarray(hot_vectors2class_index_forweights(labels))
    samples_weights = list()
    for instance in class_index:
        sample_weights = [class_weight[category] for category in instance]
        samples_weights.append(sample_weights)
    return samples_weights

def get_sample_weights_binaryclass(weghtis, label):
    sample_weights = label.copy()
    for i in range(sample_weights.shape[0]):
        for j in range(sample_weights.shape[1]):
            if sample_weights[i][j] == 1:
                sample_weights[i][j] = weghtis
            else:
                sample_weights[i][j] = 1
    return sample_weights



# #######################get sample weights multiclass #########################  #trainy_operator = load_pos("training_alloperatorlabels3")
# y_label = load_pos("training_sentence/training_allexplicitoperatorlabels3")
# print y_label[100]
#
# sample_weights = get_sample_weights_multiclass(y_label,0.05)
# np.save("data/training_sentence/sampleweights_allexplicitoperatorlabels3", sample_weights)
# sample_weights = np.load("data/training_sentence/sampleweights_allexplicitoperatorlabels3.npy")
#
# print sample_weights[150][0:100]

#################################get sample weights binary_class ########################
# label = load_pos("training_sentence/one_training_sentence_labels_addmarks2")#("training_sentence/training_one_hot_sentence_labels_addmarks")  #
#
# weights = label.reshape((label.shape[0],label.shape[1]))
# sample_weights = get_sample_weights_binaryclass(4,weights)
# np.save("data/training_sentence/one_sample_weights4_addmarks2", sample_weights)
# sample_weights = np.load("data/training_sentence/one_sample_weights4_addmarks2.npy")
#
#print sample_weights

############################################################## test whether the preprocess step is correct or not ############################

# x_char,x_pos,x_unic,x_vocab = load_input("training_sentence/training_sentence_input_addmarks2")#("training_sentence/training_sentence_input_addmarks") #
# char2int = read.read_from_json('char2int')
# int2char = dict((int,char) for char,int in char2int.items())
# start = 31
# end = start +1
# print ''.join([int2char[ints] for ints in  x_char[start][0:608]])
# print x_char[start][0:608]
# print x_pos[start][0:608]
# print x_unic[start][0:608]
# print x_vocab[start][0:608]

# label = load_pos("training_sentence/training_sentence_allintervallabels_addmarks3")#("training_sentence/training_one_hot_sentence_labels_addmarks")  #
# #
# a = hot_vectors2class_index(label[start])
# print a
# #
# for item in index_no_zeros[0]:
#     print int2char[x_char[start][0:608][item[0]]]

# import test
# #
# n_marks = "3"
# data_name = "val"
#
# add_marks = "_addmarks" + str(n_marks)
#
# x_char, x_pos, x_unic, x_vocab = load_input("training_sentence/" + data_name + "_sentence_doc_input" + add_marks)  # ("training_sentence/"+data_name+"_sentence_input_addmarks")#
# label = load_pos("training_sentence/" + data_name + "_sentence_doc_labels" + add_marks)
#
# gold = test.hot_vectors2class_index(label)
#
# print "asdawd"