import numpy as np
import get_training_data as read
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import cPickle


##############     english_tokenizer = StanfordTokenizer('C:/Users/dongfangxu9/PycharmProjects/pos_tagger/stanford-postagger.jar',
##############                                          options={"americanize": True, }, java_options='-mx1000m')
##############    tokens=english_tokenizer.tokenize(txt)
def pos_tagger(text):
    from nltk.tag.stanford import StanfordPOSTagger
    english_postagger = StanfordPOSTagger(
            'C:/Users/dongfangxu9/PycharmProjects/pos_tagger/models/english-left3words-distsim.tagger',
            'C:/Users/dongfangxu9/PycharmProjects/pos_tagger/stanford-postagger.jar')
    english_postagger.java_options = '-mx4096m'
    tags = english_postagger.tag(text)
    return tags


def spans(txt):
    english_tokenizer = StanfordTokenizer('C:/Users/dongfangxu9/PycharmProjects/pos_tagger/stanford-postagger.jar',
                                       options={"americanize": True, }, java_options='-mx1000m')
    tokens=english_tokenizer.tokenize(txt)
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield token, offset, offset+len(token)
        offset += len(token)


# s = "ABC19980108.1830.0711"
#
# for token in spans(s):
#     print token
#     assert token[0]==s[token[1]:token[2]]

def get_word_from_sentence(start,end):
    wordnet_lemmatizer = WordNetLemmatizer()
    raw_text_dir = read.read_from_json('raw_data_dir')
    raw_dir_simple = read.read_from_json('raw_dir_simple')
    for data_id in range(start, end):
        word_level_chunk = list()
        word_level_chunk_lemma = list()
        sentences = read.read_from_json("training_sentence/sentences/" + raw_dir_simple[data_id])
        for sent in sentences:

            tokens = list()
            tokens_lemma = list()
            tokens_spans = list()
            for token in spans(sent[0]):
                print token[0]
                tokens_lemma.append(wordnet_lemmatizer.lemmatize(token[0].lower()))
                tokens.append(token[0])
                tokens_spans.append((sent[1]+token[1],sent[1]+token[2]))
            word_level_chunk.append([tokens,tokens_spans])
            word_level_chunk_lemma.append([tokens_lemma,tokens_spans])
        read.save_in_json("training_sentence/word_level_sentence/"+raw_dir_simple[data_id],word_level_chunk)
        read.save_in_json("training_sentence/word_level_sentence_lemma/" + raw_dir_simple[data_id], word_level_chunk_lemma)

#get_word_from_sentence(0,63)

def get_pos(start,end):
    # raw_dir_simple = read.read_from_json('raw_dir_simple')
    # max_len = 0  # 106
    # pos_tag_vocab = defaultdict(float)
    # for data_id in range(start, end):
    #     pos_tags = []
    #     sent_spans = read.read_from_json("training_sentence/word_level_sentence/" + raw_dir_simple[data_id])
    #     for [sent,span] in sent_spans:
    #         pos_tag = pos_tagger(sent)
    #         pos_tag = [item[1] for item in pos_tag]
    #         for tag in pos_tag:
    #             pos_tag_vocab[tag]+=1
    #
    #         pos_tags.append(pos_tag)
    #
    #     read.save_in_json("training_sentence/word_level_postag/" + raw_dir_simple[data_id], pos_tags)

    #read.save_in_json("training_sentence/pos_vocab_word",pos_tag_vocab)
    pos_tag_vocab = read.read_from_json("training_sentence/pos_vocab_word")

    pos_tag_vocab['eof'] = 1

    pos_idx_map = dict()
    i =1
    for word in pos_tag_vocab:
        pos_idx_map[word] = i
        i+=1

    read.save_in_json("training_sentence/pos_id", pos_idx_map)

#get_pos(0,63)

def extract_tag(tags):
    result = list()

    tag = tags[1]
    tag_part = list()
    k = len(tag)
    for i in range(k-2):
        tag_part.append(tag[i+2])
    result +=tag_part
    return result

def get_word_tag(start,end):
    multi_labels = read.textfile2list("data/label/multi-hot.txt")
    multi_hot = read.counterList2Dict(list(enumerate(multi_labels, 1)))
    multi_hot = {y:x for x,y in multi_hot.iteritems()}

    raw_text_dir = read.read_from_json('raw_data_dir')
    raw_dir_simple = read.read_from_json('raw_dir_simple')
    max_len = 0    #  106
    for data_id in range(start, end):
        xml_tags = read.read_from_json("training_sentence/xml_tags/" + raw_dir_simple[data_id])
        sent_spans = read.read_from_json("training_sentence/word_level_sentence/" + raw_dir_simple[data_id])
        word_level_tags = list()
        for sent_index in range(len(sent_spans)):
            tags = list()
            for word_index in range(len(sent_spans[sent_index][0])):
                if len(xml_tags[sent_index]) == 0:
                    tags.append(0)
                elif sent_spans[sent_index][1][word_index][0] == int(xml_tags[sent_index][0][0]) and sent_spans[sent_index][1][word_index][1] ==int(xml_tags[sent_index][0][1][0]):
                    xml_tag = extract_tag(xml_tags[sent_index][0])
                    intersection = [x for x in xml_tag if x in multi_labels]
                    if len(intersection) > 0:
                        tags.append(multi_hot[intersection[0]])
                    xml_tags[sent_index].pop(0)
                elif sent_spans[sent_index][1][word_index][1] < int(xml_tags[sent_index][0][0]):
                    tags.append(0)
                else:
                    tags.append(0)
                    while len(xml_tags[sent_index]) >0 and int(xml_tags[sent_index][0][1][0]) <= int(sent_spans[sent_index][1][word_index][1]):
                        xml_tags[sent_index].pop(0)



            word_level_tags.append(tags)

            max_len = max(len(tags), max_len)
        print max_len
        read.save_in_json("training_sentence/word_level_sentence_tag/" + raw_dir_simple[data_id], word_level_tags)

#get_word_tag(0,63)

def get_vocabulary(start, end,lemma):

    #raw_text_dir = read.read_from_json('raw_data_dir')

    raw_dir_simple = read.read_from_json('raw_dir_simple')
    vocab = defaultdict(float)
    for data_id in range(start, end):
        sent_spans = read.read_from_json("training_sentence/word_level_sentence"+lemma+"/" + raw_dir_simple[data_id])
        for sent_index in range(len(sent_spans)):
            for word_index in range(len(sent_spans[sent_index][0])):
                vocab[sent_spans[sent_index][0][word_index]] +=1



    read.save_in_json("training_sentence/vocab_word"+lemma,vocab)

    vocab["\n"] += 1

    word_idx_map = dict()
    i =1
    for word in vocab:
        word_idx_map[word] = i
        i+=1

    read.save_in_json("training_sentence/word_id"+lemma, word_idx_map)

#get_vocabulary(0, 63,"")

def padding(tags,max_l,type_l,pad):
    x = np.zeros((max_l+2*pad,type_l))
    tag_index = np.zeros(max_l+2*pad)
    for i in xrange(pad):
        x[i,0] =1
    index =0
    for word in tags:
        x[pad+index,word] =1
        tag_index[pad+index] = word
        index+=1
    for i in xrange(index+pad,max_l+2*pad):
        x[i,0] =1
    return x,tag_index

def get_idx_from_sent(padding_char,sent, word_idx_map, max_l,pad):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []

    for i in xrange(pad):
        x.append(word_idx_map[padding_char])

    for word in sent:
        if word in word_idx_map:
            x.append(word_idx_map[word])

    for i in xrange(pad):
        x.append(word_idx_map[padding_char])

    while len(x) < max_l+ 2 *pad:
        x.append(0)
    return x

def counterList2Dict (counter_list):
    dict_new = dict()
    for item in counter_list:
        dict_new[item[0]]=item[1]
    return dict_new

def create_class_weight(labels,mu):
    n_softmax = labels.shape[-1]
    counts = np.zeros(n_softmax, dtype='int32')
    for softmax_index in labels:
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
            class_weight[key] = 40.0

    return class_weight

def get_sample_weights_multiclass(labels,mu1):
    class_weight = create_class_weight(labels,mu=mu1)
    samples_weights = list()
    for instance in labels:
        sample_weights = [class_weight[category] for category in instance]
        samples_weights.append(sample_weights)
    return samples_weights

def build_input(pad,lemmenize):

    multi_labels = read.textfile2list("data/label/multi-hot.txt")
    type_l = len(multi_labels) +1
    voca_id = read.read_from_json("training_sentence/word_id"+lemmenize)
    raw_dir_simple = read.read_from_json('raw_dir_simple')
    pos_id = read.read_from_json("training_sentence/pos_id")
    train, dev = [], []
    train_pos, dev_pos = [], []
    train_tag, dev_tag = [], []
    train_weights = []

    max_l = 106
    for data_id in range(10, 63):
        sent_spans = read.read_from_json("training_sentence/word_level_sentence"+lemmenize+"/" + raw_dir_simple[data_id])
        tags = read.read_from_json("training_sentence/word_level_sentence_tag/"+ raw_dir_simple[data_id])
        pos_tags = read.read_from_json("training_sentence/word_level_postag/"+ raw_dir_simple[data_id])
        i=0
        for [sent, span] in sent_spans:
            tag, tag_index = padding(tags[i], max_l, type_l,pad)
            a = [k for k in tags[i] if k != 0]
            if len(a) >0:
                train.append(get_idx_from_sent("\n",sent, voca_id, max_l,pad))
                train_pos.append(get_idx_from_sent("eof",pos_tags[i], pos_id, max_l,pad))
                train_tag.append(tag)
                train_weights.append(tag_index)
            i+=1

    for data_id in range(0, 10):
        sent_spans = read.read_from_json("training_sentence/word_level_sentence"+lemmenize+"/" + raw_dir_simple[data_id])
        tags = read.read_from_json("training_sentence/word_level_sentence_tag/"+ raw_dir_simple[data_id])
        pos_tags = read.read_from_json("training_sentence/word_level_postag/"+ raw_dir_simple[data_id])
        i = 0
        for [sent, span] in sent_spans:
            dev.append(get_idx_from_sent("\n",sent,voca_id,max_l,pad))
            dev_pos.append(get_idx_from_sent("eof",pos_tags[i],pos_id,max_l,pad))
            tag, tag_index = padding(tags[i],max_l,type_l,pad)
            dev_tag.append(tag)
            i+=1

    train = np.asarray(train, dtype="int")
    dev = np.asarray(dev, dtype="int")

    train_pos = np.asarray(train_pos, dtype="int")
    dev_pos = np.asarray(dev_pos, dtype="int")

    #print len(train)


    train_tag = np.asarray(train_tag, dtype="int")
    dev_tag = np.asarray(dev_tag, dtype="int")

    train_weights = get_sample_weights_multiclass(np.asarray(train_weights), mu1=0.1)
    #print train_weights[0]

    cPickle.dump(
        [train, train_pos,dev,dev_pos,train_tag,dev_tag,train_weights], open("data/training_sentence/word_input"+"_pad"+str(pad)+lemmenize, "wb"))

#build_input(3,lemmenize = "")




b = "" #"_lemma"
voca_id = read.read_from_json("training_sentence/word_id"+b)
print len(voca_id)

print len(read.read_from_json("training_sentence/pos_id"))




















