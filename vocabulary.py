import get_training_data as read
import re
import numpy as np
import h5py
from keras.preprocessing.sequence import pad_sequences

def get_vocab_dict ():
    data = read.read_from_dir("data/vocab/vocab2.txt")
    vocab_dict = dict()
    for line in data.splitlines():
        items = line.split()
        if vocab_dict.has_key(items[1]):
            vocab_dict[items[1]].append(items[0])
        else:
            values = [items[0]]
            vocab_dict[items[1]] = values

    return vocab_dict



def generate_vocab_match(outputfilename):
    vocab_dict = get_vocab_dict()
    n_vocab = max(map(int,vocab_dict.keys()))-1

    #print vocab
    # time_terms = re.compile('|'.join(vocab), re.IGNORECASE)

    raw_text_dir = read.read_from_json('raw_data_dir')
    raw_dir_simple = read.read_from_json('raw_dir_simple')
    data_size = len(raw_text_dir)
    text_length = read.read_from_json('texts_length')

    f = h5py.File("data/" + outputfilename + ".hdf5", "w")
    max_len_text = read.get_char2id_dict(raw_text_dir)
    dset = f.create_dataset("input", (data_size, max_len_text), dtype='int8')
    text_vocab_dict = dict()

    for data_id in range(data_size):
        raw_text = read.read_from_dir(raw_text_dir[data_id])
        a = np.ones(text_length[data_id])
        for index in range(n_vocab):
            vocab = vocab_dict[str(index+2)]
            time_terms = re.compile('|'.join(vocab), re.IGNORECASE)
            for m in time_terms.finditer(raw_text):
                a[m.span()[0]:m.span()[1]] = index+2

        text_vocab_dict[raw_dir_simple[data_id]] = a.tolist()
        data_x = pad_sequences([a.tolist()], dtype='int8', maxlen=max_len_text, padding="post")

        dset[data_id, :] = data_x[0]
    read.save_in_json("text_vocab_dict", text_vocab_dict)

generate_vocab_match("vocab_training")
# vocab = read.load_pos("vocab_training")
# print vocab[49][3000:3050]