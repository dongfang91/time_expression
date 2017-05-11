import unicodedata
import get_training_data as read
import h5py
from keras.preprocessing.sequence import pad_sequences


def char2int_unicate2int():
    char2int = read.read_from_json('char2int')
    print char2int
    del char2int[u'empty']
    unicatelist_new = list()
    unicatedict =dict()
    unicatelist = list()
    for key,item in char2int.items():
        unicatelist.append(unicodedata.category(key))
    unicatelist_new =  list(enumerate(set(unicatelist), start=1))
    for cate in unicatelist_new:
        unicatedict[cate[1]] = cate[0]
    read.save_in_json("unicatedict",unicatedict)



#char2int_unicate2int()


def generate_unicode_categories(outputfilename):


    raw_text_dir = read.read_from_json('raw_data_dir')
    unicatedict = read.read_from_json("unicatedict")
    data_size = len(raw_text_dir)

    f = h5py.File("data/" + outputfilename + ".hdf5", "w")
    max_len_text = read.get_char2id_dict(raw_text_dir)
    dset = f.create_dataset("input", (data_size, max_len_text), dtype='int8')

    text_unicate_dict = dict()


    for data_id in range(data_size):
        raw_text = read.read_from_dir(raw_text_dir[data_id])
        text_inputs = [[unicatedict[unicodedata.category(char.decode("utf-8"))] for char in raw_text]]
        text_unicate_dict[raw_text_dir[data_id]] = text_inputs[0]
        data_x = pad_sequences(text_inputs, dtype='int8', maxlen=max_len_text, padding="post")
        dset[data_id, :] = data_x[0]
    read.save_in_json("text_unicate_dict",text_unicate_dict)

#generate_unicode_categories("unicode_category_training")

