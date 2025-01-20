# import numpy as np
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical

# def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
#     X1, X2, y = list(), list(), list()
#     n = 0
#     while 1:
#         for key in data_keys:
#             n += 1
#             captions = mapping[key]
#             for caption in captions:
#                 seq = tokenizer.texts_to_sequences([caption])[0]
#                 for i in range(1, len(seq)):
#                     in_seq, out_seq = seq[:i], seq[i]
#                     in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
#                     out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
#                     X1.append(features[key][0])
#                     X2.append(in_seq)
#                     y.append(out_seq)
#             if n == batch_size:
#                 X1, X2, y = np.array(X1), np.array(X2), np.array(y)
#                 yield {"image": X1, "text": X2}, y
#                 X1, X2, y = list(), list(), list()
#                 n = 0


import os
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_captions(captions_path):
    with open(captions_path, 'r', encoding='utf-8') as f:
        next(f)
        return f.read()

def save_features(features, path):
    with open(path, 'wb') as f:
        pickle.dump(features, f)

def load_features(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
