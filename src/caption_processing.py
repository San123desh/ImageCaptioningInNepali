
# import os
# import pickle
# import numpy as np
# from tqdm import tqdm
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical

# def create_mapping(captions_doc):
#     mapping = {}
#     for line in tqdm(captions_doc.split('\n')):
#         tokens = line.split(',')
#         if len(line) < 2:
#             continue
#         image_id, caption = tokens[0], tokens[1:]
#         image_id = image_id.split('.')[0]
#         caption = " ".join(caption)
#         if image_id not in mapping:
#             mapping[image_id] = []
#         mapping[image_id].append(caption)
#     return mapping

# def clean_mapping(mapping):
#     for key, captions in mapping.items():
#         for i in range(len(captions)):
#             caption = captions[i].lower()
#             caption = caption.replace('[^A-Za-z]', '')
#             caption = caption.replace('\s+', ' ')
#             caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
#             captions[i] = caption

# def create_tokenizer(captions):
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(captions)
#     return tokenizer

# def get_max_length(captions):
#     return max(len(caption.split()) for caption in captions)


import re
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_mapping(captions_doc):
    mapping = {}
    for line in tqdm(captions_doc.split("\n")):
        tokens = line.split(",")
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split(".")[0]
        caption = " ".join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping

def clean_mapping(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].strip().lower()
            caption = re.sub(r"[^\u0900-\u097F\s]", "", caption)  # Keep Devanagari
            caption = re.sub(r"\s+", " ", caption).strip() # Remove leading and trailing spaces(extra spaces)
            if caption:
                caption = f"startseq {caption} endseq"
            else:
                captions[i] = None
        mapping[key] = [caption for caption in captions if caption is not None]

def create_tokenizer(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer

def get_max_length(captions):
    return max(len(caption.split()) for caption in captions)
