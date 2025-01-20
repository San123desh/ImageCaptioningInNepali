

# import os
# import pickle
# import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences


class CaptionProcessor:
    def __init__(self):
        self.tokenizer = None

    def create_mapping(self, captions_doc):
        mapping = {}
        for line in tqdm(captions_doc.split('\n')):
            tokens = line.split(',')
            if len(line) < 2:
                continue
            image_id, caption = tokens[0], tokens[1:]
            image_id = image_id.split('.')[0]
            caption = " ".join(caption)
            if image_id not in mapping:
                mapping[image_id] = []
            mapping[image_id].append(caption)
        return mapping

    def clean_mapping(self, mapping):
        for key, captions in mapping.items():
            for i in range(len(captions)):
                caption = captions[i].lower()
                caption = caption.replace('[^A-Za-z]', '')
                caption = caption.replace('\s+', ' ')
                caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
                captions[i] = caption

    def create_tokenizer(self, captions):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(captions)
        return self.tokenizer

    def get_max_length(self, captions):
        return max(len(caption.split()) for caption in captions)
