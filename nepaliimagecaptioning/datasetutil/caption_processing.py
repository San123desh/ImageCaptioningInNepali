

# # import os
# # import pickle
# # import numpy as np
# from tqdm import tqdm
# from tensorflow.keras.preprocessing.text import Tokenizer
# # from tensorflow.keras.preprocessing.sequence import pad_sequences


# class CaptionProcessor:
#     def __init__(self):
#         self.tokenizer = None

#     def create_mapping(self, captions_doc):
#         mapping = {}
#         for line in tqdm(captions_doc.split('\n')):
#             tokens = line.split(',')
#             if len(line) < 2:
#                 continue
#             image_id, caption = tokens[0], tokens[1:]
#             image_id = image_id.split('.')[0]
#             caption = " ".join(caption)
#             if image_id not in mapping:
#                 mapping[image_id] = []
#             mapping[image_id].append(caption)
#         return mapping

#     def clean_mapping(self, mapping):
#         for key, captions in mapping.items():
#             for i in range(len(captions)):
#                 caption = captions[i].lower()
#                 caption = caption.replace('[^A-Za-z]', '')
#                 caption = caption.replace('\s+', ' ')
#                 caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
#                 captions[i] = caption

#     def create_tokenizer(self, captions):
#         self.tokenizer = Tokenizer()
#         self.tokenizer.fit_on_texts(captions)
#         return self.tokenizer

#     def get_max_length(self, captions):
#         return max(len(caption.split()) for caption in captions)


import re
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer

class CaptionProcessor:
    def __init__(self, min_freq=1, oov_token="<unk>"):
        self.min_freq = min_freq
        self.oov_token = oov_token
        self.tokenizer = Tokenizer(oov_token=self.oov_token)

    def create_mapping(self, captions_doc):
        mapping = {}
        for line in tqdm(captions_doc.split("\n"), desc="Processing captions"):
            tokens = line.split(",")
            if len(tokens) < 2:
                continue
            image_id, caption = tokens[0], tokens[1:]
            image_id = image_id.split(".")[0]
            caption = " ".join(caption).strip()
            if not caption:
                continue
            if image_id not in mapping:
                mapping[image_id] = []
            mapping[image_id].append(caption)
        print(f"Total images with captions: {len(mapping)}")
        return mapping

    def clean_mapping(self, mapping):
        for key, captions in mapping.items():
            for i in range(len(captions)):
                caption = captions[i].strip().lower()
                caption = re.sub(r"[^\u0900-\u097F\s,.!?0-9]", "", caption)
                # caption = re.sub(r"[^\u0900-\u097F\s]", "", caption)
                caption = re.sub(r"\s+", " ", caption).strip()
                if caption:
                    captions[i] = f"startseq {caption} endseq"
                else:
                    captions[i] = None
            mapping[key] = [caption for caption in captions if caption is not None]
        print(f"Cleaned mapping size: {len(mapping)}")

    def create_tokenizer(self, captions):
        self.tokenizer.fit_on_texts(captions)
        word_counts = self.tokenizer.word_counts
        self.tokenizer.word_index = {k: v for k, v in self.tokenizer.word_index.items() 
                                     if word_counts.get(k, 0) >= self.min_freq or k == self.oov_token}
        return self.tokenizer

    def get_max_length(self, captions):
        max_len = max(len(caption.split()) for caption in captions)
        print(f"Maximum caption length: {max_len}")
        return max_len
