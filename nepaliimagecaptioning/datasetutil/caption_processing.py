

# # # # import os
# # # # import pickle
# # # # import numpy as np
# # # from tqdm import tqdm
# # # from tensorflow.keras.preprocessing.text import Tokenizer
# # # # from tensorflow.keras.preprocessing.sequence import pad_sequences


# # # class CaptionProcessor:
# # #     def __init__(self):
# # #         self.tokenizer = None

# # #     def create_mapping(self, captions_doc):
# # #         mapping = {}
# # #         for line in tqdm(captions_doc.split('\n')):
# # #             tokens = line.split(',')
# # #             if len(line) < 2:
# # #                 continue
# # #             image_id, caption = tokens[0], tokens[1:]
# # #             image_id = image_id.split('.')[0]
# # #             caption = " ".join(caption)
# # #             if image_id not in mapping:
# # #                 mapping[image_id] = []
# # #             mapping[image_id].append(caption)
# # #         return mapping

# # #     def clean_mapping(self, mapping):
# # #         for key, captions in mapping.items():
# # #             for i in range(len(captions)):
# # #                 caption = captions[i].lower()
# # #                 caption = caption.replace('[^A-Za-z]', '')
# # #                 caption = caption.replace('\s+', ' ')
# # #                 caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
# # #                 captions[i] = caption

# # #     def create_tokenizer(self, captions):
# # #         self.tokenizer = Tokenizer()
# # #         self.tokenizer.fit_on_texts(captions)
# # #         return self.tokenizer

# # #     def get_max_length(self, captions):
# # #         return max(len(caption.split()) for caption in captions)


# # import re
# # from tqdm import tqdm
# # from tensorflow.keras.preprocessing.text import Tokenizer

# # class CaptionProcessor:
# #     def __init__(self, min_freq=1, oov_token="<unk>"):
# #         self.min_freq = min_freq
# #         self.oov_token = oov_token
# #         self.tokenizer = Tokenizer(oov_token=self.oov_token)

# #     def create_mapping(self, captions_doc):
# #         mapping = {}
# #         for line in tqdm(captions_doc.split("\n"), desc="Processing captions"):
# #             tokens = line.split(",")
# #             if len(tokens) < 2:
# #                 continue
# #             image_id, caption = tokens[0], tokens[1:]
# #             image_id = image_id.split(".")[0]
# #             caption = " ".join(caption).strip()
# #             if not caption:
# #                 continue
# #             if image_id not in mapping:
# #                 mapping[image_id] = []
# #             mapping[image_id].append(caption)
# #         print(f"Total images with captions: {len(mapping)}")
# #         return mapping

# #     def clean_mapping(self, mapping):
# #         for key, captions in mapping.items():
# #             for i in range(len(captions)):
# #                 caption = captions[i].strip().lower()
# #                 caption = re.sub(r"[^\u0900-\u097F\s,.!?0-9]", "", caption)
# #                 # caption = re.sub(r"[^\u0900-\u097F\s]", "", caption)
# #                 caption = re.sub(r"\s+", " ", caption).strip()
# #                 if caption:
# #                     captions[i] = f"startseq {caption} endseq"
# #                 else:
# #                     captions[i] = None
# #             mapping[key] = [caption for caption in captions if caption is not None]
# #         print(f"Cleaned mapping size: {len(mapping)}")

# #     def create_tokenizer(self, captions):
# #         self.tokenizer.fit_on_texts(captions)
# #         word_counts = self.tokenizer.word_counts
# #         self.tokenizer.word_index = {k: v for k, v in self.tokenizer.word_index.items() 
# #                                      if word_counts.get(k, 0) >= self.min_freq or k == self.oov_token}
# #         return self.tokenizer

# #     def get_max_length(self, captions):
# #         max_len = max(len(caption.split()) for caption in captions)
# #         print(f"Maximum caption length: {max_len}")
# #         return max_len

# import os
# from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
# from indicnlp.tokenize import indic_tokenize  
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tqdm import tqdm

# # Ensure resources are set correctly
# from indicnlp import common, loader

# # Set the resources path
# INDIC_RESOURCES_PATH = os.getenv('INDIC_RESOURCES_PATH')
# common.set_resources_path(INDIC_RESOURCES_PATH)
# loader.load()

# class CaptionProcessor:
#     def __init__(self, min_freq=1, oov_token="<unk>"):
#         self.min_freq = min_freq
#         self.oov_token = oov_token
#         self.tokenizer = Tokenizer(oov_token=self.oov_token)

#         # Initialize normalizer for Nepali
#         self.normalizer_factory = IndicNormalizerFactory()
#         self.normalizer = self.normalizer_factory.get_normalizer('ne')

#     def create_mapping(self, captions_doc):
#         mapping = {}
#         for line in tqdm(captions_doc.split("\n"), desc="Processing captions"):
#             tokens = line.split(",")
#             if len(tokens) < 2:
#                 continue
#             image_id, caption = tokens[0], tokens[1:]
#             image_id = image_id.split(".")[0]
#             caption = " ".join(caption).strip()
#             if not caption:
#                 continue
#             if image_id not in mapping:
#                 mapping[image_id] = []
#             mapping[image_id].append(caption)
#         print(f"Total images with captions: {len(mapping)}")
#         return mapping

#     def clean_mapping(self, mapping):
#         for key, captions in mapping.items():
#             for i in range(len(captions)):
#                 caption = captions[i].strip().lower()
#                 # Normalize using Indic NLP Library
#                 caption = self.normalizer.normalize(caption)
#                 # Tokenize and join back to string
#                 tokens = indic_tokenize.trivial_tokenize(caption, 'ne')
#                 caption = ' '.join(tokens)
#                 if caption:
#                     captions[i] = f"startseq {caption} endseq"
#                 else:
#                     captions[i] = None
#             mapping[key] = [caption for caption in captions if caption is not None]
#         print(f"Cleaned mapping size: {len(mapping)}")

#     def create_tokenizer(self, captions):
#         self.tokenizer.fit_on_texts(captions)
#         word_counts = self.tokenizer.word_counts
#         self.tokenizer.word_index = {k: v for k, v in self.tokenizer.word_index.items() 
#                                      if word_counts.get(k, 0) >= self.min_freq or k == self.oov_token}
#         return self.tokenizer

#     def get_max_length(self, captions):
#         max_len = max(len(caption.split()) for caption in captions)
#         print(f"Maximum caption length: {max_len}")
#         return max_len

# # Usage example
# if __name__ == "__main__":
#     caption_processor = CaptionProcessor()
#     captions_doc = open('path/to/captions.txt').read()
#     mapping = caption_processor.create_mapping(captions_doc)
#     caption_processor.clean_mapping(mapping)
#     all_captions = [caption for captions in mapping.values() for caption in captions]
#     tokenizer = caption_processor.create_tokenizer(all_captions)
#     max_length = caption_processor.get_max_length(all_captions)

import os
from indicnlp import common, loader

# Set the environment variable explicitly (if needed)
os.environ['INDIC_RESOURCES_PATH'] = 'C:\\Users\\Acer\\ImageCaptioning\\indic_nlp_resources'

# Ensure resources are set correctly
INDIC_RESOURCES_PATH = os.getenv('INDIC_RESOURCES_PATH')
print(f"INDIC_RESOURCES_PATH set to: {INDIC_RESOURCES_PATH}")
common.set_resources_path(INDIC_RESOURCES_PATH)
resources_path = common.get_resources_path()
print(f"Resources Path from common.get_resources_path(): {resources_path}")
loader.load()
print("Indic NLP Library resources loaded successfully")

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

class CaptionProcessor:
    def __init__(self, min_freq=1, oov_token="<unk>"):
        self.min_freq = min_freq
        self.oov_token = oov_token
        self.tokenizer = Tokenizer(oov_token=self.oov_token)

        # Initialize normalizer for Nepali
        self.normalizer_factory = IndicNormalizerFactory()
        self.normalizer = self.normalizer_factory.get_normalizer('ne')

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
                # Normalize using Indic NLP Library
                caption = self.normalizer.normalize(caption)
                # Tokenize and join back to string
                tokens = indic_tokenize.trivial_tokenize(caption, 'ne')
                caption = ' '.join(tokens)
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

