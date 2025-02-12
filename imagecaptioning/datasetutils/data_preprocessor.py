import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import re

class ImageCaptioningDataPreprocessor:
    def __init__(self, image_path, captions_file):
        self.image_path = image_path
        self.data = self.load_and_process_captions(captions_file)
        self.tokenizer = Tokenizer()
        self.vocab_size = 0
        self.max_length = 0

    def load_and_process_captions(self, captions_file):
        
        data = []
        with open(captions_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(',', 1)  
                if len(parts) == 2:
                    image, caption = parts
                    data.append((image, caption))
        return pd.DataFrame(data, columns=['image', 'caption'])

    def preprocess_text(self):
        
        if 'caption' in self.data.columns:
            # self.data['caption'].fillna('', inplace=True)
            self.data['caption'] = self.data['caption'].fillna('')
            self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())
            self.data['caption'] = self.data['caption'].apply(lambda x: "startseq " + x + " endseq")
        else:
            raise KeyError("Column 'caption' not found in DataFrame")

    def prepare_tokenizer(self):
        captions = self.data['caption'].tolist()
        self.tokenizer.fit_on_texts(captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_length = max(len(word_tokenize(caption)) for caption in captions)

    def split_data(self, train_ratio=0.85):
        if 'image' in self.data.columns:
            images = self.data['image'].unique().tolist()
            split_index = round(train_ratio * len(images))
            train_images = images[:split_index]
            val_images = images[split_index:]
            train_data = self.data[self.data['image'].isin(train_images)]
            val_data = self.data[self.data['image'].isin(val_images)]
            return train_data, val_data
        else:
            raise KeyError("Column 'image' not found in DataFrame")

    def clean_captions(self):
        if 'caption' in self.data.columns:
            self.data['caption'] = self.data['caption'].apply(lambda x: re.sub(r'["]', '', x))
        else:
            raise KeyError("Column 'caption' not found in DataFrame")

