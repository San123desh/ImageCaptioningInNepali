
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import re

class ImageCaptioningDataPreprocessor:
    def __init__(self, image_path, captions_file, tokenizer=None):
        self.image_path = image_path
        self.data = self.load_and_process_captions(captions_file)
        
        self.tokenizer = tokenizer if tokenizer else Tokenizer(oov_token="<UNK>")
        self.vocab_size = 0
        self.max_length = 0

    def load_and_process_captions(self, captions_file):
        # Load and process the captions file.
        data = []
        with open(captions_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(',', 1)  
                if len(parts) == 2:
                    image, caption = parts 
                    data.append((image, caption))
        return pd.DataFrame(data, columns=['image', 'caption'])

    def preprocess_text(self):
        # Preprocess Nepali captions by cleaning and adding start/end tokens.
        if 'caption' not in self.data.columns:
            raise KeyError("Column 'caption' not found in DataFrame")
            # self.data['caption'].fillna('', inplace=True)
        self.data['caption'] = self.data['caption'].fillna('')
        self.data['caption'] = self.data['caption'].apply(lambda x: re.sub(r'["]', '', x))
        self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())
        self.data['caption'] = self.data['caption'].apply(lambda x: "startseq " + x + " endseq")
    
    def prepare_tokenizer(self):
        if 'caption' not in self.data.columns:
            raise KeyError("Column 'caption' not found in DataFrame")
        # Fit the tokenizer on Nepali captions and calculate vocab size and max length
        captions = self.data['caption'].tolist()
        self.tokenizer.fit_on_texts(captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_length = max(len(word_tokenize(caption)) for caption in captions)

    def split_data(self, train_ratio=0.85):
        # Split data into training and validation sets
        if 'image' not in self.data.columns:
            raise KeyError("Column 'image' not found in DataFrame")
        images = self.data['image'].unique().tolist()
        split_index = round(train_ratio * len(images))
        train_images = images[:split_index]
        val_images = images[split_index:]
        train_data = self.data[self.data['image'].isin(train_images)]
        val_data = self.data[self.data['image'].isin(val_images)]

        return train_data, val_data
        

    def clean_captions(self):
        # Remove double quotes and other problematic characters
        if 'caption' in self.data.columns:
            self.data['caption'] = self.data['caption'].apply(lambda x: re.sub(r'["]', '', x))
        else:
            raise KeyError("Column 'caption' not found in DataFrame")
