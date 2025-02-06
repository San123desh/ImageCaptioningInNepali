

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import re

class ImageCaptioningDataPreprocessor:
    def __init__(self, image_path, captions_file):
        self.image_path = image_path
        self.data = self.load_and_process_captions(captions_file)
        
        # # Debugging: Print the columns and first few rows
        # print("Columns in DataFrame:", self.data.columns)
        # print("First few rows of DataFrame:\n", self.data.head())

        self.tokenizer = Tokenizer()
        self.vocab_size = 0
        self.max_length = 0

    def load_and_process_captions(self, captions_file):
        """Load and process the captions file."""
        data = []
        with open(captions_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(',', 1)  # Split on the first space
                if len(parts) == 2:
                    image, caption = parts
                    # image = image.split('#')[0]  # Remove the suffix (e.g., '#0')
                    data.append((image, caption))
        return pd.DataFrame(data, columns=['image', 'caption'])

    def preprocess_text(self):
        """Preprocess Nepali captions by cleaning and adding start/end tokens."""
        if 'caption' in self.data.columns:
            # self.data['caption'].fillna('', inplace=True)
            self.data['caption'] = self.data['caption'].fillna('')
            self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())
            self.data['caption'] = self.data['caption'].apply(lambda x: "startseq " + x + " endseq")
        else:
            raise KeyError("Column 'caption' not found in DataFrame")

    def prepare_tokenizer(self):
        """Fit the tokenizer on Nepali captions and calculate vocab size and max length."""
        captions = self.data['caption'].tolist()
        self.tokenizer.fit_on_texts(captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_length = max(len(word_tokenize(caption)) for caption in captions)

    def split_data(self, train_ratio=0.85):
        """Split data into training and validation sets."""
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
        """Remove double quotes and other problematic characters."""
        if 'caption' in self.data.columns:
            self.data['caption'] = self.data['caption'].apply(lambda x: re.sub(r'["]', '', x))
        else:
            raise KeyError("Column 'caption' not found in DataFrame")

