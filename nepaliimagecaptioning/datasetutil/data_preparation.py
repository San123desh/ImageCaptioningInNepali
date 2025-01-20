

import os
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input


class DataPreparation:
    @staticmethod
    def load_captions(captions_path):
        """Load captions from a file."""
        with open(captions_path, 'r', encoding='utf-8') as f:
            next(f)  
            return f.read()

    @staticmethod
    def save_features(features, path):
        """Save extracted features to a file."""
        with open(path, 'wb') as f:
            pickle.dump(features, f)

    @staticmethod
    def load_features(path):
        """Load extracted features from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
