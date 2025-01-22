

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
