# utils/model_utils.py
# def save_model(model, path):
#     model.save(path)

import os
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

class DataHandler:
    def __init__(self):
        pass

    @staticmethod
    def load_captions(captions_path):
        with open(captions_path, 'r', encoding='utf-8') as f:
            next(f)
            return f.read()

    @staticmethod
    def save_features(features, path):
        with open(path, 'wb') as f:
            pickle.dump(features, f)

    @staticmethod
    def load_features(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


