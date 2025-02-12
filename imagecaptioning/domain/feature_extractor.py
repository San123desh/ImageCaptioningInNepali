import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class FeatureExtractor:
    def __init__(self, image_path, img_size=224):
        
        self.image_path = image_path
        self.img_size = img_size
        self.model = DenseNet201()
        self.feature_extractor = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)

    def extract_features(self, image_paths):
       
        features = {}
        for image in tqdm(image_paths):
            img = load_img(os.path.join(self.image_path, image), target_size=(self.img_size, self.img_size))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            feature = self.feature_extractor.predict(img, verbose=0)
            features[image] = feature
        return features
    










    