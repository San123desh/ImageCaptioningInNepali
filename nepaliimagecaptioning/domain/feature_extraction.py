
# import os
# from tqdm import tqdm
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# class FeatureExtractor:
#     def __init__(self):
        
#         self.model = self.load_vgg16()

#     def load_vgg16(self):
        
#         model = VGG16()
#         model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
#         return model

#     def extract_features(self, directory):
       
#         features = {}
#         if not os.path.exists(directory):
#             print(f"The directory {directory} does not exist. Please check the path.")
#         else:
#             for img_name in tqdm(os.listdir(directory), desc="Extracting features"):
#                 img_path = os.path.join(directory, img_name)
#                 try:
#                     image = load_img(img_path, target_size=(224, 224))
#                     image = img_to_array(image)
#                     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#                     image = preprocess_input(image)
#                     feature = self.model.predict(image, verbose=0)
#                     image_id = img_name.split('.')[0]
#                     features[image_id] = feature
#                 except Exception as e:
#                     print(f"Error processing {img_name}: {e}")
#         return features

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import os

class ImageFeatureExtractor:
    def __init__(self):
        self.model = self.load_inceptionv3()

    def load_inceptionv3(self):
        from tensorflow.keras.layers import GlobalAveragePooling2D
        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = GlobalAveragePooling2D()(base_model.output)
        image_model = Model(inputs=base_model.input, outputs=x)
        return image_model

    def extract_features(self, directory):
        features = {}
        if not os.path.exists(directory):
            print(f"Error: The directory {directory} does not exist. Please check the path.")
            return features

        for img_name in tqdm(os.listdir(directory), desc="Extracting features"):
            img_path = os.path.join(directory, img_name)
            if not img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                print(f"Skipping non-image file: {img_name}")
                continue

            try:
                image = load_img(img_path, target_size=(299, 299))
                image = img_to_array(image)
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                image = preprocess_input(image)
                feature = self.model.predict(image, verbose=0)
                image_id = os.path.splitext(img_name)[0]
                features[image_id] = feature
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

        return features
