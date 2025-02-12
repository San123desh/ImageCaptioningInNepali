


# import os
# import numpy as np
# from tqdm import tqdm
# from tensorflow.keras.applications import DenseNet201
# from tensorflow.keras.applications.densenet import preprocess_input
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.layers import GlobalAveragePooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras import Input

# class FeatureExtractor:
#     def __init__(self, image_path, img_size=224):
       
#         self.image_path = image_path
#         self.img_size = img_size
#         # self.model = DenseNet201(weights="imagenet", include_top=False)
#         # self.feature_extractor = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
#         # base_model = DenseNet201(weights="imagenet", include_top=False)
#         # self.feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
#         base_model = DenseNet201(weights="imagenet", include_top=False)
#         output = GlobalAveragePooling2D()(base_model.output)  # Flatten feature map
#         self.feature_extractor = Model(inputs=base_model.input, outputs=output)
#     def extract_features(self, image_paths):
#         features = {}
#         for image in tqdm(image_paths, desc="Extracting Features"):
#             try:
#                 img = load_img(os.path.join(self.image_path, image), target_size=(self.img_size, self.img_size))
#                 img = img_to_array(img)
#                 img = np.expand_dims(img, axis=0)
#                 img = preprocess_input(img)  # Ensure proper normalization

#                 feature = self.feature_extractor.predict(img, verbose=0)
#                 feature = np.squeeze(feature)  # Remove unnecessary dimensions
                
#                 features[image] = feature
#             except Exception as e:
#                 print(f"Error processing {image}: {e}")
#     # def extract_features(self, image_paths):
#     #     features = {}
#     #     for image in tqdm(image_paths):
#     #         try:
#     #             img = load_img(os.path.join(self.image_path, image), target_size=(self.img_size, self.img_size))
#     #             img = img_to_array(img) / 255.0
#     #             img = np.expand_dims(img, axis=0)
#     #             feature = self.feature_extractor.predict(img, verbose=0)
#     #             features[image] = feature
#     #         except Exception as e:
#     #             print(f"Error processing {image}: {e}")
#         return features



import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D

class FeatureExtractor:
    def __init__(self, image_path, img_size=224):
        self.image_path = image_path
        self.img_size = img_size
        
        # Load DenseNet201 without top layers
        base_model = DenseNet201(weights="imagenet", include_top=False)
        base_model.trainable = False  # Freeze base model
        output = GlobalAveragePooling2D()(base_model.output)  # Convert to (1920,)

        self.feature_extractor = Model(inputs=base_model.input, outputs=output)

    def extract_features(self, image_paths):
        features = {}
        for image in tqdm(image_paths, desc="Extracting Features"):
            try:
                # Load image
                img_path = os.path.join(self.image_path, image)
                img = load_img(img_path, target_size=(self.img_size, self.img_size), color_mode="rgb")
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)  # Shape should be (1, 224, 224, 3)
                img = preprocess_input(img)

                # Validate image shape
                if img.shape != (1, 224, 224, 3):
                    print(f"Unexpected image shape: {img.shape} for {image}. Skipping.")
                    continue

                # Extract features
                feature = self.feature_extractor.predict(img, verbose=0)
                # print(f"Feature shape before squeeze: {feature.shape}")  # Debugging

                # Validate extracted feature shape
                if feature.shape != (1, 1920):
                    print(f"Warning: Invalid feature shape for {image}. Expected (1, 1920), found {feature.shape}. Skipping.")
                    continue

                feature = np.squeeze(feature)  # Ensure (1920,)
                feature = feature.astype(np.float32)  # Convert to float32

                features[image] = feature
            except Exception as e:
                print(f"Error processing {image}: {e}")
        return features
