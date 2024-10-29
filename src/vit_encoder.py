import torch
from transformers import ViTModel, ViTFeatureExtractor, ViTImageProcessor
from PIL import Image
import os
# import torchvision.transforms as transforms


class ViTEncoder:
    def __init__(self, model_name='google/vit-base-patch16-224'):
        # Load the pre-trained ViT model and feature extractor
        self.model = ViTModel.from_pretrained(model_name, output_attentions=True)
        # self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        # self.image_processor = ViTImageProcessor.from_pretrained(model_name)
        self.feature_processor = ViTImageProcessor.from_pretrained(model_name)


    def extract_features(self, image_path, image_tensor):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        # inputs = self.feature_extractor(images=image, return_tensors="pt")
        # inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = self.feature_processor(images=image, return_tensors="pt")

        
        # Pass the preprocessed image through the ViT model to get image features
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract the last hidden state as image features
        image_features = outputs.last_hidden_state
        return image_features
        
        

    def process_batch(self, image_dir):
        features = []
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            image_features = self.extract_features(image_path)
            features.append(image_features)
        return features


