import os
import cv2
import torch
import pandas as pd
from vit_encoder import ViTEncoder
from gpt2_decoder import GPT2Decoder
from translator import NepaliTranslator
from torchvision import transforms
from PIL import Image


def preprocess_image(image_path):
    # image = cv2.imread(image_path)
    # if image is None:
    #     raise ValueError(f"Error reading image {image_path}")
    
    # if image.shape[-1] == 1:  # If grayscale, convert to RGB
    #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # else:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image = cv2.resize(image, (224, 224))  # Resize to match the input dimensions of the ViT model
    
    # return image

    """Preprocesses an image to be compatible with the ViT model."""
    image = Image.open(image_path)

    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply the transformations and add a batch dimension
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor





def main():
    # Set paths
    image_dir = 'data/images/'  # Directory containing images
    captions_file = 'data/captions.csv'  # Path to your CSV file

    # Load the captions CSV file
    df = pd.read_csv(captions_file, encoding='ISO-8859-1')


    # Initialize the ViT encoder and GPT-2 decoder
    vit_encoder = ViTEncoder(model_name='google/vit-base-patch16-224')
    gpt2_decoder = GPT2Decoder(model_name='gpt2')
    nepali_translator = NepaliTranslator()

    # Extract image features and generate captions
    for idx, row in df.iterrows():
        image_file = row['image_id']
        english_caption = row['caption']  # Captions in English from CSV
        image_path = os.path.join(image_dir, image_file)
        
        try:
            # image = preprocess_image(image_path)
            image_tensor = preprocess_image(image_path)
        except ValueError as e:
            print(e)
            continue  # Skip this image if it's not found or unreadable

        # image_features = vit_encoder.extract_features(image_path)  # Extract image features using ViT
        image_tensor = preprocess_image(image_path)
        # image_features = vit_encoder.extract_features(image_tensor)

        # generated_caption = gpt2_decoder.generate_caption(image_features)

        # Check if the caption is empty before translation
        if not english_caption:
            print(f"Skipping translation for Image {idx + 1}: Caption is empty.")
            continue
        
        # Translate the English caption to Nepali
        translated_caption, tokenized_caption = nepali_translator.translate_and_tokenize(english_caption)

        if not translated_caption:
            print(f"Translation failed for Image {idx + 1}. Skipping this image.")
            continue  # Skip if translation fails
        
        # Print or store the results
        print(f"Image {idx + 1}: {image_file}")
        print(f"English Caption: {english_caption}")
        print(f"Nepali Caption: {translated_caption}")
        print(f"Tokenized Nepali Caption: {tokenized_caption}")
        print()


if __name__ == "__main__":
    main()
