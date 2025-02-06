import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from tensorflow.keras.models import load_model
from utils.file_utils import load_tokenizer
from service.caption_generator import CaptionGenerator

# Define the base directory
BASE_DIR = 'checking/Flickr8k_Dataset'

def generate_caption_for_new_image(image_path):
    # Create a CaptionGenerator instance with paths to the model, tokenizer, and feature extractor
    caption_generator = CaptionGenerator(
        model_path="model.keras", 
        tokenizer_path="tokenizer.pkl",
        feature_extractor_path="feature_extractor.keras"
    )
    
    # Generate the caption for the new image
    caption_generator.generate_caption(image_path)

if __name__ == "__main__":
    # Example usage
    new_image_path = "road.jpeg"
    full_image_path = os.path.abspath(os.path.join(BASE_DIR, new_image_path))
    
    # Print the full path to check if it's correct
    print(f"Full image path: {full_image_path}")
    
    # Check if the file exists at the specified path
    if os.path.isfile(full_image_path):
        generate_caption_for_new_image(full_image_path)
    else:
        print(f"File not found: {full_image_path}")
