import os
import cv2
import torch
import numpy as np
from vit_encoder import ViTEncoder
from gpt2_decoder import GPT2Decoder
from translator import NepaliTranslator
# from visualization import AttentionVisualizer

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error reading image {image_path}")
    
    if image.shape[-1] == 1:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))  # Resize to match the input dimensions of the ViT model
    
    return image

def main():
    # Set paths
    image_dir = 'data/images/'  # Directory containing images
    captions_file = 'data/captions.csv'  # Placeholder for future use
    
    # Initialize the ViT encoder and GPT-2 decoder
    vit_encoder = ViTEncoder(model_name='google/vit-base-patch16-224')
    gpt2_decoder = GPT2Decoder(model_name='gpt2')
    nepali_translator = NepaliTranslator()
    
    # Load ViT model for visualization
    # attention_visualizer = AttentionVisualizer(vit_encoder.model)
    
    # Extract image features using the ViT model
    image_features_batch = []
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        image = preprocess_image(image_path)
        image_features = vit_encoder.extract_features(image_path)  # Assuming you have this method
        image_features_batch.append(image_features)
    
    # Generate captions for each image using GPT-2
    captions = gpt2_decoder.generate_batch_captions(image_features_batch)
    
    # Display the generated captions and visualize attention
    for idx, (image_file, image_features) in enumerate(zip(os.listdir(image_dir), image_features_batch)):
        image_path = os.path.join(image_dir, image_file)
        image = preprocess_image(image_path)
        
        # Translate the English caption to Nepali
        translated_caption, tokenized_caption = nepali_translator.translate_and_tokenize(captions[idx])
        
        print(f"Image {idx + 1}:")
        print(f"English Caption: {captions[idx]}")
        print(f"Nepali Caption: {translated_caption}")
        print(f"Tokenized Nepali: {tokenized_caption}")
        print()
        
        # Visualizing attention for the image
        # print(f"Visualizing attention for Image {idx + 1}: {image_file}")
        # image_tensor = torch.from_numpy(np.array(image_features)).permute(0,3,1,2).float().unsqueeze(0)  # Add batch dimension

        # print(f"Image Features Shape: {image_features.shape}")
        # if image_features.dim() == 3:
        #     image_tensor = torch.from_numpy(np.array(image_features)).unsqueeze(0).permute(0, 3, 1, 2).float()  # Add batch dimension and permute
        # else:
        #     raise ValueError(f"Expected 3 dimensions, got {image_features.dim()} dimensions")
        
        # print(f"Image Tensor Shape: {image_tensor.shape}")
        # image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
        # print(f"Image Tensor Shape: {image_tensor.shape}")
        
        # Visualize the attention using ViT features
        # attention_visualizer.visualize(image_tensor, image)
        
        # Handle attention mask and pad token for GPT-2 caption generation
        input_ids = gpt2_decoder.tokenizer.encode(tokenized_caption, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape)  # Attention mask: 1s for real tokens
        
        # Generate caption using GPT-2 decoder
        generated_caption = gpt2_decoder.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # Pass attention mask
            pad_token_id=gpt2_decoder.tokenizer.eos_token_id  # Handle pad token
        )
        
        generated_text = gpt2_decoder.tokenizer.decode(generated_caption[0], skip_special_tokens=True)
        print(f"Generated Nepali Caption: {generated_text}")

if __name__ == "__main__":
    main()
