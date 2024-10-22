import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, encoder, decoder, dataloader, device):
        self.encoder = encoder
        self.decoder = decoder
        self.dataloader = dataloader
        self.device = device

    def train(self):
        images = self.dataloader.load_images().to(self.device)
        captions = self.dataloader.load_captions()

        for i, image in tqdm(enumerate(images), total=len(images)):
            image_embedding = self.encoder.encode_image(image)
            generated_caption = self.decoder.generate_caption(image_embedding)

            print(f"Image {i+1}:")
            print(f"Generated Caption: {generated_caption}")
            print(f"Original Caption: {captions[i]}")
            print('-' * 50)
