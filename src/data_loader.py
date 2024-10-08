import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm


class DataLoader:
    def __init__(self, image_dir, captions_file, image_size=(224,224)):
        self.image_dir = image_dir
        self.captions_file = captions_file
        self.image_size = image_size
        # self.image_captions = self.load_captions()

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.captions = pd.read_csv(self.captions_file)

    def load_images(self):
        images = []
        for img_file in tqdm(os.listdir(self.image_dir)):
            try:
                img_path = os.path.join(self.image_dir, img_file)
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
                images.append(image)
            except Exception as e:
                print(f"Error loading image {img_file}: {e}")
        return torch.stack(images)

    def load_captions(self):
        captions = self.captions['caption'].tolist()
        return captions
            
    