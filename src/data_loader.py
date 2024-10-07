import os

class DataLoader:
    def __init__(self, image_dir, captions_file):
        self.image_dir = image_dir
        self.captions_file = captions_file
        # self.image_captions = self.load_captions()

    def load_images(self):
        image_files = os.listdir(self.image_dir)
        return image_files
    
    


    def load_captions(self):
        with open(self.captions_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            captions = {}
            for line in lines:
                image_id, caption = line.strip().split('\t')
                if image_id not in captions:
                    captions[image_id] = []
                captions[image_id].append(caption)
                # print(f'Image: {image_id}, Caption: {caption}')

                # TODO: Preprocess captions (e.g., convert to lowercase, remove punctuation, etc.)
                # captions[image_id] = preprocess_caption(caption)
                return captions
            
    