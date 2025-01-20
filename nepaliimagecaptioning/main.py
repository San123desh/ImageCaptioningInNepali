

# import os

# # Set environment variable to avoid OpenMP runtime issues 
# # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# from datasetutil.caption_processing import create_mapping, clean_mapping, create_tokenizer, get_max_length
# from datasetutil.data_preparation import load_captions, save_features, load_features
# from datasetutil.feature_extraction import load_vgg16, extract_features
# from domain.model_definition import define_model
# from service.trainer import train_model, save_model
# from service.evaluation import evaluate_model, generate_caption
# from keras.models import Model

# BASE_DIR = 'data/Flickr8k_Dataset'
# WORKING_DIR = 'working'

# # Load and preprocess captions
# captions_path = os.path.join(BASE_DIR, 'captions.txt')
# captions_doc = load_captions(captions_path)
# mapping = create_mapping(captions_doc)
# clean_mapping(mapping)

# all_captions = []
# for key in mapping:
#     for caption in mapping[key]:
#         all_captions.append(caption)

# # Create tokenizer and get max caption length
# tokenizer = create_tokenizer(all_captions)
# vocab_size = len(tokenizer.word_index) + 1
# max_length = get_max_length(all_captions)

# # Split data into training and testing sets
# image_ids = list(mapping.keys())
# split = int(len(image_ids) * 0.90)
# train = image_ids[:split]
# test = image_ids[split:]

# # Load VGG16 model and extract features
# model_vgg = load_vgg16()
# directory = os.path.join(BASE_DIR, 'Images')
# features = extract_features(model_vgg, directory)
# save_features(features, os.path.join(WORKING_DIR, 'features.pkl'))

# # Define and compile the image captioning model
# model = define_model(vocab_size, max_length)

# # Train the model
# epochs = 20
# batch_size = 32
# steps = len(train) // batch_size
# train_model(model, train, mapping, features, tokenizer, max_length, vocab_size, batch_size, epochs, steps)

# # Save the trained model
# save_model(model, os.path.join(WORKING_DIR, 'model.h5'))

# # Evaluate the model
# features = load_features(os.path.join(WORKING_DIR, 'features.pkl'))
# bleu1, bleu2 = evaluate_model(model, test, mapping, features, tokenizer, max_length)
# print(f"BLEU-1: {bleu1:.6f}")
# print(f"BLEU-2: {bleu2:.6f}")

# # Generate captions for sample images
# generate_caption("1001773457_577c3a7d70.jpg", model, mapping, features, tokenizer, max_length, BASE_DIR)
# generate_caption("1002674143_1b742ab4b8.jpg", model, mapping, features, tokenizer, max_length, BASE_DIR)


# # # Assuming `model` is your full encoder-decoder model
# # intermediate_layer = model.get_layer('dense')  # Replace 'encoder_dense' with your layer's name
# # intermediate_model = Model(inputs=model.input, outputs=intermediate_layer.output)

# # for layer in model.layers:
# #     print(layer.name)


import os
from datasetutil.caption_processing import CaptionProcessor
from datasetutil.data_preparation import DataPreparation
from datasetutil.feature_extraction import FeatureExtractor
from domain.model_definition import ModelDefinition
from service.trainer import Trainer
from service.evaluation import Evaluation


class ImageCaptioningPipeline:
    def __init__(self, base_dir, working_dir):
        """
        Initialize the ImageCaptioningPipeline class.

        :param base_dir: Base directory for the dataset
        :param working_dir: Directory to store intermediate files
        """
        self.base_dir = base_dir
        self.working_dir = working_dir
        self.caption_processor = CaptionProcessor()
        self.data_preparation = DataPreparation()
        self.feature_extractor = FeatureExtractor()
        self.model_builder = ModelDefinition()
        self.trainer = None  # To be initialized later
        self.evaluator = None  # To be initialized later

    def run(self):
        """
        Run the image captioning pipeline.
        """
        # Load and preprocess captions
        captions_path = os.path.join(self.base_dir, 'captions.txt')
        captions_doc = self.data_preparation.load_captions(captions_path)
        mapping = self.caption_processor.create_mapping(captions_doc)
        self.caption_processor.clean_mapping(mapping)

        # Prepare tokenizer and get max caption length
        all_captions = [caption for captions in mapping.values() for caption in captions]
        tokenizer = self.caption_processor.create_tokenizer(all_captions)
        vocab_size = len(tokenizer.word_index) + 1
        max_length = self.caption_processor.get_max_length(all_captions)

        # Split data into training and testing sets
        image_ids = list(mapping.keys())
        split_index = int(len(image_ids) * 0.90)
        train_keys = image_ids[:split_index]
        test_keys = image_ids[split_index:]

        # Extract features using VGG16
        model_vgg = self.feature_extractor.load_vgg16()
        image_directory = os.path.join(self.base_dir, 'Images')
        features = self.feature_extractor.extract_features(image_directory)
        features_path = os.path.join(self.working_dir, 'features.pkl')
        self.data_preparation.save_features(features, features_path)

        # Define and compile the model
        model = self.model_builder.define_model(vocab_size, max_length)
        self.trainer = Trainer(model, tokenizer, max_length, vocab_size)

        # Train the model
        epochs = 10
        batch_size = 32
        steps_per_epoch = len(train_keys) // batch_size
        self.trainer.train(train_keys, mapping, features, batch_size, epochs, steps_per_epoch)

        # Save the trained model
        model_path = os.path.join(self.working_dir, 'model.h5')
        self.trainer.save_model(model, model_path)

        # Evaluate the model
        features = self.data_preparation.load_features(features_path)
        self.evaluator = Evaluation(model, tokenizer, max_length)
        bleu1, bleu2 = self.evaluator.evaluate_model(test_keys, mapping, features)
        print(f"BLEU-1: {bleu1:.6f}")
        print(f"BLEU-2: {bleu2:.6f}")

        # Generate captions for sample images
        self.evaluator.generate_caption(
            "1001773457_577c3a7d70.jpg", mapping, features, self.base_dir
        )
        self.evaluator.generate_caption(
            "1002674143_1b742ab4b8.jpg", mapping, features, self.base_dir
        )


if __name__ == "__main__":
    BASE_DIR = 'data/Flickr8k_Dataset'
    WORKING_DIR = 'working'

    pipeline = ImageCaptioningPipeline(BASE_DIR, WORKING_DIR)
    pipeline.run()
