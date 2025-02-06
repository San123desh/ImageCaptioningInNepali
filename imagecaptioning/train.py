import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from datasetutils.data_preprocessor import ImageCaptioningDataPreprocessor
from domain.feature_extractor import FeatureExtractor
from datasetutils.data_generator import CustomDataGenerator
from imagecaptioning.service.image_captioning_model import ImageCaptioningModel
from utils.logging_utils import setup_logger
from tensorflow.keras.preprocessing.text import Tokenizer

if __name__ == '__main__':
    logger = setup_logger()
    # Define the base directory
    # BASE_DIR = 'checking/Flickr8k_Dataset'
    BASE_DIR = 'Flickr8k_Dataset'
    # Initialize preprocessor
    tokenizer = Tokenizer()
    preprocessor = ImageCaptioningDataPreprocessor(
        image_path=os.path.join(BASE_DIR, 'Images'),
        captions_file=os.path.join(BASE_DIR, 'captions.txt'),
        tokenizer=tokenizer
    )
    preprocessor.preprocess_text()
    preprocessor.prepare_tokenizer()
    train_data, val_data = preprocessor.split_data()

    # Extract features
    feature_extractor = FeatureExtractor(image_path=os.path.join(BASE_DIR, 'Images'))
    features = feature_extractor.extract_features(preprocessor.data['image'].unique().tolist())

    # Create data generators
    train_generator = CustomDataGenerator(
        df=train_data,
        X_col='image',
        y_col='caption',
        batch_size=64,
        tokenizer=preprocessor.tokenizer,
        vocab_size=preprocessor.vocab_size,
        max_length=preprocessor.max_length,
        features=features
    )
    val_generator = CustomDataGenerator(
        df=val_data,
        X_col='image',
        y_col='caption',
        batch_size=64,
        tokenizer=preprocessor.tokenizer,
        vocab_size=preprocessor.vocab_size,
        max_length=preprocessor.max_length,
        features=features
    )

    # Build and train the model
    caption_model = ImageCaptioningModel(vocab_size=preprocessor.vocab_size, max_length=preprocessor.max_length)
    history = caption_model.train(train_generator, val_generator, preprocessor.tokenizer, preprocessor.max_length)
