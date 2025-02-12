import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from datasetutils.data_preprocessor import ImageCaptioningDataPreprocessor
from domain.feature_extractor import FeatureExtractor
from datasetutils.data_generator import CustomDataGenerator
from imagecaptioning_onwork.service.caption_generation_model import ImageCaptioningModel
from utils.logging_utils import setup_logger
from tensorflow.keras.preprocessing.text import Tokenizer

if __name__ == '__main__':
    logger = setup_logger()
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

    # Ensure vocab size is correct
    vocab_size = preprocessor.vocab_size + 1

    # Create data generators
    train_generator = CustomDataGenerator(
        df=train_data,
        X_col='image',
        y_col='caption',
        batch_size=64,
        tokenizer=preprocessor.tokenizer,
        # vocab_size=preprocessor.vocab_size,
        vocab_size= vocab_size,
        max_length=preprocessor.max_length,
        features=features
    )
    val_generator = CustomDataGenerator(
        df=val_data,
        X_col='image',
        y_col='caption',
        batch_size=64,
        tokenizer=preprocessor.tokenizer,
        # vocab_size=preprocessor.vocab_size,
        vocab_size=vocab_size,
        max_length=preprocessor.max_length,
        features=features
    )
    print("Checking Data Generator Output...")
    try:
        (X1, X2), y = next(iter(train_generator))
        print(f"X1 (Image Features) Shape: {X1.shape}")
        print(f" X2 (Tokenized Captions) Shape: {X2.shape}")
        print(f" y (One-hot Captions) Shape: {y.shape}")
    except Exception as e:
        print(f"Data Generator Error: {e}")
        exit(1) 

    # Build and train the model
    # caption_model = ImageCaptioningModel(vocab_size=preprocessor.vocab_size, max_length=preprocessor.max_length)
    caption_model = ImageCaptioningModel(vocab_size=vocab_size, max_length=preprocessor.max_length)
    # Check input shapes
    (X1, X2), y = train_generator.__getitem__(0)
    print(f'X1 Shape: {X1.shape}, X2 Shape: {X2.shape}, y Shape: {y.shape}')
    
    # Ensure input dtype is correct
    X1 = np.array(X1, dtype=np.float32)  # Convert image features to float32
    X2 = np.array(X2, dtype=np.int32)  # Convert tokenized captions to int32
    y = np.array(y, dtype=np.float32)  # Ensure labels are float32
    history = caption_model.train(train_generator, val_generator, preprocessor.tokenizer, max_length=preprocessor.max_length,feature_extractor=feature_extractor)
