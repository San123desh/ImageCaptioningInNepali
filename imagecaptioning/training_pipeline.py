import os
from tensorflow.keras.models import load_model
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from datasetutils.data_preprocessor import ImageCaptioningDataPreprocessor
from domain.feature_extractor import FeatureExtractor
from datasetutils.data_generator import CustomDataGenerator
from service.image_captioning_model import ImageCaptioningModel
from service.caption_generator import CaptionGenerator
from utils.file_utils import save_tokenizer
from service.model_trainer import ModelTrainer
from utils.logging_utils import setup_logger, log_metrics
from utils.metrics import calculate_corpus_bleu_score, tokenize_captions
import numpy as np

# Define the base directory
BASE_DIR = 'Flickr8k_Dataset'
# Initialize preprocessor
preprocessor = ImageCaptioningDataPreprocessor(
    image_path=os.path.join(BASE_DIR, 'Images'),
    captions_file=os.path.join(BASE_DIR, 'captions.txt')
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

# Set up logger+
logger = setup_logger()

# Build and train the model
caption_model = ImageCaptioningModel(vocab_size=preprocessor.vocab_size, max_length=preprocessor.max_length)
model_trainer = ModelTrainer(caption_model.model)
history = model_trainer.train(train_generator, val_generator, preprocessor.tokenizer, preprocessor.max_length)

# Log training history
for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
    log_metrics(logger, epoch, loss, val_loss, history.history.get('val_bleu', [None] * len(history.history['loss']))[epoch])

# Define the new directory for saving the models and tokenizer
NEW_DIR = 'testingmodel'

# Ensure the directory exists
os.makedirs(NEW_DIR, exist_ok=True)

# Save tokenizer and feature extractor
save_tokenizer(preprocessor.tokenizer, os.path.join(NEW_DIR, "tokenizer.pkl"))
feature_extractor.feature_extractor.save(os.path.join(NEW_DIR, "feature_extractor.keras"))

# Save the trained model
caption_model.model.save(os.path.join(NEW_DIR, "model.keras"))

# Generate captions for a sample image
caption_generator = CaptionGenerator(
    model_path=os.path.join(NEW_DIR, "model.keras"), 
    tokenizer_path=os.path.join(NEW_DIR, "tokenizer.pkl"),
    feature_extractor_path=os.path.join(NEW_DIR, "feature_extractor.keras")
)
caption_generator.generate_caption(os.path.join(BASE_DIR,'Images','12830823_87d2654e31.jpg'))

# Evaluate BLEU score on the validation set
def evaluate_bleu_score(model, val_generator, tokenizer):
    references = []
    candidates = []

    for batch in val_generator:
        (X1, X2), y_true = batch
        y_pred = model.predict([X1, X2], verbose=0)
        y_pred_indices = np.argmax(y_pred, axis=-1)

        for i in range(len(y_true)):
            true_caption = tokenizer.sequences_to_texts([np.argmax(y_true[i], axis=-1)])[0]
            pred_caption = tokenizer.sequences_to_texts([y_pred_indices[i]])[0]

            # Tokenize the captions
            references.append([tokenize_captions([true_caption])[0]])
            candidates.append(tokenize_captions([pred_caption])[0])

    # Calculate corpus BLEU score
    bleu_score = calculate_corpus_bleu_score(references, candidates)
    print(f"Validation BLEU Score: {bleu_score:.4f}")
    return bleu_score

# Evaluate BLEU score on the validation set
bleu_score = evaluate_bleu_score(caption_model.model, val_generator, preprocessor.tokenizer)
logger.info(f"Final Validation BLEU Score: {bleu_score:.4f}")
