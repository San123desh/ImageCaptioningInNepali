import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from domain.feature_extraction import ImageFeatureExtractor
from datasetutil.caption_processing import CaptionProcessor
from utils.model_utils import DataHandler
from utils.embedding_utils import load_glove_embeddings, create_embedding_matrix
from service.evaluation import CaptionEvaluator
from service.train import Trainer
from domain.model_definition import ImageCaptionModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Directories

BASE_DIR = 'Flickr8k_Dataset'
WORKING_DIR = 'working'
GLOVE_FILEPATH = 'data/glove/glove.6B.300d.txt'

# Ensure directories exist
# WORKING_DIR.mkdir(parents=True, exist_ok=True)


# Load and preprocess captions
data_handler = DataHandler()
captions_path = os.path.join(BASE_DIR, 'captions.txt')
captions_doc = data_handler.load_captions(captions_path)

caption_processor = CaptionProcessor()
mapping = caption_processor.create_mapping(captions_doc)
caption_processor.clean_mapping(mapping)

# Prepare tokenizer and dataset
all_captions = [caption for captions in mapping.values() for caption in captions]
tokenizer = caption_processor.create_tokenizer(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = caption_processor.get_max_length(all_captions)

# Load GloVe embeddings
embedding_dim = 300
embeddings_index = load_glove_embeddings(GLOVE_FILEPATH, embedding_dim)
embedding_matrix = create_embedding_matrix(tokenizer, embeddings_index, embedding_dim)

# Validate special tokens
assert "startseq" in tokenizer.word_index, "Error: 'startseq' missing in tokenizer."
assert "endseq" in tokenizer.word_index, "Error: 'endseq' missing in tokenizer."

# Debugging Tokenizer
print("Tokenizer Vocabulary Size:", len(tokenizer.word_index))
print("Sample Tokenizer Mapping (First 10):", {k: tokenizer.word_index[k] for k in list(tokenizer.word_index)[:10]})

# Split Dataset
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.70)
train = image_ids[:split]
test = image_ids[split:]

# Load InceptionV3 model and extract features
feature_extractor = ImageFeatureExtractor()
features = feature_extractor.extract_features(os.path.join(BASE_DIR, 'Images'))

# Save features for later use
data_handler.save_features(features, os.path.join(WORKING_DIR, 'features.pkl'))

# Debugging Missing Keys
missing_keys = [key for key in mapping.keys() if key not in features]
if missing_keys:
    print(f"Warning: Missing feature keys (first 5): {missing_keys[:5]}")
else:
    print("All keys are properly aligned between mapping and features.")

# Train Model
image_caption_model = ImageCaptionModel(vocab_size, max_length, embedding_matrix)
model = image_caption_model.get_model()
epochs = 150
batch_size = 32
steps = len(train) // batch_size

# Create Directories for Checkpoints and Logs
os.makedirs("checkpoints", exist_ok=True)
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1, min_lr=1e-4)
model_checkpoint_callback = ModelCheckpoint(filepath="checkpoints/model-{epoch:02d}.keras", save_best_only=True, monitor='loss', mode='min', verbose=1)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

trainer = Trainer(model, tokenizer, max_length, vocab_size)
generator = trainer.data_generator(train, mapping, features, batch_size)
model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1, callbacks=[model_checkpoint_callback, tensorboard_callback, lr_scheduler])

# Save the trained model
trainer.save_model(model, os.path.join(WORKING_DIR, 'model.h5'))

# Evaluate and Generate Captions
features = data_handler.load_features(os.path.join(WORKING_DIR, 'features.pkl'))
evaluator = CaptionEvaluator(model, tokenizer, max_length)

# BLEU Scores Evaluation
bleu_scores = evaluator.evaluate_model(test, mapping, features)
print("BLEU-Scores:", bleu_scores)

# Generate Captions for Sample Images
sample_images = ["10815824_2997e03d76.jpg", "23445819_3a458716c1.jpg", "3703960010_1e4c922a25.jpg", "36422830_55c844bc2d.jpg"]
for image_name in sample_images:
    evaluator.generate_caption(image_name, mapping, features, BASE_DIR)
    print("\n")


