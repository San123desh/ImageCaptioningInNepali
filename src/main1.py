import datetime
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import pad_sequences, to_categorical
from caption_processing import create_mapping, clean_mapping, create_tokenizer, get_max_length
from data_preparation import load_captions, save_features, load_features
# from feature_extraction import load_vgg16, extract_features
from feature_extraction import load_inceptionv3, extract_features
from evaluation import evaluate_model, generate_caption
import numpy as np
import tensorflow as tf 
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Directories
BASE_DIR = 'Flickr8k_Dataset'
WORKING_DIR = 'working'

# Load and preprocess captions
captions_path = os.path.join(BASE_DIR, 'captions.txt')
captions_doc = load_captions(captions_path)
mapping = create_mapping(captions_doc)
clean_mapping(mapping)

# Prepare tokenizer and dataset
all_captions = [caption for captions in mapping.values() for caption in captions]
tokenizer = create_tokenizer(all_captions, min_freq=1)
vocab_size = len(tokenizer.word_index) + 1
max_length = get_max_length(all_captions)

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

from feature_extraction import load_inceptionv3, extract_features

# Load InceptionV3 model
model_inception = load_inceptionv3()

directory = os.path.join(BASE_DIR, 'Images')

# Extract features
features = extract_features(model_inception, directory)

# Save features for later use
save_features(features, os.path.join(WORKING_DIR, 'features.pkl'))


# Debugging Missing Keys
missing_keys = [key for key in mapping.keys() if key not in features]
if missing_keys:
    print(f"Warning: Missing feature keys (first 5): {missing_keys[:5]}")
else:
    print("All keys are properly aligned between mapping and features.")

from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, add, Attention, Lambda, Layer, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf

def define_model(vocab_size, max_length):
    # Encoder model (Image features)
    inputs1 = Input(shape=(2048,), name="image")
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(512, activation='relu')(fe1)
    fe2 = BatchNormalization()(fe2)

    # Sequence feature layers (Text)
    inputs2 = Input(shape=(max_length,), name="text")
    se1 = Embedding(vocab_size, 300, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(512)(se2)
    se3 = BatchNormalization()(se3)

    # Decoder model
    decoder1 = add([fe2,se3])
    # decoder1 = add([fe2, context_vector])  # Combine the image feature vector and context vector
    decoder2 = Dense(512, activation='relu')(decoder1)
    decoder2 = BatchNormalization()(decoder2)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Define the model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model


def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key in data_keys:
            captions = mapping.get(key, [])
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1
                    
                    if n == batch_size:
                        yield {"image": np.array(X1), "text": np.array(X2)}, np.array(y)
                        X1, X2, y = [], [], []
                        n = 0
        if n > 0:
            yield {"image": np.array(X1), "text": np.array(X2)}, np.array(y)
            X1, X2, y = [], [], []
            n = 0

# Train Model
model = define_model(vocab_size, max_length)
epochs = 100
batch_size = 32
steps = len(train) // batch_size

# Create Directories for Checkpoints and Logs
os.makedirs("checkpoints", exist_ok=True)
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


from tensorflow.keras.callbacks import ReduceLROnPlateau

# Callback for learning rate reduction
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1, min_lr=1e-4)


# Callbacks
# ModelCheckpoint callback with .keras extension
model_checkpoint_callback = ModelCheckpoint(
    filepath="checkpoints/model-{epoch:02d}.keras",  # Save as .keras format
    save_best_only=True,  # Save only the best model
    monitor='loss',  # Monitor training loss
    mode='min',  # Save when loss is minimized
    verbose=1,  # Print saving status
)


tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,write_images=True)

generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
model.fit(
    generator, 
    epochs=epochs, 
    steps_per_epoch=steps, 
    verbose=1, 
    callbacks=[model_checkpoint_callback, tensorboard_callback, lr_scheduler]
)

def save_model(model, path):
    model.save(path)

save_model(model, os.path.join(WORKING_DIR, 'model.h5'))

# Evaluate and Generate Captions
features = load_features(os.path.join(WORKING_DIR, 'features.pkl'))
# bleu1, bleu2 = evaluate_model(model, test, mapping, features, tokenizer, max_length)
bleu_scores = evaluate_model(model, test, mapping, features, tokenizer, max_length)

print("BLEU-Scores:", bleu_scores)

# Generate Captions for Sample Images
sample_images = ["10815824_2997e03d76.jpg","23445819_3a458716c1.jpg","3703960010_1e4c922a25.jpg","36422830_55c844bc2d.jpg"]
for image_name in sample_images:
    generate_caption(image_name, model, mapping, features, tokenizer, max_length, BASE_DIR)
    print("\n")
