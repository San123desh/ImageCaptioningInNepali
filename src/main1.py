import datetime
import os

from keras.src.callbacks import ModelCheckpoint, TensorBoard
from keras.src.utils import pad_sequences, to_categorical

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from caption_processing import create_mapping, clean_mapping, create_tokenizer, get_max_length
from data_preparation import load_captions, save_features, load_features
from feature_extraction import load_vgg16, extract_features
from evaluation import evaluate_model, generate_caption
import numpy as np

BASE_DIR = 'Flickr8k_Dataset'
WORKING_DIR = 'working'

# Load and preprocess Nepali captions
captions_path = os.path.join(BASE_DIR, 'captions.txt')
captions_doc = load_captions(captions_path)
mapping = create_mapping(captions_doc)
clean_mapping(mapping)

# Prepare tokenizer and dataset
all_captions = [caption for captions in mapping.values() for caption in captions]
tokenizer = create_tokenizer(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = get_max_length(all_captions)

# debugging
# tokenizer
print("Tokenizer Vocabulary Size:", len(tokenizer.word_index))
print("Sample Tokenizer Mapping (First 10):", {k: tokenizer.word_index[k] for k in list(tokenizer.word_index)[:10]})

image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train, test = image_ids[:split], image_ids[split:]

# Extract image features
model_vgg = load_vgg16()
directory = os.path.join(BASE_DIR, 'Images')
features = extract_features(model_vgg, directory)
save_features(features, os.path.join(WORKING_DIR, 'features.pkl'))

# Debugging
# Check if all keys in mapping are in features
missing_keys = [key for key in mapping.keys() if key not in features]
if missing_keys:
    print(f"Missing feature keys: {missing_keys[:10]}")  # Limit to 10 for readability
else:
    print("All keys are properly aligned between mapping and features.")

import keras


def define_model(vocab_size, max_length):
    # Encoder model
    inputs1 = keras.Input(shape=(4096,), name="image")
    fe1 = keras.layers.Dropout(0.4)(inputs1)
    fe2 = keras.layers.Dense(256, activation='relu')(fe1)
    # Sequence feature layers
    inputs2 = keras.Input(shape=(max_length,), name="text")
    se1 = keras.layers.Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = keras.layers.Dropout(0.4)(se1)
    se3 = keras.layers.LSTM(256)(se2)
    # Decoder model
    decoder1 = keras.layers.Add([fe2, se3])
    decoder2 = keras.layers.Dense(256, activation='relu')(decoder1)
    outputs = keras.layers.Dense(vocab_size, activation='softmax')(decoder2)
    model = keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield {"image": X1, "text": X2}, y
                X1, X2, y = list(), list(), list()
                n = 0


def train_model(model, train, mapping, features, tokenizer, max_length, vocab_size, batch_size, epochs, steps):
    for i in range(epochs):
        generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)


def save_model(model, path):
    model.save(path)


model = define_model(vocab_size, max_length)

epochs = 50
batch_size = 32
steps = len(train) // batch_size

checkpoint_path = "checkpoints/model-{epoch:02d}-{val_loss:.2f}.h5"
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ModelCheckpoint callback
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,  # Path to save the model
    save_best_only=True,  # Save only the best model
    monitor='val_loss',  # Monitor validation loss
    mode='min',  # Save when `val_loss` is minimized
    verbose=1  # Print saving status
)

# TensorBoard callback
tensorboard_callback = TensorBoard(
    log_dir=log_dir,  # Directory for storing logs
    histogram_freq=1,  # Enable histogram visualizations
    write_graph=True,  # Save the computational graph
    write_images=True  # Save model weights as images
)

generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1,
          callbacks=[model_checkpoint_callback, tensorboard_callback])

# Evaluate and generate captions
features = load_features(os.path.join(WORKING_DIR, 'features.pkl'))
bleu1, bleu2 = evaluate_model(model, test, mapping, features, tokenizer, max_length)
print(f"BLEU-1: {bleu1:.6f}, BLEU-2: {bleu2:.6f}")

print(list(features.keys())[:5])
print(list(mapping.keys())[:5])
print(tokenizer.word_index)

# # Generate captions for sample images
generate_caption("1001773457_577c3a7d70.jpg", model, mapping, features, tokenizer, max_length, BASE_DIR)
generate_caption("1002674143_1b742ab4b8.jpg", model, mapping, features, tokenizer, max_length, BASE_DIR)
