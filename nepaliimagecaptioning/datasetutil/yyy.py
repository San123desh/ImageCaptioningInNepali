import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def create_mapping(captions_doc):
    mapping = {}
    for line in tqdm(captions_doc.split('\n')):
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        caption = " ".join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping

def clean_mapping(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = caption.replace('\s+', ' ')
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption

def create_tokenizer(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer

def get_max_length(captions):
    return max(len(caption.split()) for caption in captions)
caption_processing.py




import os
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_captions(captions_path):
    with open(captions_path, 'r') as f:
        next(f)
        return f.read()

def save_features(features, path):
    with open(path, 'wb') as f:
        pickle.dump(features, f)

def load_features(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
data_preparation.py




from tensorflow.keras.applications.vgg16 import VGG16 , preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import os

def load_vgg16():
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model

def extract_features(model, directory):
    features = {}
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist. Please check the path.")
    else:
        for img_name in tqdm(os.listdir(directory)):
            img_path = os.path.join(directory, img_name)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            image_id = img_name.split('.')[0]
            features[image_id] = feature
    return features
feature_extraction.py




from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, add
from tensorflow.keras.models import Model

def define_model(vocab_size, max_length):
    # Encoder model
    inputs1 = Input(shape=(4096,), name="image")
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # Sequence feature layers
    inputs2 = Input(shape=(max_length,), name="text")
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)
    # Decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
model_definition.py





import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
from PIL import Image
import os

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

def evaluate_model(model, test, mapping, features, tokenizer, max_length):
    actual, predicted = list(), list()
    for key in tqdm(test):
        captions = mapping[key]
        y_pred = predict_caption(model, features[key], tokenizer, max_length)
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        actual.append(actual_captions)
        predicted.append(y_pred)
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    return bleu1, bleu2

def generate_caption(image_name, model, mapping, features, tokenizer, max_length, base_dir):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(base_dir, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)
    plt.show()
evaluation.py




import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

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
trainer.py




import os

# Set environment variable to avoid OpenMP runtime issues 
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from caption_processing import create_mapping, clean_mapping, create_tokenizer, get_max_length
from data_preparation import load_captions, save_features, load_features
from feature_extraction import load_vgg16, extract_features
from model_definition import define_model
from trainer import train_model, save_model
from evaluation import evaluate_model, generate_caption

BASE_DIR = 'Flickr8k_Dataset'
WORKING_DIR = 'working'

# Load and preprocess captions
captions_path = os.path.join(BASE_DIR, 'captions.txt')
captions_doc = load_captions(captions_path)
mapping = create_mapping(captions_doc)
clean_mapping(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

# Create tokenizer and get max caption length
tokenizer = create_tokenizer(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = get_max_length(all_captions)

# Split data into training and testing sets
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]

# Load VGG16 model and extract features
model_vgg = load_vgg16()
directory = os.path.join(BASE_DIR, 'Images')
features = extract_features(model_vgg, directory)
save_features(features, os.path.join(WORKING_DIR, 'features.pkl'))

# Define and compile the image captioning model
model = define_model(vocab_size, max_length)

# Train the model
epochs = 20
batch_size = 32
steps = len(train) // batch_size
train_model(model, train, mapping, features, tokenizer, max_length, vocab_size, batch_size, epochs, steps)

# Save the trained model
save_model(model, os.path.join(WORKING_DIR, 'model.h5'))

# Evaluate the model
features = load_features(os.path.join(WORKING_DIR, 'features.pkl'))
bleu1, bleu2 = evaluate_model(model, test, mapping, features, tokenizer, max_length)
print(f"BLEU-1: {bleu1:.6f}")
print(f"BLEU-2: {bleu2:.6f}")

# Generate captions for sample images
generate_caption("1001773457_577c3a7d70.jpg", model, mapping, features, tokenizer, max_length, BASE_DIR)
generate_caption("1002674143_1b742ab4b8.jpg", model, mapping, features, tokenizer, max_length, BASE_DIR)
main.py




















