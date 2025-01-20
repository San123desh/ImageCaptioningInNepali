import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
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
    actual, predicted = [], []
    smoothing_fn = SmoothingFunction().method1

    for key in tqdm(test):
        captions = mapping[key]  # Ground truth captions for the image
        y_pred = predict_caption(model, features[key], tokenizer, max_length)  # Predicted caption
        
        # Prepare actual captions as a list of tokenized words
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()  # Tokenize predicted caption
        
        actual.append(actual_captions)
        predicted.append(y_pred)

    # Debugging: Print the first 5 actual and predicted captions
    print("\nDebugging Outputs:")
    for idx in range(min(5, len(actual))):  # Limit to 5
        print(f"Actual Caption {idx + 1}: {actual[idx]}")
        print(f"Predicted Caption {idx + 1}: {predicted[idx]}")

    # Compute BLEU scores
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smoothing_fn)
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_fn)
    
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
