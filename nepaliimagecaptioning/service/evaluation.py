




from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

class Evaluation:
    def __init__(self, model, tokenizer, max_length):
        
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _idx_to_word(self, integer):
       
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def predict_caption(self, image):
        
        in_text = 'startseq'
        for i in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], self.max_length)
            yhat = self.model.predict([image, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self._idx_to_word(yhat)
            if word is None:
                break
            in_text += " " + word
            if word == 'endseq':
                break
        return in_text

    def evaluate_model(self, test, mapping, features):
        
        actual, predicted = list(), list()
        for key in tqdm(test, desc="Evaluating"):
            captions = mapping[key]
            y_pred = self.predict_caption(features[key])
            actual_captions = [caption.split() for caption in captions]
            y_pred = y_pred.split()
            actual.append(actual_captions)
            predicted.append(y_pred)
        bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
        bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
        return bleu1, bleu2

    def generate_caption(self, image_name, mapping, features, base_dir):
       
        image_id = image_name.split('.')[0]
        img_path = os.path.join(base_dir, "Images", image_name)
        image = Image.open(img_path)
        captions = mapping[image_id]
        print('---------------------Actual---------------------')
        for caption in captions:
            print(caption)
        y_pred = self.predict_caption(features[image_id])
        print('--------------------Predicted--------------------')
        print(y_pred)
        plt.imshow(image)
        plt.show()
