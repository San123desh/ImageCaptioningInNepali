
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import matplotlib.pyplot as plt
from PIL import Image
import os

class CaptionEvaluator:
    def __init__(self, model, tokenizer, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def idx_to_word(self, integer):
        return self.tokenizer.index_word.get(integer, None) if hasattr(self.tokenizer, "index_word") else None

    # def predict_caption(self, image, beam_size=3, repetition_penalty=1.2):
    #     start_token = 'startseq'
    #     end_token = 'endseq'
    #     sequences = [([self.tokenizer.word_index[start_token]], 0.0)]

    #     for _ in range(self.max_length):
    #         all_candidates = []
    #         for seq, score in sequences:
    #             sequence = pad_sequences([seq], maxlen=self.max_length, padding='post')
    #             y_pred = self.model.predict([image, sequence], verbose=0)[0]
    #             top_preds = np.argsort(y_pred)[-beam_size:]

    #             for word_id in top_preds:
    #                 candidate_seq = seq + [word_id]
    #                 word_counts = {}
    #                 for word in candidate_seq:
    #                     if word in word_counts:
    #                         word_counts[word] += 1
    #                     else:
    #                         word_counts[word] = 1
    #                 penalty = sum([count * repetition_penalty for count in word_counts.values() if count > 1])
    #                 candidate_score = score - np.log(y_pred[word_id] + penalty)
    #                 all_candidates.append((candidate_seq, candidate_score))

    #         sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_size]

    #     final_seq, _ = sequences[0]
    #     final_caption = [self.tokenizer.index_word.get(idx, '') for idx in final_seq if idx > 0]
    #     final_caption = [word for word in final_caption if word not in {start_token, end_token}]
    #     return ' '.join(final_caption)

    def predict_caption(self, image, beam_size=3, repetition_penalty=1.2):
        start_token = 'startseq'
        end_token = 'endseq'
        sequences = [([self.tokenizer.word_index[start_token]], 0.0)]

        for _ in range(self.max_length):
            all_candidates = []
            for seq, score in sequences:
                sequence = pad_sequences([seq], maxlen=self.max_length, padding='post')
                
                # Reshape the image input to have a batch dimension
                image_batch = np.expand_dims(image, axis=0)  # Shape: (1, 2048)
                
                # Debug: Print input shapes
                print(f"Image input shape: {image_batch.shape}")
                print(f"Sequence input shape: {sequence.shape}")
                
                # Predict using the model
                y_pred = self.model.predict([image_batch, sequence], verbose=0)[0]
                top_preds = np.argsort(y_pred)[-beam_size:]

                for word_id in top_preds:
                    candidate_seq = seq + [word_id]
                    word_counts = {}
                    for word in candidate_seq:
                        if word in word_counts:
                            word_counts[word] += 1
                        else:
                            word_counts[word] = 1
                    penalty = sum([count * repetition_penalty for count in word_counts.values() if count > 1])
                    candidate_score = score - np.log(y_pred[word_id] + penalty)
                    all_candidates.append((candidate_seq, candidate_score))

            sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_size]

        final_seq, _ = sequences[0]
        final_caption = [self.tokenizer.index_word.get(idx, '') for idx in final_seq if idx > 0]
        final_caption = [word for word in final_caption if word not in {start_token, end_token}]
        return ' '.join(final_caption)       
    def evaluate_model(self, test, mapping, features):
        actual, predicted = [], []
        smoothing_fn = SmoothingFunction().method1

        for key in tqdm(test, desc="Evaluating"):
            if key not in mapping or key not in features:
                continue

            captions = mapping[key]
            if not captions:
                continue

            y_pred = self.predict_caption(features[key])
            y_pred = y_pred.split()

            actual.append([caption.split() for caption in captions])
            predicted.append(y_pred)

        bleu_scores = {
            "BLEU-1": corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smoothing_fn),
            "BLEU-2": corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_fn),
            "BLEU-3": corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_fn),
            "BLEU-4": corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_fn)
        }

        print("\nBLEU Scores:", bleu_scores)
        return bleu_scores

    def generate_caption(self, image_name, mapping, features, base_dir):
        try:
            image_id = image_name.split('.')[0]
            img_path = os.path.join(base_dir, "Images", image_name)
            image = Image.open(img_path)
            captions = mapping.get(image_id, [])

            if not captions:
                print(f"No ground-truth captions found for image ID: {image_id}")
                return

            print('---------------------Actual---------------------')
            for caption in captions:
                print(caption)

            y_pred = self.predict_caption(features[image_id])
            print('--------------------Predicted--------------------')
            print(y_pred)

            plt.imshow(image)
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"Error generating caption for {image_name}: {e}")


