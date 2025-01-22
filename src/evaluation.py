
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import matplotlib.pyplot as plt
from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def idx_to_word(integer, tokenizer):
    return tokenizer.index_word.get(integer, None) if hasattr(tokenizer, "index_word") else None

def predict_caption(model, image, tokenizer, max_length, beam_size=3,repetition_penalty=1.2):
    start_token = 'startseq'
    end_token = 'endseq'

    # Initialize the beam with the start token and a score of 0.0
    sequences = [([tokenizer.word_index[start_token]], 0.0)]
    
    for _ in range(max_length):
        all_candidates = []
        
        for seq, score in sequences:
            # Pad the current sequence
            sequence = pad_sequences([seq], maxlen=max_length, padding='post')
            # Predict the next word probabilities
            y_pred = model.predict([image, sequence], verbose=0)[0]
            # Get the top `beam_size` predictions
            top_preds = np.argsort(y_pred)[-beam_size:]

            for word_id in top_preds:
                # Create a new candidate sequence with updated score
                candidate_seq = seq + [word_id]

                # Apply repetition penalty
                word_counts = {}
                for word in candidate_seq:
                    if word in word_counts:
                        word_counts[word] += 1
                    else:
                        word_counts[word] = 1
                penalty = sum([count * repetition_penalty for count in word_counts.values() if count > 1])
                
                candidate_score = score - np.log(y_pred[word_id]  +penalty)
                all_candidates.append((candidate_seq, candidate_score))

        # Sort all candidates by score and keep the top `beam_size`
        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_size]

    # Select the sequence with the best score
    final_seq, _ = sequences[0]
    # Convert token IDs to words
    final_caption = [tokenizer.index_word.get(idx, '') for idx in final_seq if idx > 0]
    # Remove start and end tokens
    final_caption = [word for word in final_caption if word not in {start_token, end_token}]
    return ' '.join(final_caption)

def evaluate_model(model, test, mapping, features, tokenizer, max_length):
    actual, predicted = [], []
    smoothing_fn = SmoothingFunction().method1

    for key in tqdm(test, desc="Evaluating"):
        if key not in mapping or key not in features:
            logging.warning(f"Key '{key}' not found in mapping or features. Skipping...")
            continue

        captions = mapping[key]
        if not captions:
            logging.warning(f"No captions found for image ID '{key}'. Skipping...")
            continue

        y_pred = predict_caption(model, features[key], tokenizer, max_length)
        y_pred = y_pred.split()

        actual.append([caption.split() for caption in captions])
        predicted.append(y_pred)

    # Compute BLEU scores
    bleu_scores = {
        "BLEU-1": corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smoothing_fn),
        "BLEU-2": corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_fn),
        "BLEU-3": corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_fn),
        "BLEU-4": corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_fn)
    }

    # Debugging outputs
    logging.info("\nDebugging Outputs:")
    for idx in range(min(5, len(actual))):
        logging.info(f"Actual Caption {idx + 1}: {actual[idx]}")
        logging.info(f"Predicted Caption {idx + 1}: {predicted[idx]}")

    logging.info("\nBLEU Scores: %s", bleu_scores)
    return bleu_scores

def generate_caption(image_name, model, mapping, features, tokenizer, max_length, base_dir):
    try:
        image_id = image_name.split('.')[0]
        img_path = os.path.join(base_dir, "Images", image_name)
        image = Image.open(img_path)
        captions = mapping.get(image_id, [])

        if not captions:
            logging.warning(f"No ground-truth captions found for image ID: {image_id}")
            return

        print('---------------------Actual---------------------')
        for caption in captions:
            print(caption)

        y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
        print('--------------------Predicted--------------------')
        print(y_pred)

        plt.imshow(image)
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        logging.error(f"Error generating caption for {image_name}: {e}")

