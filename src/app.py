import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging
from caption_processing import create_mapping, clean_mapping, create_tokenizer, get_max_length
from data_preparation import load_captions, load_features
from evaluation import predict_caption
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Starting the Streamlit app")

# Load the saved model
def load_caption_model(filepath):
    try:
        model = load_model(
            filepath,
            custom_objects={"Functional": tf.keras.models.Model},  # Add custom objects if needed
        )
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


# Define paths
BASE_DIR = "Flickr8k_Dataset"
WORKING_DIR = "working"
MODEL_PATH = "checkpoints/model-100.keras"

# Load the model
caption_model = load_caption_model(MODEL_PATH)

# Load and preprocess captions
captions_path = os.path.join(BASE_DIR, "captions.txt")
captions_doc = load_captions(captions_path)
mapping = create_mapping(captions_doc)
clean_mapping(mapping)

# Prepare tokenizer and dataset
all_captions = [caption for captions in mapping.values() for caption in captions]
tokenizer = create_tokenizer(all_captions, min_freq=1)
vocab_size = len(tokenizer.word_index) + 1
max_length = get_max_length(all_captions)

# Load pre-extracted features
features = load_features(os.path.join(WORKING_DIR, "features.pkl"))


def predict_caption_with_beam_search(
    model, image, tokenizer, max_length, beam_size=3, repetition_penalty=1.2
):
    start_token = "startseq"
    end_token = "endseq"

    # Initialize the beam with the start token and a score of 0.0
    sequences = [([tokenizer.word_index[start_token]], 0.0)]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            # Pad the current sequence
            sequence = pad_sequences([seq], maxlen=max_length, padding="post")
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
                    word_counts[word] = word_counts.get(word, 0) + 1
                penalty = sum(
                    [count * repetition_penalty for count in word_counts.values() if count > 1]
                )

                candidate_score = score - np.log(y_pred[word_id] + penalty)
                all_candidates.append((candidate_seq, candidate_score))

        # Sort all candidates by score and keep the top `beam_size`
        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_size]

    # Select the sequence with the best score
    final_seq, _ = sequences[0]
    # Convert token IDs to words
    final_caption = [tokenizer.index_word.get(idx, "") for idx in final_seq if idx > 0]
    # Remove start and end tokens
    final_caption = [word for word in final_caption if word not in {start_token, end_token}]
    return " ".join(final_caption)


def generate_caption(image, model, mapping, features, tokenizer, max_length):
    try:
        image_id = os.path.splitext(image.name)[0]
        img_path = os.path.join("Images", image.name)
        image.save(img_path)

        captions = mapping.get(image_id, [])

        # Display ground-truth captions if available
        if not captions:
            logging.warning(f"No ground-truth captions found for image ID: {image_id}")
            return "No ground-truth captions found."

        st.write("---------------------Actual---------------------")
        for caption in captions:
            st.write(" ".join(caption))

        # Predict caption using the model
        feature_vector = features.get(image_id)
        if feature_vector is None:
            logging.warning(f"No features found for image ID: {image_id}")
            return "No features available for this image."

        y_pred = predict_caption_with_beam_search(
            model, feature_vector, tokenizer, max_length
        )
        st.write("--------------------Predicted--------------------")
        st.write(y_pred)

        return y_pred

    except Exception as e:
        logging.error(f"Error generating caption for {image.name}: {e}")
        return "Error generating caption."


def main():
    st.title("Image Caption Generator")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Generate caption for the uploaded image using the loaded model
        if caption_model:
            caption = generate_caption(uploaded_file, caption_model, mapping, features, tokenizer, max_length)
            st.write("Caption: ", caption)
        else:
            st.write("Failed to load model. Please check the logs for details.")


if __name__ == "__main__":
    main()
