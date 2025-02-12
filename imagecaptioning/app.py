
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Set up paths
MODEL_PATH = "testingmodel/model.keras"
TOKENIZER_PATH = "testingmodel/tokenizer.pkl"
FEATURE_EXTRACTOR_PATH = "testingmodel/feature_extractor.keras"
IMAGE_SIZE = 224
MAX_LENGTH = 31  

# Load the tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load the feature extractor
feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)

# Load and compile the captioning model
caption_model = load_model(MODEL_PATH)
caption_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Function to generate caption
def generate_caption(image_path):
    """Generate a caption for the given image."""
    # Load and preprocess the image
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Extract image features
    image_features = feature_extractor.predict(img, verbose=0)

    # Generate caption
    in_text = "startseq"
    for i in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    # Clean up the caption
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption

# Streamlit app
def main():
    st.title("Image Captioning App")
    st.write("Upload an image, and the app will generate a caption for it.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Save the uploaded file temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Generate caption
        with st.spinner("Generating caption..."):
            caption = generate_caption("temp_image.jpg")
        st.write("**Generated Caption:**")
        st.success(caption)

        # Clean up the temporary file
        os.remove("temp_image.jpg")

# Run the app
if __name__ == "__main__":
    main()

