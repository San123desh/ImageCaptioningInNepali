from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

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
def generate_caption(image):
    # Extract image features
    image_features = feature_extractor.predict(image, verbose=0)
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

@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/generate_caption', methods=['POST'])
def caption_endpoint():
    # Check if an image file is present
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Load and preprocess the image
    img = load_img(file, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Generate caption
    caption = generate_caption(img)
    return jsonify({"caption": caption})

if __name__ == '__main__':
    app.run(debug=True)
