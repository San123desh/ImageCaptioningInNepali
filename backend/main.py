# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
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
feature_extraction_model = load_model(FEATURE_EXTRACTOR_PATH)

# Load and compile the captioning model
caption_model = load_model(MODEL_PATH)
caption_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Function to generate caption
def generate_caption(image_features):
    """Generate a caption for the given image features."""
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

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the React build directory
# app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../nepali-caption-generator/build"), html=True), name="static")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = tf.image.decode_image(contents, channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = np.expand_dims(image, axis=0) / 255.0

    # Extract image features
    image_features = feature_extraction_model.predict(image, verbose=0)

    # Generate caption
    caption = generate_caption(image_features)
    return JSONResponse(content={"caption": caption})
