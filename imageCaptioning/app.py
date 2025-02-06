
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from service.caption_generator import CaptionGenerator

# Set up paths
MODEL_PATH = "testingmodel/model.keras"
TOKENIZER_PATH = "testingmodel/tokenizer.pkl"
FEATURE_EXTRACTOR_PATH = "testingmodel/feature_extractor.keras"
IMAGE_SIZE = 224
MAX_LENGTH = 31  # Ensure this matches the value used during training

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
        caption = generate_caption("temp_image.jpg")
        st.write("**Generated Caption:**")
        st.success(caption)

        # Clean up the temporary file
        os.remove("temp_image.jpg")

# Run the app
if __name__ == "__main__":
    main()


# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle
# import os

# # Set up paths
# MODEL_PATH = "testingmodel/model.keras"
# TOKENIZER_PATH = "testingmodel/tokenizer.pkl"
# FEATURE_EXTRACTOR_PATH = "testingmodel/feature_extractor.keras"
# IMAGE_SIZE = 224
# MAX_LENGTH = 31  # Ensure this matches the value used during training

# # Load the tokenizer
# with open(TOKENIZER_PATH, "rb") as f:
#     tokenizer = pickle.load(f)

# # Load the feature extractor
# feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)

# # Load and compile the captioning model
# caption_model = load_model(MODEL_PATH)
# caption_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Function to generate caption using beam search
# def beam_search_predictions(image_features, beam_index=3):
#     start = [tokenizer.word_index['startseq']]
#     start_word = [[start, 0.0]]
    
#     while len(start_word[0][0]) < MAX_LENGTH:
#         temp = []
#         for s in start_word:
#             par_seq = pad_sequences([s[0]], maxlen=MAX_LENGTH, padding='post')
#             preds = caption_model.predict([image_features, par_seq], verbose=0)
#             word_preds = np.argsort(preds[0])[-beam_index:]
            
#             for w in word_preds:
#                 next_cap, prob = s[0][:], s[1]
#                 next_cap.append(w)
#                 prob += preds[0][w]
#                 temp.append([next_cap, prob])
                
#         start_word = temp
#         start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
#         start_word = start_word[-beam_index:]
        
#     start_word = start_word[-1][0]
#     final_caption = [tokenizer.index_word[i] for i in start_word]
#     final_caption = final_caption[1:]
#     final_caption = ' '.join(final_caption)
#     return final_caption

# # Function to generate caption
# def generate_caption(image_path):
#     """Generate a caption for the given image."""
#     # Load and preprocess the image
#     img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     img = img_to_array(img) / 255.0
#     img = np.expand_dims(img, axis=0)

#     # Extract image features
#     image_features = feature_extractor.predict(img, verbose=0)

#     # Use beam search to generate caption
#     caption = beam_search_predictions(image_features, beam_index=3)

#     # Clean up the caption
#     caption = caption.replace("startseq", "").replace("endseq", "").strip()
#     return caption

# # Streamlit app
# def main():
#     st.title("Image Captioning App")
#     st.write("Upload an image, and the app will generate a caption for it.")

#     # File uploader
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Display the uploaded image
#         st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

#         # Save the uploaded file temporarily
#         with open("temp_image.jpg", "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # Generate caption
#         caption = generate_caption("temp_image.jpg")
#         st.write("**Generated Caption:**")
#         st.success(caption)

#         # Clean up the temporary file
#         os.remove("temp_image.jpg")

# # Run the app
# if __name__ == "__main__":
#     main()




# streamlit for frontend


# import streamlit as st
# import requests
# from service.caption_generator import CaptionGenerator
# import os

# # Flask backend URL
# FLASK_URL = "http://127.0.0.1:5000"

# # Initialize the caption generator
# caption_generator = CaptionGenerator("testingmodel/model.keras", "testingmodel/tokenizer.pkl", "testingmodel/feature_extractor.keras")

# # Streamlit app
# def main():
#     st.title("Nepali Image Captioning App")
#     st.write("Upload an image, and the app will generate a Nepali caption for it.")

#     # Initialize session state for user authentication
#     if 'user_id' not in st.session_state:
#         st.session_state.user_id = None

#     # Sidebar for login/register
#     st.sidebar.header("Login / Register")
#     choice = st.sidebar.selectbox("Choose Action", ["Login", "Register"])

#     if choice == "Login":
#         username = st.sidebar.text_input("Username")
#         password = st.sidebar.text_input("Password", type="password")
#         if st.sidebar.button("Login"):
#             response = requests.post(f"{FLASK_URL}/login", json={'username': username, 'password': password})
#             if response.status_code == 200:
#                 st.session_state.user_id = response.json().get('user_id')
#                 st.sidebar.success("Login successful!")
#             else:
#                 st.sidebar.error("Invalid username or password")

#     elif choice == "Register":
#         username = st.sidebar.text_input("Username")
#         password = st.sidebar.text_input("Password", type="password")
#         if st.sidebar.button("Register"):
#             response = requests.post(f"{FLASK_URL}/register", json={'username': username, 'password': password})
#             if response.status_code == 201:
#                 st.sidebar.success("Registration successful! Please login.")
#             else:
#                 st.sidebar.error("Registration failed")

#     # Display image upload and caption generation only if the user is logged in
#     if st.session_state.user_id is not None:
#         st.sidebar.header(f"Logged in as User {st.session_state.user_id}")
#         if st.sidebar.button("Logout"):
#             st.session_state.user_id = None
#             st.sidebar.success("Logged out successfully!")

#         # File uploader
#         uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#         if uploaded_file is not None:
#             # Display the uploaded image
#             st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

#             # Save the uploaded file temporarily
#             with open("temp_image.jpg", "wb") as f:
#                 f.write(uploaded_file.getbuffer())

#             # Generate caption
#             caption = caption_generator.generate_caption("temp_image.jpg")
#             st.write("**Generated Caption:**")
#             st.success(caption)

#             # Save caption and image
#             if st.button("Save Caption"):
#                 response = requests.post(f"{FLASK_URL}/save", json={
#                     'user_id': st.session_state.user_id,
#                     'image_path': "temp_image.jpg",
#                     'caption': caption
#                 })
#                 if response.status_code == 201:
#                     st.success("Caption saved successfully!")
#                 else:
#                     st.error("Failed to save caption")

#             # Clean up the temporary file
#             os.remove("temp_image.jpg")

# # Run the app
# if __name__ == "__main__":
#     main()
