import os

import streamlit as st

from imagecaptioning_onwork.service.caption_generator import CaptionGenerator

MODEL_PATH = "./models"
caption_generator = CaptionGenerator(model_dir=MODEL_PATH)


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
        caption = caption_generator.generate_caption("temp_image.jpg")
        st.write("**Generated Caption:**")
        st.success(caption)

        # Clean up the temporary file
        os.remove("temp_image.jpg")


if __name__ == "__main__":
    main()
