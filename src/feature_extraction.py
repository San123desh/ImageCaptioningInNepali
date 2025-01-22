

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import os


def load_inceptionv3():
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    image_model = Model(inputs=base_model.input, outputs=x)
    return image_model



def extract_features(model, directory):
    
    features = {}
    if not os.path.exists(directory):
        print(f"Error: The directory {directory} does not exist. Please check the path.")
        return features  # Return an empty dictionary

    for img_name in tqdm(os.listdir(directory), desc="Extracting features"):
        img_path = os.path.join(directory, img_name)

        # Ensure the file is an image
        if not img_name.lower().endswith(('png', 'jpg', 'jpeg')):
            print(f"Skipping non-image file: {img_name}")
            continue

        try:
            # Load and preprocess the image
            image = load_img(img_path, target_size=(299, 299))  # InceptionV3 expects 299x299 images
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)  # Preprocess for InceptionV3

            # Predict features
            feature = model.predict(image, verbose=0)
            image_id = os.path.splitext(img_name)[0]  # Get the image ID without extension
            features[image_id] = feature
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    return features
