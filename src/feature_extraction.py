


from tensorflow.keras.applications.vgg16 import VGG16 , preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import os

def load_vgg16():
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model

def extract_features(model, directory):
    features = {}
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist. Please check the path.")
    else:
        for img_name in tqdm(os.listdir(directory)):
            img_path = os.path.join(directory, img_name)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            image_id = img_name.split('.')[0]
            features[image_id] = feature
    return features
