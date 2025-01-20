

# import os

# # Set environment variable to avoid OpenMP runtime issues 
# # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# from caption_processing import create_mapping, clean_mapping, create_tokenizer, get_max_length
# from data_preparation import load_captions, save_features, load_features
# from feature_extraction import load_vgg16, extract_features
# from model_definition import define_model
# from trainer import train_model, save_model
# from evaluation import evaluate_model, generate_caption

# BASE_DIR = 'Flickr8k_Dataset'
# WORKING_DIR = 'working'

# # Load and preprocess captions
# captions_path = os.path.join(BASE_DIR, 'captions.txt')
# captions_doc = load_captions(captions_path)
# mapping = create_mapping(captions_doc)
# clean_mapping(mapping)

# all_captions = []
# for key in mapping:
#     for caption in mapping[key]:
#         all_captions.append(caption)

# # Create tokenizer and get max caption length
# tokenizer = create_tokenizer(all_captions)
# vocab_size = len(tokenizer.word_index) + 1
# max_length = get_max_length(all_captions)

# # Split data into training and testing sets
# image_ids = list(mapping.keys())
# split = int(len(image_ids) * 0.90)
# train = image_ids[:split]
# test = image_ids[split:]

# # Load VGG16 model and extract features
# model_vgg = load_vgg16()
# directory = os.path.join(BASE_DIR, 'Images')
# features = extract_features(model_vgg, directory)
# save_features(features, os.path.join(WORKING_DIR, 'features.pkl'))

# # Define and compile the image captioning model
# model = define_model(vocab_size, max_length)

# # Train the model
# epochs = 20
# batch_size = 32
# steps = len(train) // batch_size
# train_model(model, train, mapping, features, tokenizer, max_length, vocab_size, batch_size, epochs, steps)

# # Save the trained model
# save_model(model, os.path.join(WORKING_DIR, 'model.h5'))

# # Evaluate the model
# features = load_features(os.path.join(WORKING_DIR, 'features.pkl'))
# bleu1, bleu2 = evaluate_model(model, test, mapping, features, tokenizer, max_length)
# print(f"BLEU-1: {bleu1:.6f}")
# print(f"BLEU-2: {bleu2:.6f}")

# # Generate captions for sample images
# generate_caption("1001773457_577c3a7d70.jpg", model, mapping, features, tokenizer, max_length, BASE_DIR)
# generate_caption("1002674143_1b742ab4b8.jpg", model, mapping, features, tokenizer, max_length, BASE_DIR)


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from caption_processing import create_mapping, clean_mapping, create_tokenizer, get_max_length
from data_preparation import load_captions, save_features, load_features
from feature_extraction import load_vgg16, extract_features
from model_definition import define_model
from trainer import train_model, save_model
from evaluation import evaluate_model, generate_caption
from tensorflow.keras.optimizers import Adam



BASE_DIR = 'Flickr8k_Dataset'
WORKING_DIR = 'working'

# Load and preprocess Nepali captions
captions_path = os.path.join(BASE_DIR, 'captions.txt')
captions_doc = load_captions(captions_path)
mapping = create_mapping(captions_doc)
clean_mapping(mapping)

# Prepare tokenizer and dataset
all_captions = [caption for captions in mapping.values() for caption in captions]
tokenizer = create_tokenizer(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = get_max_length(all_captions)


#debugging
# tokenizer
print("Tokenizer Vocabulary Size:", len(tokenizer.word_index))
print("Sample Tokenizer Mapping (First 10):", {k: tokenizer.word_index[k] for k in list(tokenizer.word_index)[:10]})



image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train, test = image_ids[:split], image_ids[split:]

# Extract image features
model_vgg = load_vgg16()
directory = os.path.join(BASE_DIR, 'Images')
features = extract_features(model_vgg, directory)
save_features(features, os.path.join(WORKING_DIR, 'features.pkl'))

#Debugging
# Check if all keys in mapping are in features
missing_keys = [key for key in mapping.keys() if key not in features]
if missing_keys:
    print(f"Missing feature keys: {missing_keys[:10]}")  # Limit to 10 for readability
else:
    print("All keys are properly aligned between mapping and features.")


# Define and train model
model = define_model(vocab_size, max_length)
# Use a smaller learning rate for Adam optimizer
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy')

epochs = 50
batch_size = 32
steps = len(train) // batch_size
train_model(model, train, mapping, features, tokenizer, max_length, vocab_size, batch_size, epochs, steps)
save_model(model, os.path.join(WORKING_DIR, 'model.h5'))

# Evaluate and generate captions
features = load_features(os.path.join(WORKING_DIR, 'features.pkl'))
bleu1, bleu2 = evaluate_model(model, test, mapping, features, tokenizer, max_length)
print(f"BLEU-1: {bleu1:.6f}, BLEU-2: {bleu2:.6f}")

print(list(features.keys())[:5])
print(list(mapping.keys())[:5])
print(tokenizer.word_index)

# # Generate captions for sample images
generate_caption("1001773457_577c3a7d70.jpg", model, mapping, features, tokenizer, max_length, BASE_DIR)
generate_caption("1002674143_1b742ab4b8.jpg", model, mapping, features, tokenizer, max_length, BASE_DIR)

