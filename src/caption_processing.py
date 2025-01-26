

import re
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer

# Function to create the mapping from image IDs to captions
def create_mapping(captions_doc):
    mapping = {}
    for line in tqdm(captions_doc.split("\n"), desc="Processing captions"):
        tokens = line.split(",")
        if len(tokens) < 2:  # Skip invalid or incomplete lines
            continue
        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split(".")[0]  # Remove file extension
        caption = " ".join(caption).strip()  # Combine caption tokens into a single string
        if not caption:  # Skip empty captions
            continue
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    print(f"Total images with captions: {len(mapping)}")
    return mapping

# Function to clean the captions in the mapping
def clean_mapping(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].strip().lower()  # Convert to lowercase
            caption = re.sub(r"[^\u0900-\u097F\s]", "", caption)  # Keep only Devanagari characters and spaces
            caption = re.sub(r"\s+", " ", caption).strip()  # Normalize spaces
            if caption:  # Add 'startseq' and 'endseq' if valid
                captions[i] = f"startseq {caption} endseq"
            else:
                captions[i] = None  # Mark invalid captions as None
        # Filter out None captions
        mapping[key] = [caption for caption in captions if caption is not None]
    print(f"Cleaned mapping size: {len(mapping)}")

# Function to create the tokenizer
def create_tokenizer(captions, min_freq=1):
    tokenizer = Tokenizer(oov_token="<unk>")  # Ensure <unk> token is used for OOV words
    tokenizer.fit_on_texts(captions)  # Learn the word index from captions
    
    # Get the word counts from the tokenizer
    word_counts = tokenizer.word_counts

    # Filter out words that do not meet the minimum frequency
    tokenizer.word_index = {k: v for k, v in tokenizer.word_index.items() 
                            if word_counts.get(k, 0) >= min_freq or k == "<unk>"}

    return tokenizer


# Function to get the maximum caption length
def get_max_length(captions):
    max_len = max(len(caption.split()) for caption in captions)
    print(f"Maximum caption length: {max_len}")
    return max_len

# # Example usage:
# if __name__ == "__main__":
#     # Load your captions document (adjust the path as needed)
#     with open("path/to/captions.txt", "r", encoding="utf-8") as f:
#         captions_doc = f.read()

#     # Step 1: Create the mapping
#     mapping = create_mapping(captions_doc)

#     # Step 2: Clean the mapping
#     clean_mapping(mapping)

#     # Step 3: Prepare captions for tokenizer
#     all_captions = [caption for captions in mapping.values() for caption in captions]

#     # Step 4: Create the tokenizer
#     tokenizer = create_tokenizer(all_captions, min_freq=5)  # Adjust min_freq as needed

#     # Step 5: Get maximum caption length
#     max_length = get_max_length(all_captions)

  