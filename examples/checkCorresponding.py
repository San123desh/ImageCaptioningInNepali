import os

# Load captions from the .txt file
def load_captions(filename):
    """Loads captions (text) data and maps them to corresponding images.
    Args:
        filename: Path to the text file containing caption data
    """
    mapping = {}
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            line = line.strip()
            parts = line.split('#')
            if len(parts) < 2:
                continue
            image_id = parts[0]
            image_id = os.path.splitext(image_id)[0]  # Remove file extension
            if image_id not in mapping:
                mapping[image_id] = 1
            else:
                mapping[image_id] += 1
    return mapping

# Load image filenames from the directory
def load_image_filenames(directory):
    """Loads image filenames from the specified directory.
    Args:
        directory: Path to the directory containing images
    """
    return [os.path.splitext(filename)[0] for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename)) and filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

# Example usage
captions_filename = 'Flickr8k_Dataset/captions.txt'  # Path to your captions file
images_directory = 'Flickr8k_Dataset/Images'   # Path to your images directory

# Load captions and image filenames
captions = load_captions(captions_filename)
image_filenames = load_image_filenames(images_directory)

print("Images with captions:")
for image_filename in image_filenames:
    if image_filename in captions:
        print(image_filename)

print("\nImages without captions:")
for image_filename in image_filenames:
    if image_filename not in captions:
        print(image_filename)

print("\nCaptions without images:")
for image_id in captions:
    if image_id not in image_filenames:
        print(image_id)

images_with_captions = [image_filename for image_filename in image_filenames if image_filename in captions]
images_without_captions = [image_filename for image_filename in image_filenames if image_filename not in captions]

print(f"Total images: {len(image_filenames)}")
print(f"Images with captions: {len(images_with_captions)}")
print(f"Images without captions: {len(images_without_captions)}")


for image_id in captions:
    print(f"Image {image_id} has {captions[image_id]} captions")
caption_counts = {}
for image_id in captions:
    caption_count = captions[image_id]
    if caption_count not in caption_counts:
        caption_counts[caption_count] = 1
    else:
        caption_counts[caption_count] += 1

# Print the results
print("\nNumber of images with a specific number of captions:")
for caption_count in caption_counts:
    print(f"Images with {caption_count} captions: {caption_counts[caption_count]}")



    