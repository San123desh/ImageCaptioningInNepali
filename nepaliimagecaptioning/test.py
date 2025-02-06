# import os

# # Set the Indic NLP resources path
# os.environ['INDIC_RESOURCES_PATH'] = r'C:\Users\Acer\ImageCaptioning\indic_nlp_resources'
# print("INDIC_RESOURCES_PATH set to:", os.environ['INDIC_RESOURCES_PATH'])

# # Debugging: Print the resources path
# from indicnlp import common
# common.set_resources_path(os.environ['INDIC_RESOURCES_PATH'])
# print("Resources Path from common.get_resources_path():", common.get_resources_path())

# # Reinitialize the Indic NLP Library
# from indicnlp import loader
# loader.load()

# # Initialize Indic scripts
# from indicnlp.script import indic_scripts
# indic_scripts.init()


import os

def get_image_file_names(image_directory):
    """Get list of image file names (without extensions) from the image directory."""
    image_files = [os.path.splitext(f)[0] for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]
    return set(image_files)

def get_caption_image_ids(captions_file_path):
    """Get list of image IDs from the captions file."""
    with open(captions_file_path, 'r',encoding='utf-8') as file:
        captions_doc = file.read()
    image_ids = set()
    for line in captions_doc.split("\n"):
        tokens = line.split(",")
        if len(tokens) < 2:
            continue
        image_id = tokens[0].split(".")[0]
        image_ids.add(image_id)
    return image_ids

def cross_check_image_ids(image_directory, captions_file_path):
    """Cross-check image IDs in the captions file with the image files in the directory."""
    image_files = get_image_file_names(image_directory)
    caption_image_ids = get_caption_image_ids(captions_file_path)
    
    missing_image_files = caption_image_ids - image_files
    extra_image_files = image_files - caption_image_ids
    
    if missing_image_files:
        print(f"Missing image files for the following image IDs: {len(missing_image_files)}")
    else:
        print("All image IDs in the captions file have corresponding image files.")
    
    if extra_image_files:
        print(f"Extra image files not referenced in captions file: {len(extra_image_files)}")
    else:
        print("No extra image files found.")

# Set your directories and files
image_directory = 'examples/Images'
captions_file_path = 'examples/captions.txt'

# Cross-check image IDs
cross_check_image_ids(image_directory, captions_file_path)



