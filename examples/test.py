import os
from collections import defaultdict

def validate_dataset(image_dir, captions_file):
    """Validate image and caption pairs."""
    # Ensure proper handling of paths
    image_dir = os.path.normpath(image_dir)
    captions_file = os.path.normpath(captions_file)
    
    # Get all image files
    image_files = {f.split('.')[0] for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}
    
    # Process captions file
    caption_ids = set()
    caption_counts = defaultdict(int)
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        next(f)  # Skip header if present
        for line in f:
            if ',' not in line:
                continue
            img_id = line.split(',')[0].split('.')[0]
            caption_ids.add(img_id)
            caption_counts[img_id] += 1
    
    # Find mismatches
    missing_captions = image_files - caption_ids
    missing_images = caption_ids - image_files
    
    # Print results
    print(f"Total images found: {len(image_files)}")
    print(f"Total images with captions: {len(caption_ids)}")
    print("\nMissing captions for images:")
    for img in missing_captions:
        print(f"- {img}.png")
    
    print("\nMissing images for captions:")
    for img in missing_images:
        print(f"- {img}.png")
    
    print("\nCaption counts per image:")
    for img_id, count in caption_counts.items():
        print(f"- {img_id}.png: {count} captions")

if __name__ == "__main__":
    validate_dataset("data/new_test_dataset/Images", "data/new_test_dataset/captions.txt")
