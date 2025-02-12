import matplotlib.pyplot as plt

from imagecaptioning_onwork.service.caption_generator import CaptionGenerator

if __name__ == '__main__':
    # Generate captions for a sample image
    caption_generator = CaptionGenerator(model_dir="./models")
    caption, img = caption_generator.generate_caption("images/Screenshot 2024-01-03 202502.png")
    print(caption)
    # Print the caption to the terminal
    print(f"Generated Caption: {caption}")

    # Display the image with the caption
    plt.figure(figsize=(8, 8))
    plt.imshow(img[0])
    plt.axis('off')
    plt.title(caption, fontproperties=caption_generator.dev_prop)
    plt.show()
