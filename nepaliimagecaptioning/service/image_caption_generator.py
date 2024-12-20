from nepaliimagecaptioning.domain import ImageCaption


class ImageCaptionGenerator:
    def __init__(self):
        self.model = None

    def preprocess_image(self, image_path):
        """Preprocesses the image to be compatible with the model."""
        pass

    def generate_caption(self, image_path):
        """Generates a caption for the given image."""
        # Preprocess the image
        image = self.preprocess_image(image_path)

        # Generate caption
        # caption = self.model.generate_caption(image)

        caption = ImageCaption(image_path=image_path, caption="This is a test caption", score=1.0)
        return caption
