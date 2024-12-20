import torch
import matplotlib.pyplot as plt

class AttentionVisualizer:
    def __init__(self, model):
        self.model = model

    def get_attention_map(self, image_tensor):
        # Set the model in evaluation mode
        self.model.eval()

        # Ensure no gradients are being calculated
        with torch.no_grad():
            outputs = self.model(image_tensor, output_attentions=True)

        # Extract attention weights from the output
        attentions = outputs.attentions  # This is a list of attention maps from each layer

        if attentions is None:
            raise ValueError("No attentions were returned by the model.")

        # Select the attention map from the last layer for visualization
        attention_map = attentions[-1]  # Use the last layer's attention

        # Average attention heads if needed (depends on the shape)
        attention_map = attention_map.mean(dim=1)  # Average across all heads

        return attention_map

    def visualize(self, image_tensor, image):
        # Ensure the image_tensor has 3 channels
        if image_tensor.shape[1] != 3:
            raise ValueError("Expected 3 channel RGB image tensor, got {} channels.".format(image_tensor.shape[1]))

        attention_map = self.get_attention_map(image_tensor)

        # Assuming attention_map is 2D after averaging heads
        attention_map = attention_map.squeeze(0).detach().cpu().numpy()
        
        # Plot the attention map and the original image side by side
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original Image
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        
        # Attention Map
        ax[1].imshow(attention_map, cmap='viridis')
        ax[1].set_title("Attention Map")

        plt.show()
