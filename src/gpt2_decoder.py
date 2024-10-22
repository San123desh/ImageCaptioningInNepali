import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GPT2Decoder:
    def __init__(self, model_name='gpt2'):
        # Load pre-trained GPT-2 model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Add special tokens for Nepali
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]'})

        # Resize token embeddings to accommodate special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

    def generate_caption(self, image_features, max_length=50):
        # Prepare the image features to use as input prompt for GPT-2
        input_ids = torch.tensor(self.tokenizer.encode('[PAD]')).unsqueeze(0)

        # Generate caption using GPT-2
        output = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=5,        # Beam search for better results
            early_stopping=True,
            repetition_penalty=2.0,  # Penalty to discourage repeated words
        )

        # Decode the generated caption from tokens to string
        caption = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return caption

    def generate_batch_captions(self, image_features_batch):
        captions = []
        for features in image_features_batch:
            caption = self.generate_caption(features)
            captions.append(caption)
        return captions
