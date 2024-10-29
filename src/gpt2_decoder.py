import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GPT2Decoder:
    def __init__(self, model_name='gpt2'):
        # Load pre-trained GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        

    def generate_caption(self, image_features, max_length=50):
        if image_features is None or image_features.nelement() == 0:
            raise ValueError("Invalid or empty image features provided to GPT-2 decoder.")
        # Prepare the image features to use as input prompt for GPT-2
        input_ids = torch.tensor(self.tokenizer.encode("")).unsqueeze(0)
        print(f"Input IDs shape: {input_ids.shape}, Input IDs: {input_ids}")
        
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        # Check if input_ids are valid
        if input_ids.nelement() == 0:
            print("Warning: Empty input_ids passed to GPT-2 decoder.")
            return ""

        # Generate caption using GPT-2
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # Pass attention mask
            max_length=max_length,
            num_beams=5,        # Beam search for better results
            # early_stopping=True,
            repetition_penalty=2.0,  # Penalty to discourage repeated words
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1
        )

        # Decode the generated caption from tokens to string
        caption = self.tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return caption
        # input_ids = torch.zeros((1, 1), dtype=torch.long)  # Empty input to start generation

        # # Generate caption using beam search or greedy search
        # output = self.model.generate(
        #     input_ids=input_ids,
        #     max_length=50,  # Max caption length
        #     num_beams=5,    # Use beam search for more optimal results
        #     early_stopping=True
        # )
        
        # # Decode the generated output to text
        # caption = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # return caption

    def generate_batch_captions(self, image_features_batch):
        captions = []
        for features in image_features_batch:
            caption = self.generate_caption(features)
            captions.append(caption)
        return captions
