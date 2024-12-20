import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class NepaliImageCaptionTrainer:
    def __init__(self, vocab_size=5000, max_length=50, embedding_dim=256, units=512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.units = units

        # Initialize the CNN (InceptionV3)
        inception = InceptionV3(weights='imagenet')
        self.cnn_model = Model(inception.input, inception.layers[-2].output)

    def build_model(self):
        # CNN Feature Extractor
        inputs1 = Input(shape=(299, 299, 3))
        fe1 = self.cnn_model(inputs1)
        fe2 = Dense(self.embedding_dim)(fe1)

        # Sequence Model
        inputs2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, self.embedding_dim)(inputs2)
        se2 = LSTM(self.units, return_sequences=True)(se1)
        se3 = LSTM(self.units)(se2)

        # Decoder
        decoder1 = concatenate([fe2, se3])
        decoder2 = Dense(self.units, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)

        # Combined Model
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        return model

    def preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img

    def prepare_text_data(self, captions):
        # Initialize tokenizer
        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<unk>")
        tokenizer.fit_on_texts(captions)

        # Create word-to-index and index-to-word mappings
        word2idx = tokenizer.word_index
        idx2word = {v: k for k, v in word2idx.items()}

        # Add special tokens
        word2idx['<start>'] = len(word2idx) + 1
        word2idx['<end>'] = len(word2idx) + 1
        idx2word[len(idx2word) + 1] = '<start>'
        idx2word[len(idx2word) + 1] = '<end>'

        return tokenizer, word2idx, idx2word

    def create_sequences(self, tokenizer, captions):
        # Convert captions to sequences
        sequences = tokenizer.texts_to_sequences(captions)
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        return padded_sequences

    def generate_caption(self, model, image_path, tokenizer, max_length):
        # Preprocess image
        image = self.preprocess_image(image_path)

        # Initialize caption generation
        in_text = '<start>'
        for _ in range(max_length):
            # Encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)

            # Predict next word
            yhat = model.predict([image, sequence], verbose=0)
            yhat = np.argmax(yhat)

            # Convert word index to word
            word = ''
            for w, index in tokenizer.word_index.items():
                if index == yhat:
                    word = w
                    break

            # Stop if end token is generated
            if word == '<end>':
                break

            # Append word to caption
            in_text += ' ' + word

        # Remove start token
        caption = in_text.replace('<start>', '')
        return caption.strip()

    def train(self, train_images, train_captions, epochs=10, batch_size=32):
        """
        Train the model with the provided dataset

        Args:
            train_images: List of preprocessed images
            train_captions: List of corresponding Nepali captions
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Prepare text data
        tokenizer, _, _ = self.prepare_text_data(train_captions)
        sequences = self.create_sequences(tokenizer, train_captions)

        # Build and train model
        model = self.build_model()
        model.fit(
            [train_images, sequences[:, :-1]],
            sequences[:, 1:],
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        return model


# Example usage:
if __name__ == '__main__':
    # Initialize the model
    captioning_model = NepaliImageCaptionTrainer()

    # Prepare your dataset
    train_images = [...]  # Your preprocessed images
    train_captions = [...]  # Your Nepali captions

    # Train the model
    model = captioning_model.train(train_images, train_captions)

    # Generate caption for a new image
    image_path = "path_to_image.jpg"
    caption = captioning_model.generate_caption(model, image_path, tokenizer, max_length=50)
    print(f"Generated Nepali Caption: {caption}")
