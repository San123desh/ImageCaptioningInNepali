from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Embedding, LSTM, concatenate, Dropout, add,RepeatVector


class ImageCaptioningModel:
    def __init__(self, vocab_size, max_length):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.model = self.build_model()

    def build_model(self):
        """Build the image captioning model."""
        input1 = Input(shape=(1920,))
        input2 = Input(shape=(self.max_length,))
        img_features = Dense(256, activation='relu')(input1)
        img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)
        sentence_features = Embedding(self.vocab_size, 256, mask_zero=True)(input2)
        merged = concatenate([img_features_reshaped, sentence_features], axis=1)
        sentence_features = LSTM(256)(merged)
        x = Dropout(0.5)(sentence_features)
        x = add([x, img_features])
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.vocab_size, activation='softmax')(x)
        model = Model(inputs=[input1, input2], outputs=output)
        
        # Compile the model with accuracy metric
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    

