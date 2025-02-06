# # from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, add, BatchNormalization, RepeatVector, Concatenate
# # from tensorflow.keras.models import Model
# # import tensorflow as tf

# # class ImageCaptionModel:
# #     def __init__(self, vocab_size, max_length):
# #         self.vocab_size = vocab_size
# #         self.max_length = max_length
# #         self.model = self.define_model()
    
# #     def define_model(self):
# #         inputs1 = Input(shape=(2048,), name="image")
# #         fe1 = Dropout(0.5)(inputs1)
# #         fe2 = Dense(512, activation='tanh')(fe1)
# #         fe2 = BatchNormalization()(fe2)
# #         fe3 = Dense(256, activation='tanh')(fe2)
# #         fe3 = BatchNormalization()(fe3)
# #         fe3 = RepeatVector(self.max_length)(fe3)

# #         inputs2 = Input(shape=(self.max_length,), name="text")
# #         se1 = Embedding(self.vocab_size, 300, mask_zero=True)(inputs2)
# #         se2 = Dropout(0.5)(se1)
# #         se3 = LSTM(512,return_sequences=True)(se2)
# #         se4 = LSTM(256,return_sequences=True)(se3)
# #         se4 = BatchNormalization()(se4)

# #         decoder1 = add([fe3, se4])
# #         decoder2 = Dense(512, activation='tanh')(decoder1)
# #         decoder2 = BatchNormalization()(decoder2)
# #         decoder3 = Dense(256, activation='tanh')(decoder2)
# #         decoder3 = BatchNormalization()(decoder3)
# #         outputs = Dense(self.vocab_size, activation='softmax')(decoder3)

# #         model = Model(inputs=[inputs1, inputs2], outputs=outputs)
# #         model.compile(loss='categorical_crossentropy',
# #                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
# #                       metrics=['accuracy'])
# #         return model

# #     def get_model(self):
# #         return self.model


# from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, add, BatchNormalization, RepeatVector, Reshape
# from tensorflow.keras.models import Model
# import tensorflow as tf

# class ImageCaptionModel:
#     def __init__(self, vocab_size, max_length):
#         self.vocab_size = vocab_size
#         self.max_length = max_length
#         self.model = self.define_model()

#     def define_model(self):
#         inputs1 = Input(shape=(2048,), name="image", dtype=tf.float32)
#         fe1 = Dropout(0.5)(inputs1)
#         fe2 = Dense(512, activation='tanh')(fe1)
#         fe2 = BatchNormalization()(fe2)
#         fe3 = Dense(256, activation='tanh')(fe2)  
#         fe3 = BatchNormalization()(fe3)        
#         fe3 = RepeatVector(self.max_length)(fe3)

#         inputs2 = Input(shape=(self.max_length,), name="text", dtype= tf.int32)
#         se1 = Embedding(self.vocab_size, 300, mask_zero=True)(inputs2)
#         se2 = Dropout(0.5)(se1)
#         se3 = LSTM(256, return_sequences=True)(se2) 
#         se4 = LSTM(256, return_sequences=True)(se3)  
#         se4 = BatchNormalization()(se4)

#         se4 = Reshape((self.max_length, 256))(se4)

#         decoder1 = add([fe3, se4])
#         decoder2 = LSTM(512)(decoder1)
#         decoder2 = BatchNormalization()(decoder2)
#         decoder3 = Dense(512, activation='tanh')(decoder2)
#         decoder3 = BatchNormalization()(decoder3)
#         decoder4 = Dense(256, activation='tanh')(decoder3)
#         decoder4 = BatchNormalization()(decoder4)
#         outputs = Dense(self.vocab_size, activation='softmax')(decoder4)

#         model = Model(inputs=[inputs1, inputs2], outputs=outputs)
#         model.compile(loss='categorical_crossentropy',
#                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                       metrics=['accuracy'])
#         return model

#     def get_model(self):
#         return self.model

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, add, BatchNormalization, RepeatVector, Attention
from tensorflow.keras.models import Model
from utils.embedding_utils import get_embedding_layer

class ImageCaptionModel:
    def __init__(self, vocab_size, max_length, embedding_matrix):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_matrix = embedding_matrix
        self.model = self.define_model()

    def define_model(self):
        # Image feature extractor
        inputs1 = Input(shape=(2048,), name="image")
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(512, activation='tanh')(fe1)
        fe2 = BatchNormalization()(fe2)
        fe3 = Dense(256, activation='tanh')(fe2)
        fe3 = BatchNormalization()(fe3)
        fe3 = RepeatVector(self.max_length)(fe3)

        # Text feature extractor
        embedding_layer = get_embedding_layer(self.embedding_matrix, self.vocab_size, trainable=True)
        inputs2 = Input(shape=(self.max_length,), name="text")
        se1 = embedding_layer(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256, return_sequences=True, activation='tanh')(se2)
        se4 = LSTM(256, return_sequences=True, activation='tanh')(se3)
        se4 = BatchNormalization()(se4)

        # Attention mechanism
        attention = Attention()([fe3, se4])
        attention = BatchNormalization()(attention)

        # Combine image and text features
        decoder1 = add([fe3, se4])
        decoder2 = LSTM(512)(decoder1)
        decoder2 = BatchNormalization()(decoder2)
        decoder3 = Dense(512, activation='tanh')(decoder2)
        decoder3 = BatchNormalization()(decoder3)
        decoder4 = Dense(256, activation='tanh')(decoder3)
        decoder4 = BatchNormalization()(decoder4)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder4)

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])
        return model

    def get_model(self):
        return self.model


