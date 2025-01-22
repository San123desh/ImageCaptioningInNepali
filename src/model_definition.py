
# from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, add, Attention, BatchNormalization
# from tensorflow.keras.models import Model
# import tensorflow as tf

# def define_model(vocab_size, max_length):
#     # Encoder model (Image features)
#     inputs1 = Input(shape=(2048,), name="image")
#     fe1 = Dropout(0.3)(inputs1)
#     fe2 = Dense(512, activation='relu')(fe1)
#     fe2 = BatchNormalization()(fe2)

#     # Sequence feature layers (Text)
#     inputs2 = Input(shape=(max_length,), name="text")
#     se1 = Embedding(vocab_size, 300, mask_zero=True)(inputs2)
#     se2 = Dropout(0.3)(se1)
#     se3 = LSTM(512, return_sequences=True)(se2)  # Returning sequences for attention
#     se3 = BatchNormalization()(se3)

#     # Attention mechanism
#     attention = Attention()([se3, tf.expand_dims(fe2, axis=1)])  # Attention between text and image features
#     context_vector = tf.reduce_sum(attention * se3, axis=1)  # Weighted sum of features based on attention weights

#     # Decoder model
#     decoder1 = add([fe2, context_vector])  # Combine the image feature vector and context vector
#     decoder2 = Dense(512, activation='relu')(decoder1)
#     decoder2 = BatchNormalization()(decoder2)
#     outputs = Dense(vocab_size, activation='softmax')(decoder2)

#     # Define the model
#     model = Model(inputs=[inputs1, inputs2], outputs=outputs)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                   metrics=['accuracy'])
#     return model
