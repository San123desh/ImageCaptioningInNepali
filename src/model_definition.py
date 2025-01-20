# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# def create_model(vocab_size, max_length):
#     inputs1 = Input(shape=(4096,), name="image")
#     fe1 = Dropout(0.4)(inputs1)
#     fe2 = Dense(256, activation='relu')(fe1)

#     inputs2 = Input(shape=(max_length,), name="text")
#     se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
#     se2 = Dropout(0.4)(se1)
#     se3 = LSTM(256)(se2)

#     decoder1 = add([fe2, se3])
#     decoder2 = Dense(256, activation='relu')(decoder1)
#     outputs = Dense(vocab_size, activation='softmax')(decoder2)

#     model = Model(inputs=[inputs1, inputs2], outputs=outputs)
#     model.compile(loss='categorical_crossentropy', optimizer='adam')
#     return model


from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, add
from tensorflow.keras.models import Model

def define_model(vocab_size, max_length):
    # Encoder model
    inputs1 = Input(shape=(4096,), name="image")
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # Sequence feature layers
    inputs2 = Input(shape=(max_length,), name="text")
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)
    # Decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
