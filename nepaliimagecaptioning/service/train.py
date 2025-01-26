# import numpy as np
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical

# class Trainer:
#     def __init__(self, model, tokenizer, max_length, vocab_size):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.vocab_size = vocab_size

#     def data_generator(self, data_keys, mapping, features, batch_size):
#         X1, X2, y = [], [], []
#         n = 0
#         while True:
#             for key in data_keys:
#                 captions = mapping.get(key, [])
#                 for caption in captions:
#                     seq = self.tokenizer.texts_to_sequences([caption])[0]
#                     for i in range(1, len(seq)):
#                         in_seq, out_seq = seq[:i], seq[i]
#                         in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
#                         out_seq = to_categorical([out_seq], num_classes=self.vocab_size)
                        
#                         X1.append(features[key][0])
#                         X2.append(in_seq)
#                         y.append(out_seq)
#                         n += 1
                        
#                         if n == batch_size:
#                             yield {"image": np.array(X1), "text": np.array(X2)}, np.array(y)
#                             X1, X2, y = [], [], []
#                             n = 0
#             if n > 0:
#                 yield {"image": np.array(X1), "text": np.array(X2)}, np.array(y)
#                 X1, X2, y = [], [], []
#                 n = 0

#     def train(self, train_keys, mapping, features, batch_size, epochs, steps_per_epoch):
#         for epoch in range(epochs):
#             print(f"Epoch {epoch + 1}/{epochs}")
#             generator = self.data_generator(train_keys, mapping, features, batch_size)
#             self.model.fit(generator, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)

#     @staticmethod
#     def save_model(model, path):
#         model.save(path)
#         print(f"Model saved to {path}")


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class Trainer:
    def __init__(self, model, tokenizer, max_length, vocab_size):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size

    def data_generator(self, data_keys, mapping, features, batch_size):
        X1, X2, y = [], [], []
        n = 0
        while True:
            for key in data_keys:
                captions = mapping.get(key, [])
                for caption in captions:
                    seq = self.tokenizer.texts_to_sequences([caption])[0]
                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                        out_seq = to_categorical([out_seq], num_classes=self.vocab_size)
                        # out_seq = np.expand_dims(out_seq, axis=0)

                        X1.append(features[key][0])
                        X2.append(in_seq)
                        y.append(out_seq)
                        n += 1
                        
                        if n == batch_size:
                            X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                            X1 = tf.cast(X1, tf.float32)
                            X2 = tf.cast(X2, tf.int32)
                            y = np.array(y).reshape((batch_size, self.vocab_size))
                            y = tf.cast(y, tf.float32)
                            yield {"image": X1, "text": X2}, y
                            X1, X2, y = [], [], []
                            n = 0
            if n > 0:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                X1 = tf.cast(X1, tf.float32)
                X2 = tf.cast(X2, tf.int32)
                y = np.array(y).reshape((n, self.vocab_size))
                y = tf.cast(y, tf.float32)
                yield {"image": X1, "text": X2}, y
                X1, X2, y = [], [], []
                n = 0

    def train(self, train_keys, mapping, features, batch_size, epochs, steps_per_epoch):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            generator = self.data_generator(train_keys, mapping, features, batch_size)
            self.model.fit(generator, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)

    @staticmethod
    def save_model(model, path):
        model.save(path)
        print(f"Model saved to {path}")
