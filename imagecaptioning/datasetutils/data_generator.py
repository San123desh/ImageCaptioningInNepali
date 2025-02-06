

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class CustomDataGenerator(Sequence):
    def __init__(self, df, X_col, y_col, batch_size, tokenizer, vocab_size, max_length, features, shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.features = features
        self.shuffle = shuffle
        self.n = len(self.df)

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        """Return the number of batches per epoch."""
        return self.n // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size, :]
        X1, X2, y = self.__get_data(batch)
        return (X1, X2), y

    # def __get_data(self, batch):
    #     """Prepare data for a single batch."""
    #     X1, X2, y = list(), list(), list()
    #     images = batch[self.X_col].tolist()
    #     for image in images:
    #         feature = self.features[image][0]
    #         captions = batch.loc[batch[self.X_col] == image, self.y_col].tolist()
    #         for caption in captions:
    #             seq = self.tokenizer.texts_to_sequences([caption])[0]
    #             for i in range(1, len(seq)):
    #                 in_seq, out_seq = seq[:i], seq[i]
    #                 in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
    #                 out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
    #                 X1.append(feature)
    #                 X2.append(in_seq)
    #                 y.append(out_seq)
    #     return np.array(X1), np.array(X2), np.array(y)
    def __get_data(self, batch):
        """Prepare data for a single batch."""
        X1, X2, y = list(), list(), list()
        images = batch[self.X_col].tolist()
        for image in images:
            if image not in self.features:
                print(f"Warning: Feature for image '{image}' not found. Skipping this image.")
                continue
            feature = self.features[image][0]
            captions = batch.loc[batch[self.X_col] == image, self.y_col].tolist()
            if not captions:
                print(f"Warning: No captions found for image '{image}'. Skipping this image.")
                continue
            for caption in captions:
                seq = self.tokenizer.texts_to_sequences([caption])[0]
                if len(seq) == 0:
                    print(f"Warning: Unable to tokenize caption '{caption}'. Skipping this caption.")
                    continue
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length, padding='pre')[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y)
