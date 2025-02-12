

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class CustomDataGenerator(Sequence):
    def __init__(self, df, X_col, y_col, batch_size, tokenizer, vocab_size, max_length, features, shuffle=True):
        super().__init__()
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
        self.on_epoch_end()
        

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        """Return the number of batches per epoch."""
        # return self.n // self.batch_size
        # return math.ceil(self.n / self.batch_size)
        return (self.n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, self.n)  # Prevent index overflow
        batch = self.df.iloc[start:end, :]

        # If batch is empty (due to missing features/captions), return next batch
        if batch.empty:
            next_index = (index + 1) % self.__len__()
            if next_index == index:  # Avoid infinite loop if all batches are empty
                raise ValueError("All batches are empty. Check your data and generator settings.")
            return self.__getitem__(next_index)
        

        """Generate one batch of data."""
        # batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size, :]
        X1, X2, y = self.__get_data(batch)
        if len(X1) == 0:
            print(f"Warning: Empty batch at index {index}. Skipping.")
            return self.__getitem__((index + 1) % self.__len__())
        return (X1, X2), y

    def __get_data(self, batch):
        """Prepare data for a single batch."""
        X1, X2, y = list(), list(), list()
        images = batch[self.X_col].tolist()
        for image in images:
            if image not in self.features:
                print(f"Warning: Feature for image '{image}' not found. Skipping this image.")
                continue
            feature = self.features[image][0]
            feature = np.squeeze(feature)
            if feature.ndim != 1 or feature.shape[0] != 1920:
                print(f"Warning: Invalid feature shape for image '{image}'. Expected (1920,), found {feature.shape}. Skipping.")
                continue
           
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
                    # in_seq = pad_sequences([in_seq], maxlen=self.max_length, padding='pre',dtype='int32')[0]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length, padding='pre')[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
                    
        # return np.array(X1), np.array(X2), np.array(y)
        return np.array(X1, dtype=np.float32), np.array(X2, dtype=np.int32), np.array(y, dtype=np.float32)


