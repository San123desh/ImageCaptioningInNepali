
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding

def load_glove_embeddings(filepath, embedding_dim):
    """Load GloVe embeddings from a file."""
    embeddings_index = {}
    with open(filepath, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def create_embedding_matrix(tokenizer, embeddings_index, embedding_dim):
    """Create an embedding matrix for the given tokenizer and embeddings."""
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_embedding_layer(embedding_matrix, vocab_size, trainable=True):
    """Create an embedding layer with the given embedding matrix."""
    return Embedding(input_dim=embedding_matrix.shape[0],
                     output_dim=embedding_matrix.shape[1],
                     weights=[embedding_matrix],
                     trainable=trainable)

