import pickle

def save_tokenizer(tokenizer, path):
    """Save the tokenizer to a file."""
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(path):
    """Load the tokenizer from a file."""
    with open(path, "rb") as f:
        return pickle.load(f)