from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize

def calculate_bleu_score(reference, candidate):
    """
    Calculate the BLEU score for a single candidate sentence against a reference sentence.

    Args:
        reference (list): List of reference sentences (each sentence is a list of words).
        candidate (list): Candidate sentence (list of words).

    Returns:
        float: BLEU score.
    """
    return sentence_bleu(reference, candidate)

def calculate_corpus_bleu_score(references, candidates):
    """
    Calculate the corpus-level BLEU score for a list of candidate sentences against a list of reference sentences.

    Args:
        references (list): List of reference sentences (each sentence is a list of words).
        candidates (list): List of candidate sentences (each sentence is a list of words).

    Returns:
        float: Corpus BLEU score.
    """
    return corpus_bleu(references, candidates)

def tokenize_sentence(sentence):
    """
    Tokenize a sentence into words.

    Args:
        sentence (str): Input sentence.

    Returns:
        list: List of tokens.
    """
    return word_tokenize(sentence.lower())

def tokenize_captions(captions):
    """
    Tokenize a list of captions into words.

    Args:
        captions (list): List of captions.

    Returns:
        list: List of tokenized captions.
    """
    return [tokenize_sentence(caption) for caption in captions]