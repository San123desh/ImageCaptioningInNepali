from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize

def calculate_bleu_score(reference, candidate):
    return sentence_bleu(reference, candidate)

def calculate_corpus_bleu_score(references, candidates):
    return corpus_bleu(references, candidates)

def tokenize_sentence(sentence):
    return word_tokenize(sentence.lower())

def tokenize_captions(captions):
    return [tokenize_sentence(caption) for caption in captions]










