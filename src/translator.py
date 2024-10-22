from deep_translator import GoogleTranslator
from indicnlp.tokenize import indic_tokenize

class NepaliTranslator:
    def __init__(self):
        self.translator = GoogleTranslator(source='en', target='ne')

    def translate_to_nepali(self, text):
        try:
            translated = self.translator.translate(text)
        except Exception as e:
            print(f"Translation failed: {e}")
            translated = text  # Fallback to the original text in case of failure
        return translated

    def tokenize_nepali(self, text):
        tokens = indic_tokenize.trivial_tokenize(text)
        return tokens

    def translate_and_tokenize(self, text):
        translated_text = self.translate_to_nepali(text)
        tokenized_text = self.tokenize_nepali(translated_text)
        return translated_text, tokenized_text
