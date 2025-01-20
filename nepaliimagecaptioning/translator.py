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

    # def translate_and_tokenize(self, text):
    #     translated_text = self.translate_to_nepali(text)
    #     print(f"Translated text: %s" % translated_text)
    #     if translated_text:
    #         tokenized_text = self.tokenize_nepali(translated_text)
    #         print(f"Tokenized text: {tokenized_text}")
    #     else:
    #         tokenized_text = []  # Empty list if translation fails
    #         print("Translation failed. Using empty tokenized text.")
    #     return translated_text, tokenized_text
    def translate_and_tokenize(self, caption):
        if not caption:
            print("Translation failed. Using empty tokenized text.")
            return "", []
        
        # Add actual translation logic here
        translated_caption = self.translate_to_nepali(caption)
        tokenized_caption = self.tokenize_nepali(translated_caption)
        
        if not translated_caption:
            print("Translation failed. Skipping this image.")
            return "", []

        return translated_caption, tokenized_caption
