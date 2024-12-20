# import os
# import pandas as pd
# from indicnlp.tokenize import sentence_tokenize
# from googletrans import Translator

# class DataLoader:
#     def __init__(self, image_dir, captions_file, language='ne'):
#         self.image_dir = image_dir
#         self.captions_file = captions_file
#         self.language = language
#         self.translator = Translator()
#         self.captions = pd.read_csv(self.captions_file)
#         self._prepare_data()

#     def _prepare_data(self):
#         # Tokenizing and translating captions to Nepali
#         translated_captions = []
#         for caption in self.captions['caption']:
#             tokenized_caption = sentence_tokenize(caption, lang='en')
#             translated_caption = self.translator.translate(tokenized_caption, src='en', dest=self.language).text
#             translated_captions.append(translated_caption)

#         self.captions['nepali_caption'] = translated_captions
#         print("Translation completed and added as 'nepali_caption' in captions.")

#     def get_data(self):
#         return self.captions


# data_loader.py
