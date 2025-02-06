

# # new
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.optimizers import Adam
# import matplotlib.font_manager as fm

# class CaptionGenerator:
#     def __init__(self, model_path, tokenizer_path, feature_extractor_path, max_length=31, img_size=224):
#         self.model_path = model_path
#         self.tokenizer_path = tokenizer_path
#         self.feature_extractor_path = feature_extractor_path
#         self.max_length = max_length
#         self.img_size = img_size

#         # Load the model and compile it
#         self.caption_model = load_model(model_path)
#         self.caption_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

#         # Load the feature extractor
#         self.feature_extractor = load_model(feature_extractor_path)

#         # Load the tokenizer
#         with open(tokenizer_path, "rb") as f:
#             self.tokenizer = pickle.load(f)

#         # Set the font for Devanagari text
#         self.font_prop = fm.FontProperties(fname='kalimati-regular/Kalimati Regular.otf')

#     def generate_caption(self, image_path):
#         """Generate and display a caption for the given image."""
#         # Load and preprocess the image
#         img = load_img(image_path, target_size=(self.img_size, self.img_size))
#         img = img_to_array(img) / 255.0
#         img = np.expand_dims(img, axis=0)

#         # Extract image features
#         image_features = self.feature_extractor.predict(img, verbose=0)

#         # Generate caption
#         in_text = "startseq"
#         for i in range(self.max_length):
#             sequence = self.tokenizer.texts_to_sequences([in_text])[0]
#             sequence = pad_sequences([sequence], maxlen=self.max_length)
#             yhat = self.caption_model.predict([image_features, sequence], verbose=0)
#             yhat_index = np.argmax(yhat)
#             word = self.tokenizer.index_word.get(yhat_index, None)
#             if word is None:
#                 break
#             in_text += " " + word
#             if word == "endseq":
#                 break

#         # Clean up the caption
#         caption = in_text.replace("startseq", "").replace("endseq", "").strip()

#         # Print the caption to the terminal
#         print(f"Generated Caption: {caption}")

#         # Display the image with the caption
#         plt.figure(figsize=(8, 8))
#         plt.imshow(img[0])
#         plt.axis('off')
#         plt.title(caption, fontproperties=self.font_prop)
#         plt.show()



import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import matplotlib.font_manager as fm

class CaptionGenerator:
    def __init__(self, model_path, tokenizer_path, feature_extractor_path, max_length=31, img_size=224, font_path='kalimati-regular/Kalimati Regular.otf'):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.feature_extractor_path = feature_extractor_path
        self.max_length = max_length
        self.img_size = img_size

        # Load the model and compile it
        self.caption_model = load_model(model_path)
        self.caption_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        # Load the feature extractor
        self.feature_extractor = load_model(feature_extractor_path)

        # Load the tokenizer
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        # Set the font for Nepali text (Kalimati Regular)
        self.dev_prop = fm.FontProperties(fname=font_path)

    def beam_search_predictions(self, image_features, beam_index=3):
        start = [self.tokenizer.word_index['startseq']]
        start_word = [[start, 0.0]]
        
        while len(start_word[0][0]) < self.max_length:
            temp = []
            for s in start_word:
                par_seq = pad_sequences([s[0]], maxlen=self.max_length, padding='post')
                preds = self.caption_model.predict([image_features, par_seq], verbose=0)[0]
                
                word_preds = np.argsort(preds)[-beam_index:]
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds[w]
                    temp.append([next_cap, prob])
                    
            start_word = sorted(temp, reverse=True, key=lambda l: l[1])
            start_word = start_word[:beam_index]
        
        final_caption = start_word[-1][0]
        final_caption = [self.tokenizer.index_word.get(idx, '') for idx in final_caption]
        final_caption = final_caption[1:]  # Remove 'startseq'
        final_caption = ' '.join(final_caption)
        return final_caption

    def generate_caption(self, image_path):
        """Generate and display a caption for the given image."""
        # Load and preprocess the image
        img = load_img(image_path, target_size=(self.img_size, self.img_size))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Extract image features
        image_features = self.feature_extractor.predict(img, verbose=0)

        # Use beam search to generate caption
        caption = self.beam_search_predictions(image_features, beam_index=3)

        # Clean up the caption
        caption = caption.replace("startseq", "").replace("endseq", "").strip()

        # Print the caption to the terminal
        print(f"Generated Caption: {caption}")

        # Display the image with the caption
        plt.figure(figsize=(8, 8))
        plt.imshow(img[0])
        plt.axis('off')
        plt.title(caption, fontproperties=self.dev_prop)
        plt.show()

