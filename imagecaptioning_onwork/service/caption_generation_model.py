import logging
import os
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Input, Dense, Reshape, Embedding, LSTM, concatenate, Dropout, add
from tensorflow.keras.models import Model
from imagecaptioning_onwork.utils.file_utils import save_tokenizer
from imagecaptioning_onwork.utils.logging_utils import log_metrics
from imagecaptioning_onwork.utils.metrics import tokenize_captions, calculate_corpus_bleu_score
import tensorflow as tf
from tensorflow.keras.layers import Lambda

logger = logging.getLogger(__name__)

class BLEUScoreCallback(Callback):
    def __init__(self, val_generator, tokenizer, max_length, frequency=1):
        self.val_generator = val_generator
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.smoothing_function = SmoothingFunction().method1  # Use smoothing
        self.frequency = frequency

    def on_epoch_end(self, epoch, logs=None):
        total_bleu = 0.0
        num_samples = 0

        for batch in self.val_generator:
            (X1, X2), y_true = batch
            if X1.size == 0 or X2.size == 0:  # Skip empty batches
                continue
            y_pred = self.model.predict([X1, X2], verbose=0)
            y_pred_indices = np.argmax(y_pred, axis=-1)

            for i in range(len(y_true)):
                # Convert one-hot encoded y_true to class index
                true_word_indices = np.argmax(y_true[i], axis=-1)
                true_caption = self.tokenizer.sequences_to_texts([[true_word_indices]])[0]

                # Convert predicted word indices to text
                pred_caption = self.tokenizer.sequences_to_texts([[y_pred_indices[i]]])[0]

                # Tokenize the captions
                true_tokens = true_caption.split()
                pred_tokens = pred_caption.split()

                # Calculate BLEU score with smoothing
                bleu_score = sentence_bleu(
                    [true_tokens],
                    pred_tokens,
                    weights=(0.5, 0.5, 0, 0), 
                    smoothing_function=self.smoothing_function
                )
                total_bleu += bleu_score
                num_samples += 1

        avg_bleu = total_bleu / num_samples if num_samples > 0 else 0.0
        print(f"\nEpoch {epoch + 1}: BLEU Score = {avg_bleu:.4f}")
        logs["val_bleu"] = avg_bleu  # Add BLEU score to logs


class ImageCaptioningModel:
    def __init__(self, vocab_size, max_length):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.model = self.build_model()
        self.tokenizer = None
        self.feature_extractor = None

    def build_model(self):
        """Build the image captioning model."""
        input1 = Input(shape=(1920,))
        input2 = Input(shape=(self.max_length,), dtype="int32")
        # input2 = Input(shape=(self.max_length,))
        # input2_casted = Lambda(lambda x: tf.cast(x, dtype=tf.int32))(input2)
        img_features = Dense(256, activation='relu')(input1)
        # img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)
        img_features_reshaped = Reshape((1, 256))(img_features)
        sentence_features = Embedding(self.vocab_size, 256, mask_zero=True)(input2)
        # sentence_features = Embedding(self.vocab_size, 256, mask_zero=True)(input2_casted)
        merged = concatenate([img_features_reshaped, sentence_features], axis=1)
        sentence_features = LSTM(256)(merged)
        x = Dropout(0.5)(sentence_features)
        # x = add([x, img_features])
        x = add([x, Reshape((256,))(img_features)])
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.vocab_size, activation='softmax')(x)
        model = Model(inputs=[input1, input2], outputs=output)
        # model = Model(inputs=[input1, input2_casted], outputs=output)
        print("input1 shape:", input1.shape)
        print("img_features shape:", img_features.shape)
        print("img_features_reshaped shape:", img_features_reshaped.shape)
        print("input2 shape:", input2.shape)
        # print("input2 shape:", input2_casted.shape)
        print("sentence_features shape:", sentence_features.shape)
        print("merged shape:", merged.shape)

        # Compile the model with accuracy metric
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, train_generator, val_generator, tokenizer, feature_extractor, max_length, epochs=50,
              model_dir="./models"):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        # Define callbacks
        checkpoint = ModelCheckpoint(
            # "model.keras",
            filepath="checkpoints/model_epoch_{epoch:02d}.keras",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        earlystopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=5,
            verbose=1,
            restore_best_weights=True
        )
        learning_rate_reduction = ReduceLROnPlateau(
            monitor='val_loss',
            patience=3,
            verbose=1,
            factor=0.2,
            min_lr=0.000001
        )
        bleu_score_callback = BLEUScoreCallback(val_generator, tokenizer, max_length)

        def convert_dtype(generator):
            for (X1, X2), y in generator:
                X1 = np.array(X1, dtype=np.float32)
                X2 = np.array(X2, dtype=np.int32)  # Ensure int32
                y = np.array(y, dtype=np.float32)
                yield (X1, X2), y

        train_generator = convert_dtype(train_generator)
        val_generator = convert_dtype(val_generator)

        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[checkpoint, earlystopping, learning_rate_reduction, bleu_score_callback]
        )
        # Log training history
        for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            log_metrics(logger, epoch, loss, val_loss,
                        history.history.get('val_bleu', [None] * len(history.history['loss']))[epoch])

        # Define the new directory for saving the models and tokenizer

        # Ensure the directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Save tokenizer and feature extractor
        save_tokenizer(self.tokenizer, os.path.join(model_dir, "tokenizer.pkl"))
        self.feature_extractor.feature_extractor.save(os.path.join(model_dir, "feature_extractor.keras"))

        # Save the trained model
        self.model.save(os.path.join(model_dir, "model.keras"))

    def evaluate_bleu_score(self, val_generator):
       
        references = []
        candidates = []

        for batch in val_generator:
            (X1, X2), y_true = batch
            if X1.size == 0 or X2.size == 0:
                continue  # Skip empty batch
            y_pred = self.model.predict([X1, X2], verbose=0)
            y_pred_indices = np.argmax(y_pred, axis=-1)

            for i in range(len(y_true)):
                true_caption = self.tokenizer.sequences_to_texts([np.argmax(y_true[i], axis=-1)])[0]
                pred_caption = self.tokenizer.sequences_to_texts([y_pred_indices[i]])[0]

                # Tokenize the captions
                references.append([tokenize_captions([true_caption])[0]])
                candidates.append(tokenize_captions([pred_caption])[0])

        # Calculate corpus BLEU score
        bleu_score = calculate_corpus_bleu_score(references, candidates)
        print(f"Validation BLEU Score: {bleu_score:.4f}")
        return bleu_score

    def evaluate(self, val_generator):
        # Evaluate BLEU score on the validation set
        bleu_score = self.evaluate_bleu_score(val_generator)
        logger.info(f"Final Validation BLEU Score: {bleu_score:.4f}")
