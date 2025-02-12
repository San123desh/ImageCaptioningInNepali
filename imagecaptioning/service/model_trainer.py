from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os

class BLEUScoreCallback(Callback):
    def __init__(self, val_generator, tokenizer, max_length):
        self.val_generator = val_generator
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.smoothing_function = SmoothingFunction().method1  # Use smoothing

    def on_epoch_end(self, epoch, logs=None):
        total_bleu = 0.0
        num_samples = 0

        for batch in self.val_generator:
            (X1, X2), y_true = batch
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

        avg_bleu = total_bleu / num_samples
        print(f"\nEpoch {epoch + 1}: BLEU Score = {avg_bleu:.4f}")
        logs["val_bleu"] = avg_bleu  # Add BLEU score to logs

        
class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, train_generator, val_generator, tokenizer, max_length, epochs=50):
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

        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[checkpoint, earlystopping, learning_rate_reduction, bleu_score_callback]
        )
        return history