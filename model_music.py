import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np

class MusicGenerationModel:
    def __init__(self, input_shape, output_shape, style_classes):
        self.model = self.build_model(input_shape, output_shape)
        self.style_classes = style_classes

    def build_model(self, input_shape, output_shape):
        # Check if GPU is available and set TensorFlow to use it
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        model = Sequential()

        # LSTM layer with 128 units
        model.add(LSTM(128, input_shape=(None, input_shape), return_sequences=True))
        # LSTM layer with 128 units
        model.add(LSTM(128))
        # Dense layer with softmax activation at the end
        model.add(Dense(output_shape, activation='softmax'))

        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def train(self, X_train, y_train, epochs=50, batch_size=64):
        # Prepare data for training
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Train the model
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def generate_music(self, initial_notes, style_class, sequence_length=100):
        generated_sequence = initial_notes

        for _ in range(sequence_length):
            # Prepare input data for the model
            input_sequence = np.array([generated_sequence[-len(initial_notes):]])

            # Prediction of the next note
            predicted_probs = self.model.predict(input_sequence)[0]

            # Choose the note based on the probability distribution
            next_note_index = np.random.choice(len(predicted_probs), p=predicted_probs)
            next_note = self.style_classes[next_note_index]

            # Add the chosen note to the generated sequence
            generated_sequence.append(next_note)

        return generated_sequence
