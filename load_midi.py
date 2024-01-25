import pretty_midi
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

class MidiDataLoader:
    def __init__(self, midi_folder_path, label_file_path):
        self.midi_folder_path = midi_folder_path
        self.label_file_path = label_file_path
        self.data = None
        self.labels = None
        self.styles = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        try:
            data = []
            labels = []
            styles = []

            with open(self.label_file_path, 'r') as file:
                for line in file:
                    print('Loading', line)
                    line = line.strip().split(',')
                    if len(line) == 2:
                        file_name, genre = line
                        midi_path = os.path.join(self.midi_folder_path, file_name)
                        print(midi_path)
                        if os.path.exists(midi_path):
                            midi_data = pretty_midi.PrettyMIDI(midi_path)
                            processed_data = self.process_midi_data(midi_data, genre)
                            data.append(processed_data)
                            print(data, "--------------------------------")
                            labels.append(genre)
                        else:
                            print(f"Warning: MIDI file not found for {file_name}")

            if not data:
                print("Warning: No MIDI data loaded.")

            self.data = data
            self.labels = np.array(labels)
            self.styles = np.array(styles)
        except Exception as e:
            print(f"Error during data loading: {e}")

    def midi_to_notes(self, midi_data: pretty_midi.PrettyMIDI) -> pd.DataFrame:
        # Remaining code unchanged

    def process_midi_data(self, midi_data: pretty_midi.PrettyMIDI, genre: str) -> pd.DataFrame:
        # Remaining code unchanged

    def encode_labels(self):
        try:
            self.labels = self.label_encoder.fit_transform(self.labels)
        except Exception as e:
            print(f"Error during label encoding: {e}")

    def plot_first_file(self, count: Optional[int] = None):
        try:
            if not self.data or not self.data[0]:
                print("Warning: No data available for plotting.")
                return

            if count:
                title = f'First {count} notes'
            else:
                title = f'Whole track'
                count = len(self.data[0]['pitch'])

            plt.figure(figsize=(20, 4))
            plot_pitch = np.stack([self.data[0]['pitch'], self.data[0]['pitch']], axis=0)
            plot_start_stop = np.stack([self.data[0]['start'], self.data[0]['end']], axis=0)
            plt.plot(
                plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
            plt.xlabel('Time [s]')
            plt.ylabel('Pitch')
            _ = plt.title(title)
            plt.show()
        except Exception as e:
            print(f"Error during plotting: {e}")

    def split_data(self, test_size=0.2, random_state=42) -> tuple[list[pd.DataFrame], list[pd.DataFrame], np.ndarray, np.ndarray]:
        try:
            if test_size >= 1.0:
                raise ValueError("test_size should be less than 1.0")

            if not self.data or not self.labels or not self.styles:
                print("Insufficient data for training and testing.")
                return [], [], np.array([]), np.array([])

            X_train, X_test, y_train, y_test, styles_train, styles_test = train_test_split(
                self.data, self.labels, self.styles, test_size=test_size, random_state=random_state)

            # Check if the resulting train set will be empty
            while len(set(y_train)) < len(set(self.labels)):
                X_train, X_test, y_train, y_test, styles_train, styles_test = train_test_split(
                    self.data, self.labels, self.styles, test_size=test_size, random_state=random_state)

            # Check if resulting sets are empty
            if not X_train or not X_test or not y_train.size or not y_test.size:
                print("Insufficient data for training and testing.")
                return [], [], np.array([]), np.array([])

            processed_X_train = [self.process_midi_data(data, style) for data, style in zip(X_train, styles_train)]
            processed_X_test = [self.process_midi_data(data, style) for data, style in zip(X_test, styles_test)]

            return processed_X_train, processed_X_test, y_train, y_test
        except Exception as e:
            print(f"Error during data splitting: {e}")
            return [], [], np.array([]), np.array([])