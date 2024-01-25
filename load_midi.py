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
        data = []
        labels = []
        styles = []

        with open(self.label_file_path, 'r') as file:
            for line in file:
                line = line.strip().split(',')
                if len(line) == 3:
                    file_name, genre, style = line
                    midi_path = os.path.join(self.midi_folder_path, file_name)

                    if os.path.exists(midi_path):
                        midi_data = pretty_midi.PrettyMIDI(midi_path)
                        processed_data = self.process_midi_data(midi_data, style)
                        data.append(processed_data)
                        labels.append(genre)
                        styles.append(style)

        self.data = data
        self.labels = np.array(labels)
        self.styles = np.array(styles)

    def midi_to_notes(self, midi_data: pretty_midi.PrettyMIDI) -> pd.DataFrame:
        instrument = midi_data.instruments[0]
        notes = {'pitch': [], 'start': [], 'end': [], 'step': [], 'duration': []}

        # Sort the notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes['pitch'].append(note.pitch)
            notes['start'].append(start)
            notes['end'].append(end)
            notes['step'].append(start - prev_start)
            notes['duration'].append(end - start)
            prev_start = start

        return pd.DataFrame(notes)

    def process_midi_data(self, midi_data: pretty_midi.PrettyMIDI, style: str) -> pd.DataFrame:
        # Extracting information from the PrettyMIDI object
        notes_df = self.midi_to_notes(midi_data)

        if not notes_df.empty:
            # Generate random slices of musical fragments
            fragment_duration = 5.0  # You can adjust this value based on your preference
            num_fragments = int(notes_df['end'].max() / fragment_duration)

            fragments = []
            styles = []

            for _ in range(num_fragments):
                start_time = np.random.uniform(0, notes_df['end'].max() - fragment_duration)
                end_time = start_time + fragment_duration

                fragment_notes = notes_df[(notes_df['start'] >= start_time) & (notes_df['end'] <= end_time)]

                if not fragment_notes.empty:
                    fragments.append(fragment_notes)
                    style = fragment_notes['style'].iloc[0]  # Assuming the style is the same for all notes in the fragment
                    styles.append(style)

            if fragments:
                # Concatenate the fragments and create a DataFrame
                result_df = pd.concat(fragments, ignore_index=True)
                result_df['style'] = styles
                return result_df
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    def encode_labels(self):
        self.labels = self.label_encoder.fit_transform(self.labels)

    def split_data(self, test_size=0.2, random_state=42) -> tuple[list[pd.DataFrame], list[pd.DataFrame], np.ndarray, np.ndarray]:
        if test_size >= 1.0:
            raise ValueError("test_size should be less than 1.0")

        X_train, X_test, y_train, y_test, styles_train, styles_test = train_test_split(
            self.data, self.labels, self.styles, test_size=test_size, random_state=random_state)

        # Check if the resulting train set will be empty
        while len(set(y_train)) < len(set(self.labels)):
            X_train, X_test, y_train, y_test, styles_train, styles_test = train_test_split(
                self.data, self.labels, self.styles, test_size=test_size, random_state=random_state)

        processed_X_train = [self.process_midi_data(data, style) for data, style in zip(X_train, styles_train)]
        processed_X_test = [self.process_midi_data(data, style) for data, style in zip(X_test, styles_test)]

        return processed_X_train, processed_X_test, y_train, y_test
