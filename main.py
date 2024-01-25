from model_music import MusicGenerationModel
from load_midi import MidiDataLoader
import pretty_midi
import numpy as np

# Step 1: Create an instance of MidiDataLoader
midi_loader = MidiDataLoader('midi_files', 'midi_files/piosenka-gatunek.txt')

# Step 2: Load data using the MidiDataLoader
midi_loader.load_data()
print(midi_loader.data)
# Step 1.1: Print Plot of the first file and list of loaded files
#midi_loader.plot_first_file(count=10)
# Step 2: Load data using the MidiDataLoader
midi_loader.load_data()
midi_loader.encode_labels()

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = midi_loader.split_data(test_size=0.2, random_state=42)

# Step 4: Train the music generation model
input_shape = len(X_train) if len(X_train) > 0 else 0  # Use the length of the list
output_shape = len(np.unique(np.array(y_train))) if y_train else 0
style_classes = len(np.unique(y_test)) if y_test else 0

if input_shape > 0 and output_shape > 0:
    model = MusicGenerationModel(input_shape, output_shape, style_classes)
    model.train(X_train, y_train, epochs=50, batch_size=64)

    # Step 5: Generate music in the style of "pop"
    pop_samples = [X[i] for i in range(len(X)) if y[i] == 'pop']
    initial_notes = pop_samples[0] if pop_samples else None
    style_class = 'pop'

    if initial_notes:
        generated_music = model.generate_music(initial_notes, style_class, sequence_length=100)

        # Save the generated music to a new MIDI file
        result_midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        for note_data in generated_music:
            note = pretty_midi.Note(
                velocity=64,
                pitch=int(note_data['pitch']),
                start=float(note_data['start']),
                end=float(note_data['end'])
            )
            instrument.notes.append(note)

        result_midi.instruments.append(instrument)
        result_midi.write('result.midi')
    else:
        print("No pop samples found for generating music.")
else:
    print("Insufficient data for training and testing.")