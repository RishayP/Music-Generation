# importing Libraries
import tensorflow 
import mido
import csv
import os
import random
import numpy as np 
import pandas as pd 
from collections import Counter
import random
import IPython
from IPython.display import Image, Audio
import music21
from music21 import *
from music21 import converter
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mido import MidiFile, MidiTrack, Message
import sys
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
np.random.seed(42)

# Function to convert MIDI to CSV
def midi_to_csv(midi_file_path, csv_file_path):
    midi = mido.MidiFile(midi_file_path)
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Track", "Message Type", "Note", "Velocity", "Time"])
        for i, track in enumerate(midi.tracks):
            for msg in track:
                if msg.type == 'note_on' or msg.type == 'note_off':
                    writer.writerow([i, msg.type, msg.note, msg.velocity, msg.time])

# Path to the base directory containing the folders with MIDI files
base_dir = "C:\\Users\\rpuri\\OneDrive - Eastside Preparatory School\\Documents\\12th Grade\\Independent Project\\database"

# Output directory for the CSV files
csv_dir = "C:\\Users\\rpuri\\OneDrive - Eastside Preparatory School\\Documents\\12th Grade\\Independent Project\\csv_files"

output_dir = "C:\\Users\\rpuri\\OneDrive - Eastside Preparatory School\\Documents\\12th Grade\\Independent Project\\output"

# Create the output CSV directory if it doesn't exist
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Traverse through all directories and files
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.mid'):  # Check if the file is a MIDI file
            midi_file_path = os.path.join(root, file)
            csv_file_name = os.path.splitext(file)[0] + ".csv"  # Create corresponding CSV file name
            csv_file_path = os.path.join(csv_dir, csv_file_name)  # Full path for the CSV file

            # Convert the MIDI file to CSV
            midi_to_csv(midi_file_path, csv_file_path)
            print(f"Converted {midi_file_path} to {csv_file_path}")

print("All MIDI files have been converted to CSV files!")

note_sequences = []

# Load and process each CSV file
for file in os.listdir(csv_dir):
    if file.endswith('.csv'):
        csv_path = os.path.join(csv_dir, file)
        data = pd.read_csv(csv_path)
        
        # Extract relevant features (e.g., note, time, velocity)
        notes = data['Note'].tolist()
        times = data['Time'].tolist()
        
        # Here you can choose to combine notes, times, and velocities into sequences
        # This example just stores notes for simplicity.
        note_sequences.append(notes)

# Combine all sequences into a training set
all_sequences = []
sequence_length = 100  # Define the length of sequences for training

for sequence in note_sequences:
    for i in range(len(sequence) - sequence_length):
        all_sequences.append(sequence[i:i + sequence_length])

print(f"Total sequences generated: {len(all_sequences)}")

# Flatten the list of sequences to encode all notes
flat_notes = [note for sequence in all_sequences for note in sequence]

# Use a label encoder to convert notes to numerical values
encoder = LabelEncoder()
encoded_notes = encoder.fit_transform(flat_notes)

# Reshape the encoded notes back into sequences
encoded_sequences = np.array(encoded_notes).reshape(-1, sequence_length)

# Prepare input (X) and output (y) for the model
X = encoded_sequences[:, :-1]  # Input sequences
y = encoded_sequences[:, -1]   # The next note (target)

# One-hot encode the target labels
y = np.eye(len(encoder.classes_))[y]

print(f"Input shape: {X.shape}, Output shape: {y.shape}")

# Model architecture
# LSTM - faster -> chaange to not bidirectional to make it faster
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], 1), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(len(encoder.classes_), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Reshape X to be compatible with LSTM (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train the model
model.fit(X, y, epochs=100, batch_size=64)

print("Training complete!")

# Pick a random sequence to start generation
start_index = random.randint(0, len(X) - 1)
generated_sequence = list(X[start_index])

# Generate a sequence of new notes
for _ in range(500):  # Generate 500 new notes
    input_seq = np.reshape(generated_sequence[-sequence_length:], (1, sequence_length, 1))
    predicted_note = model.predict(input_seq, verbose=0)
    
    # Convert the predicted note back to the original format
    predicted_note_index = np.argmax(predicted_note)
    predicted_note = encoder.inverse_transform([predicted_note_index])[0]
    
    generated_sequence.append(predicted_note)

# Convert the generated sequence into MIDI format or play it back
print("Generated sequence:", generated_sequence)

def create_midi_from_sequence(sequence, output_path):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    for note in sequence:
        track.append(Message('note_on', note=note, velocity=64, time=480))
        track.append(Message('note_off', note=note, velocity=64, time=480))

    midi.save(output_path)
    print(f"Saved generated MIDI to {output_path}")

# Save the generated sequence as a MIDI file
create_midi_from_sequence(generated_sequence, "generated_music.mid")