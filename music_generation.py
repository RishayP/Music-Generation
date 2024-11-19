import tensorflow as tf
import mido
import csv
import os
import random
import numpy as np
import pandas as pd
from collections import Counter
from music21 import *
import warnings
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import losses
from tensorflow.keras.models import Sequential

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mido import MidiFile, MidiTrack, Message
from midi2audio import FluidSynth


# Suppress warnings and set seed for reproducibility
warnings.filterwarnings("ignore")
np.random.seed(42)

# Paths for MIDI and CSV files
#midi_folder = "C:\\Users\\rpuri\\OneDrive - Eastside Preparatory School\\Documents\\12th Grade\\Independent Project\\database\\bach"
#csv_folder = "C:\\Users\\rpuri\\OneDrive - Eastside Preparatory School\\Documents\\12th Grade\\Independent Project\\csv_files\\bach"

# Path for saving generated tracks
#output_folder = "C:\\Users\\rpuri\\OneDrive - Eastside Preparatory School\\Documents\\12th Grade\\Independent Project\\output\\"

midi_folder = "C:\\Users\\rpuri\\OneDrive - Eastside Preparatory School\\Documents\\12th Grade\\Independent Project\\database\\albeinz"
csv_folder = "C:\\Users\\rpuri\\OneDrive - Eastside Preparatory School\\Documents\\12th Grade\\Independent Project\\csv_files\\albeinz"
output_folder = "C:\\Users\\rpuri\\OneDrive - Eastside Preparatory School\\Documents\\12th Grade\\Independent Project\\output"

# Create CSV folder if it does not exist
os.makedirs(csv_folder, exist_ok=True)

# Function to convert MIDI to CSV
def midi_to_csv(midi_file_path, csv_file_path):
    try:
        print(f"Processing MIDI file: {midi_file_path}")
        midi_file_path2 = midi_file_path # + "\\alb_esp1.mid"
        midi = mido.MidiFile(midi_file_path2)

        # Default tempo
        tempo = 500000  # 120 BPM (500,000 microseconds per beat)
        time = 0  # To track the running time for each note

        # Check if MIDI file has tracks
        if len(midi.tracks) == 0:
            print(f"Warning: No tracks found in {midi_file_path}")
            return False

        csv_file_path2 = csv_file_path # + "\\1.csv"
        with open(csv_file_path2, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Track", "Message Type", "Note", "Velocity", "Duration", "Tempo", "Time"])

            for i, track in enumerate(midi.tracks):
                for msg in track:
                    time += msg.time  # Accumulate time for each message

                    # Capture tempo change
                    if msg.type == 'set_tempo':
                        tempo = msg.tempo  # Update tempo

                    # Handle 'note_on' messages
                    if msg.type == 'note_on':
                        # If velocity is 0, treat it as 'note_off'
                        if msg.velocity == 0:
                            continue  # Skip zero velocity messages (equivalent to 'note_off')
                        
                        # Calculate note duration by capturing the time from 'note_on' to 'note_off' messages
                        note_duration = 0
                        for sub_msg in track:
                            if sub_msg.type == 'note_off' and sub_msg.note == msg.note:
                                note_duration = sub_msg.time
                                break
                        writer.writerow([i, msg.type, msg.note, msg.velocity, note_duration, tempo, time])

        print(f"Successfully converted {midi_file_path} to {csv_file_path}")
        return True

    except Exception as e:
        print(f"Error converting {midi_file_path} to CSV: {e}")
        return False


# Function to convert all MIDI files in the folder
def convert_all_midi_files(midi_folder, csv_folder):
    midi_files = [f for f in os.listdir(midi_folder) if f.lower().endswith('.mid')]

    if len(midi_files) == 0:
        print(f"No MIDI files found in {midi_folder}. Exiting.")
        return

    print(f"Found {len(midi_files)} MIDI files in {midi_folder}.")

    converted_count = 0
    for midi_filename in midi_files:
        midi_file_path = os.path.join(midi_folder, midi_filename)
        csv_filename = os.path.splitext(midi_filename)[0] + ".csv"
        csv_file_path = os.path.join(csv_folder, csv_filename)

        if os.path.exists(csv_file_path) and os.stat(csv_file_path).st_size > 0:
            print(f"CSV file already exists and is not empty: {csv_file_path}")
            continue

        success = midi_to_csv(midi_file_path, csv_file_path)
        if success:
            converted_count += 1

    print(f"Conversion complete. {converted_count} out of {len(midi_files)} files were converted successfully.")

# load all the data into the csv file
def load_data(csv_folder):
    """
    Function to load and combine CSV files from a given folder.
    Returns X (features) and y_note, y_velocity, y_duration, y_tempo (labels).
    """
    data_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    print(f"Found {len(data_files)} CSV files. Loading data...")
    
    X = []
    y_note = []
    y_velocity = []
    y_duration = []
    y_tempo = []

    for file in data_files:
        file_path = os.path.join(csv_folder, file)
        df = pd.read_csv(file_path)
        
        # Normalize column names to lowercase to avoid case sensitivity issues
        df.columns = [col.lower() for col in df.columns]
        
        # Check if all necessary columns exist after normalization
        required_columns = ['note', 'velocity', 'duration', 'tempo']
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: Missing one or more columns in {file}")
            continue
        
        # Extract relevant data
        notes = df['note'].values
        velocities = df['velocity'].values
        durations = df['duration'].values
        tempos = df['tempo'].values
        
        # Features - Assume 'track' and 'time' columns are features
        features = df[['track', 'time']].values  # Adjust this if you want other features
        
        # Append data for X (features) and labels
        X.append(features)
        y_note.append(notes)
        y_velocity.append(velocities)
        y_duration.append(durations)
        y_tempo.append(tempos)

    # Check if X has any data before attempting concatenation
    if X:
        X = np.concatenate(X, axis=0)
        y_note = np.concatenate(y_note, axis=0)
        y_velocity = np.concatenate(y_velocity, axis=0)
        y_duration = np.concatenate(y_duration, axis=0)
        y_tempo = np.concatenate(y_tempo, axis=0)
    else:
        print("No data found to concatenate!")
        return None, None, None, None, None

    #y_note = np.clip(y_note, 0, 47)  # Assuming output size is 48 for note
    #y_velocity = np.clip(y_velocity, 0, 76)  # Assuming output size is 77 for velocity
    #y_duration = np.clip(y_duration, 0, 0)  # Assuming only one class (binary or fixed output)
    #y_tempo = np.clip(y_tempo, 0, 0)

    return X, y_note, y_velocity, y_duration, y_tempo

# prepare the necessary steps for training
def prepare_sequences(data, seq_length=50):
    # Unpack the data (assuming data is a list of tuples)
    notes, velocities, durations, tempos = zip(*data)  
    
    # Initialize label encoders
    note_encoder = LabelEncoder()
    encoded_notes = note_encoder.fit_transform(notes)

    # Normalize other features using LabelEncoder
    velocity_encoder = LabelEncoder()
    encoded_velocities = velocity_encoder.fit_transform(velocities)
    
    duration_encoder = LabelEncoder()
    encoded_durations = duration_encoder.fit_transform(durations)
    
    tempo_encoder = LabelEncoder()
    encoded_tempos = tempo_encoder.fit_transform(tempos)

    # Print label ranges for debugging
    print("Note labels range:", np.min(note_encoder.classes_), np.max(note_encoder.classes_))
    print("Velocity labels range:", np.min(velocity_encoder.classes_), np.max(velocity_encoder.classes_))
    print("Duration labels range:", np.min(duration_encoder.classes_), np.max(duration_encoder.classes_))
    print("Tempo labels range:", np.min(tempo_encoder.classes_), np.max(tempo_encoder.classes_))

    # Prepare sequences for features and targets
    sequences = []
    y_notes = []
    y_velocities = []
    y_durations = []
    y_tempos = []

    for i in range(len(encoded_notes) - seq_length):
        seq_notes = encoded_notes[i:i + seq_length]
        seq_velocities = encoded_velocities[i:i + seq_length]
        seq_durations = encoded_durations[i:i + seq_length]
        seq_tempos = encoded_tempos[i:i + seq_length]

        # Combine the sequences into a feature set
        sequence = np.array(list(zip(seq_notes, seq_velocities, seq_durations, seq_tempos)))
        
        # Prepare the targets for each sequence (the next timestep's value)
        target_note = encoded_notes[i + seq_length]
        target_velocity = encoded_velocities[i + seq_length]
        target_duration = encoded_durations[i + seq_length]
        target_tempo = encoded_tempos[i + seq_length]

        # Append to the lists
        sequences.append(sequence)
        y_notes.append(target_note)
        y_velocities.append(target_velocity)
        y_durations.append(target_duration)
        y_tempos.append(target_tempo)

    # Convert sequences and targets to numpy arrays
    X = np.array(sequences)
    y_note = np.array(y_notes)
    y_velocity = np.array(y_velocities)
    y_duration = np.array(y_durations)
    y_tempo = np.array(y_tempos)

    # Print the shape of X before reshaping
    print(f"X shape before reshaping: {X.shape}")

    # Ensure that the shape of X matches the expected dimensions
    if len(X.shape) == 3:
        X = X.reshape((X.shape[0], X.shape[1], 4))  # Reshaping to (samples, seq_length, features)
    else:
        print(f"Warning: Unexpected shape for X: {X.shape}")
    
    # Return only the necessary variables (6 values)
    y = {
        'note': y_note,
        'velocity': y_velocity,
        'duration': y_duration,
        'tempo': y_tempo
    }

    return X, y, note_encoder, velocity_encoder, duration_encoder, tempo_encoder


# Function to create MIDI from a sequence
def create_midi_from_sequence(sequence, output_path):
    """
    Create a MIDI file from the generated sequence with better error handling.
    """
    #midi = MidiFile()
    midi = MidiFile(type=1, ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)
    
    current_time = 0
    
    for note, velocity, duration, tempo in sequence:
        try:
            # Ensure values are within MIDI bounds
            note_val = int(np.clip(note, 0, 127))
            velocity_val = int(np.clip(velocity, 1, 127))
            duration_ticks = int(max(1, duration * 480))  # Ensure positive duration
            
            # Note on event
            track.append(Message('note_on', note=note_val, 
                               velocity=velocity_val, 
                               time=current_time))
            
            # Note off event
            track.append(Message('note_off', note=note_val, 
                               velocity=0, 
                               time=duration_ticks))
            
            current_time = 0
            
        except Exception as e:
            print(f"Warning: Skipped invalid note: {e}")
            continue
    
    midi.save(output_path)
    print(f"Saved generated MIDI to {output_path}")

# second midi sequence
def create_midi_from_sequence2(sequence, output_path, ticks_per_beat=480):
    """
    Create a MIDI file from the generated sequence with proper timing and event handling.
    
    Args:
        sequence: List of tuples (note, velocity, duration, tempo)
        output_path: Path to save the MIDI file
        ticks_per_beat: MIDI ticks per quarter note (default=480)
    """
    midi = MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Add tempo track
    tempo_track = MidiTrack()
    midi.tracks.append(tempo_track)
    
    # Set initial tempo (microseconds per beat)
    initial_tempo = 500000  # 120 BPM
    tempo_track.append(Message('set_tempo', tempo=initial_tempo, time=0))
    
    current_time = 0
    previous_time = 0
    
    try:
        for i, (note, velocity, duration, tempo) in enumerate(sequence):
            try:
                # Validate and clip values
                note_val = int(np.clip(note, 0, 127))
                velocity_val = int(np.clip(velocity, 1, 127))
                
                # Convert duration to ticks, ensure it's positive and reasonable
                duration_seconds = float(duration)
                if duration_seconds <= 0:
                    duration_seconds = 0.25  # Default to quarter note
                duration_ticks = int(duration_seconds * ticks_per_beat)
                duration_ticks = max(1, min(duration_ticks, ticks_per_beat * 4))  # Limit to 4 beats
                
                # Calculate delta time (time since last event)
                delta_time = duration_ticks if i == 0 else duration_ticks - previous_time
                delta_time = max(0, delta_time)  # Ensure non-negative
                
                # Note on event
                track.append(Message('note_on', 
                                   note=note_val,
                                   velocity=velocity_val,
                                   time=delta_time))
                
                # Note off event
                track.append(Message('note_off',
                                   note=note_val,
                                   velocity=0,
                                   time=duration_ticks))
                
                # Update timing
                previous_time = duration_ticks
                current_time += duration_ticks
                
                # Update tempo if needed (optional)
                if tempo != initial_tempo:
                    new_tempo = int(np.clip(tempo, 20, 300))  # Clip BPM to reasonable range
                    tempo_microseconds = int(60000000 / new_tempo)  # Convert BPM to microseconds per beat
                    tempo_track.append(Message('set_tempo', 
                                            tempo=tempo_microseconds,
                                            time=current_time))
                
            except Exception as e:
                print(f"Warning: Error processing note at position {i}: {e}")
                continue
        
        # Ensure the file ends properly
        track.append(Message('end_of_track', time=0))
        tempo_track.append(Message('end_of_track', time=0))
        
        # Save the MIDI file
        midi.save(output_path)
        print(f"Successfully saved MIDI file to {output_path}")
        
    except Exception as e:
        print(f"Error creating MIDI file: {e}")
        raise

# Building the model
def build_model(input_shape, num_classes):
    input_layer = Input(shape=(input_shape[0], input_shape[1]))
# Hidden layers
    hidden_layer = Dense(128, activation='relu')(input_layer)
    hidden_layer = Dense(64, activation='relu')(hidden_layer)

    # Output layers
    output_note = Dense(48, activation='softmax', name='note_output')(hidden_layer)
    output_velocity = Dense(77, activation='softmax', name='velocity_output')(hidden_layer)
    output_duration = Dense(1, activation='sigmoid', name='duration_output')(hidden_layer)
    output_tempo = Dense(1, activation='sigmoid', name='tempo_output')(hidden_layer)

    # Define the model
    model = Model(inputs=input_layer, outputs=[output_note, output_velocity, output_duration, output_tempo])

    # Compile the model
    model.compile(optimizer=Adam(),
                  loss={'note_output': 'sparse_categorical_crossentropy',
                        'velocity_output': 'mse',
                        'duration_output': 'mse',
                        'tempo_output': 'mse'},
                  metrics={'note_output': 'accuracy',
                           'velocity_output': 'mae',
                           'duration_output': 'mae',
                           'tempo_output': 'mae'})

    return model

# def train_model(X, y):
    # Build model
    # model = build_model(input_shape=(X.shape[1], X.shape[2]))

    # Train the model
    history = model.fit(X, 
                        [y['note'], y['velocity'], y['duration'], y['tempo']],
                        epochs=1,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[EarlyStopping(patience=5)])

    return model, history

# preprocess data
def preprocess_data(X, y_note, y_velocity, y_duration, y_tempo):
    """
    Preprocess the labels using LabelEncoder and scale the features (if needed).
    """
    note_encoder = LabelEncoder()
    y_note_encoded = note_encoder.fit_transform(y_note)

    velocity_encoder = LabelEncoder()
    y_velocity_encoded = velocity_encoder.fit_transform(y_velocity)

    duration_encoder = LabelEncoder()
    y_duration_encoded = duration_encoder.fit_transform(y_duration)

    tempo_encoder = LabelEncoder()
    y_tempo_encoded = tempo_encoder.fit_transform(y_tempo)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1]))

    X_scaled = X_scaled.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Fit the encoders and transform the labels
    # y_note = note_encoder.fit_transform(y_note)
    # y_velocity = velocity_encoder.fit_transform(y_velocity)
    # y_duration = duration_encoder.fit_transform(y_duration)
    # y_tempo = tempo_encoder.fit_transform(y_tempo)
    
    # Check the classes and labels for debugging
    print("Unique note labels:", note_encoder.classes_)
    print("Unique velocity labels:", velocity_encoder.classes_)
    print("Unique duration labels:", duration_encoder.classes_)
    print("Unique tempo labels:", tempo_encoder.classes_)
    
    # Debugging: check the range of label values
    print(f"Note labels range: {min(y_note)} {max(y_note)}")
    print(f"Velocity labels range: {min(y_velocity)} {max(y_velocity)}")
    print(f"Duration labels range: {min(y_duration)} {max(y_duration)}")
    print(f"Tempo labels range: {min(y_tempo)} {max(y_tempo)}")
    
    # Scale the features (optional, depending on the model needs)
    # Example: Use MinMaxScaler or StandardScaler if required for your data
    # X = MinMaxScaler().fit_transform(X)  # Uncomment if feature scaling is needed

    y_note = np.clip(y_note, 0, 19)
    y_velocity = np.clip(y_velocity, 0, 61)
    
    return X_scaled, y_note_encoded, y_velocity_encoded, y_duration_encoded, y_tempo_encoded, note_encoder, velocity_encoder, duration_encoder, tempo_encoder

# Training the model using other methods as helper methods
def train_model(X, y_note, y_velocity, y_duration, y_tempo, 
                output_size_note, output_size_velocity, output_size_duration, output_size_tempo):

    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], 1, X.shape[1]))  # Add the timesteps dimension (1 timestep in this case)
    
    input_layer = Input(shape=(X.shape[1], X.shape[2]))  # (timesteps, features)
    
    # Define the model structure
    x = LSTM(128)(input_layer)
    x = Dense(64, activation='relu')(x)
    
    # Output layers for each label
    output_note = Dense(output_size_note, activation='softmax', name='note_output')(x)
    output_velocity = Dense(output_size_velocity, activation='softmax', name='velocity_output')(x)
    output_duration = Dense(output_size_duration, activation='softmax', name='duration_output')(x)
    output_tempo = Dense(output_size_tempo, activation='softmax', name='tempo_output')(x)

    model = Model(inputs=input_layer, outputs=[output_note, output_velocity, output_duration, output_tempo])

    # Define the metrics for each output
    metrics = {
        'note_output': ['accuracy'],
        'velocity_output': ['accuracy'],
        'duration_output': ['accuracy'],
        'tempo_output': ['accuracy']
    }
    
    model.compile(optimizer='adam',
                  loss={
                      'note_output': 'SparseCategoricalCrossentropy',
                      #'velocity_output': 'MeanSquaredLogarithmicError',
                      'velocity_output': 'SparseCategoricalCrossentropy',
                      'duration_output': 'SparseCategoricalCrossentropy',
                      'tempo_output': 'SparseCategoricalCrossentropy'
                  },
                  metrics=metrics)

#    model.compile(optimizer='adam',
#                  loss='Dice',
#                  metrics=metrics)

# Train the model
    X = np.array(X)
    history = model.fit(X, 
                        {'note_output': y_note, 
                         'velocity_output': y_velocity,
                         'duration_output': y_duration,
                         'tempo_output': y_tempo},
                        epochs=1, 
                        batch_size=32)
    
    return model, history

# generate music track after training
def generate_music_track(model, note_encoder, velocity_encoder, duration_encoder, tempo_encoder, X, sequence_length=100, num_notes=100):
    """
    Generate a music track using the trained model with valid label handling.
    """
    # Get valid labels for each encoder
    valid_notes = note_encoder.classes_
    valid_velocities = velocity_encoder.classes_
    valid_durations = duration_encoder.classes_
    valid_tempos = tempo_encoder.classes_
    
    # Get random starting sequence from X
    start_index = random.randint(0, len(X) - 1)
    current_sequence = X[start_index:start_index + 1]
    
    if len(current_sequence.shape) == 2:
        current_sequence = current_sequence.reshape(1, 1, current_sequence.shape[1])

    generated_sequence = []
    
    for _ in range(num_notes):
        # Make prediction using the current sequence
        predictions = model.predict(current_sequence, verbose=0)
        
        # Get predictions and ensure they're within valid ranges
        predicted_note = predictions[0][0]
        predicted_velocity = predictions[1][0]
        predicted_duration = predictions[2][0]
        predicted_tempo = predictions[3][0]
        
        # Get the most likely valid predictions
        note_idx = np.argmax(predicted_note)
        velocity_idx = np.argmax(predicted_velocity)
        duration_idx = np.argmax(predicted_duration)
        tempo_idx = np.argmax(predicted_tempo)
        
        # Clip indices to valid ranges
        note_idx = min(note_idx, len(valid_notes) - 1)
        velocity_idx = min(velocity_idx, len(valid_velocities) - 1)
        duration_idx = min(duration_idx, len(valid_durations) - 1)
        tempo_idx = min(tempo_idx, len(valid_tempos) - 1)
        
        # Get the actual values
        #note = valid_notes[note_idx]
        #velocity = valid_velocities[velocity_idx]
        #duration = valid_durations[duration_idx]
        #tempo = valid_tempos[tempo_idx]
        note = note_encoder.inverse_transform([note_idx])[0]
        velocity = velocity_encoder.inverse_transform([velocity_idx])[0]
        duration = duration_encoder.inverse_transform([duration_idx])[0]
        tempo = tempo_encoder.inverse_transform([tempo_idx])[0]
        
        # Add to generated sequence
        #duration = 750000
        #tempo = 0
        generated_sequence.append((note, velocity, duration, tempo))
        
        # Prepare input for next prediction
        new_input = np.array([[note, velocity, duration, tempo]])
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1] = new_input[0][:2]
    
    return generated_sequence

# generate sequence to put into music track
def generate_sequence(model, note_encoder, velocity_encoder, duration_encoder, X, tempo_encoder, seq_length=50):
    # Randomly choose a starting point from the data (you can modify this for more control)
    start_index = random.randint(0, X.shape[0] - 1)
    input_sequence = X[start_index:start_index+1]  # Get one sequence to generate from

    # Generate an empty sequence list for notes, velocities, durations, and tempos
    generated_notes = []
    generated_velocities = []
    generated_durations = []
    generated_tempos = []

    # Generate sequence
    for _ in range(seq_length):
        # Predict the next note, velocity, duration, and tempo
        pred_note, pred_velocity, pred_duration, pred_tempo = model.predict(input_sequence)

        # Choose the most likely prediction for each part
        next_note = note_encoder.inverse_transform([np.argmax(pred_note[0])])[0]
        next_velocity = velocity_encoder.inverse_transform([np.argmax(pred_velocity[0])])[0]
        next_duration = duration_encoder.inverse_transform([np.argmax(pred_duration[0])])[0]
        next_tempo = tempo_encoder.inverse_transform([np.argmax(pred_tempo[0])])[0]

        # Append the generated values to the lists
        generated_notes.append(next_note)
        generated_velocities.append(next_velocity)
        generated_durations.append(next_duration)
        generated_tempos.append(next_tempo)

        # Update the input sequence for the next prediction
        input_sequence = np.roll(input_sequence, -1, axis=1)
        input_sequence[0, -1, 0] = next_note  # Update last note in sequence
        input_sequence[0, -1, 1] = next_velocity  # Update last velocity in sequence
        input_sequence[0, -1, 2] = next_duration  # Update last duration in sequence
        input_sequence[0, -1, 3] = next_tempo  # Update last tempo in sequence

    return generated_notes, generated_velocities, generated_durations, generated_tempos

# Save the generated tracks
def save_generated_tracks(model, note_encoder, velocity_encoder, duration_encoder, tempo_encoder, output_folder, X, num_tracks=10):
    # Generate and save 'num_tracks' generated MIDI tracks
    for track_num in range(num_tracks):
        # Generate a sequence of notes using the model
        generated_notes, generated_velocities, generated_durations, generated_tempos = generate_sequence(
            model, note_encoder, velocity_encoder, duration_encoder, X, tempo_encoder, seq_length=50)

        # Create a MIDI file from the generated sequence
        output_path = os.path.join(output_folder, f"generated_track_{track_num+1}.mid")
        create_midi_from_sequence(generated_notes, output_path)  # Create the MIDI and save it

    print(f"Generated {num_tracks} tracks and saved to {output_folder}")

# Main function
def main():
    # convert_all_midi_files(midi_folder, csv_folder)
    midi_to_csv(midi_folder + "\\alb_esp1.mid", csv_folder + "\\1.csv")
    # Load the data
    X, y_note, y_velocity, y_duration, y_tempo = load_data(csv_folder)
    
    # Print raw data statistics before encoding
    print("\nRaw Data Statistics:")
    print(f"y_note range: min={np.min(y_note)}, max={np.max(y_note)}, unique values={len(np.unique(y_note))}")
    print(f"y_velocity range: min={np.min(y_velocity)}, max={np.max(y_velocity)}, unique values={len(np.unique(y_velocity))}")
    print(f"y_duration range: min={np.min(y_duration)}, max={np.max(y_duration)}, unique values={len(np.unique(y_duration))}")
    print(f"y_tempo range: min={np.min(y_tempo)}, max={np.max(y_tempo)}, unique values={len(np.unique(y_tempo))}")
    
    # Initialize label encoders
    note_encoder = LabelEncoder()
    velocity_encoder = LabelEncoder()
    duration_encoder = LabelEncoder()
    tempo_encoder = LabelEncoder()
    
    # Fit and transform the labels
    y_note_encode = note_encoder.fit_transform(y_note)
    y_velocity_encode = velocity_encoder.fit_transform(y_velocity)
    y_duration_encode = duration_encoder.fit_transform(y_duration)
    y_tempo_encode = tempo_encoder.fit_transform(y_tempo)
    
    # Get the number of classes for each output
    n_classes_note = len(np.unique(y_note_encode))
    n_classes_velocity = len(np.unique(y_velocity_encode))
    n_classes_duration = len(np.unique(y_duration_encode))
    n_classes_tempo = len(np.unique(y_tempo_encode))
    
    print("\nNumber of classes after encoding:")
    print(f"Notes: {n_classes_note}")
    print(f"Velocity: {n_classes_velocity}")
    print(f"Duration: {n_classes_duration}")
    print(f"Tempo: {n_classes_tempo}")
    
    # Reshape input if needed
    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    # Define model architecture
    input_layer = Input(shape=(X.shape[1], X.shape[2]))
    x = LSTM(128)(input_layer)
    x = Dense(64, activation='relu')(x)
    
    # Output layers with correct number of classes
    output_note = Dense(n_classes_note, activation='softmax', name='note_output')(x)
    output_velocity = Dense(n_classes_velocity, activation='softmax', name='velocity_output')(x)
    output_duration = Dense(n_classes_duration, activation='softmax', name='duration_output')(x)
    output_tempo = Dense(n_classes_tempo, activation='softmax', name='tempo_output')(x)
    
    # Create model
    model = Model(inputs=input_layer, 
                 outputs=[output_note, output_velocity, output_duration, output_tempo])
    #             outputs=[output_note, output_velocity])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss={
            'note_output': 'sparse_categorical_crossentropy',
            'velocity_output': 'sparse_categorical_crossentropy',
            'duration_output': 'sparse_categorical_crossentropy',
            'tempo_output': 'sparse_categorical_crossentropy'
        },
        metrics={
            'note_output': ['accuracy'],
            'velocity_output': ['accuracy'],
            'duration_output': ['accuracy'],
            'tempo_output': ['accuracy']
        }
    )
    
    # Print model summary
    model.summary()
    
    # Verify encoded values are within range
    print("\nVerifying encoded ranges:")
    print(f"Note range: {np.min(y_note_encode)} to {np.max(y_note_encode)}")
    print(f"Velocity range: {np.min(y_velocity_encode)} to {np.max(y_velocity_encode)}")
    print(f"Duration range: {np.min(y_duration_encode)} to {np.max(y_duration_encode)}")
    print(f"Tempo range: {np.min(y_tempo_encode)} to {np.max(y_tempo_encode)}")
    
    # Train model
    history = model.fit(
        X, 
        {
            'note_output': y_note_encode,
            'velocity_output': y_velocity_encode,
            'duration_output': y_duration_encode,
            'tempo_output': y_tempo_encode
        },
        epochs=1,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    print("\nModel trained successfully!")
    
    for i in range(10):
        generated_sequence = generate_music_track(model, note_encoder, velocity_encoder, duration_encoder, tempo_encoder, X, sequence_length=100, num_notes=100)
        output_midi_path = os.path.join(output_folder, f"generated_music_{i+1}.mid")
        create_midi_from_sequence(generated_sequence, output_midi_path)

        # Convert MIDI to WAV
        output_wav_path = os.path.join(output_folder, f"generated_music_{i+1}.wav")
        convert_midi_to_wav(output_midi_path, output_wav_path)

    print("All tracks generated successfully!")

# Function to convert MIDI to WAV
def convert_midi_to_wav(midi_path, wav_path):
    soundfont_path = "C:\\Users\\rpuri\\OneDrive - Eastside Preparatory School\\Documents\\12th Grade\\Independent Project\\SGM-v2.01-NicePianosGuitarsBass-V1.2.sf2"
    if not os.path.exists(soundfont_path):
        raise FileNotFoundError(f"SoundFont file not found at {soundfont_path}")
    
    fs = FluidSynth(soundfont_path)  # Specify your SoundFont path
    print (f"path1 {midi_path}")
    print (f"path2 {wav_path}")
    fs.midi_to_audio(midi_path, wav_path)

if __name__ == "__main__":
    main()