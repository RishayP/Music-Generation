from flask import Flask, render_template, jsonify, send_from_directory
import os
import random

app = Flask(__name__)

# Path to the folder where generated music files are stored
music_folder = "C:\\Users\\rpuri\\OneDrive - Eastside Preparatory School\\Documents\\12th Grade\\Independent Project\\output\\"

# List all generated music files (WAV files)
music_files = [f for f in os.listdir(music_folder) if f.endswith('.wav')]

# Keep track of the current track index
current_track_index = -1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/next-track')
def next_track():
    global current_track_index
    current_track_index = (current_track_index + 1) % len(music_files)  # Cycle through the tracks
    selected_music = music_files[current_track_index]
    return jsonify({'track': selected_music})

@app.route('/play/<track>')
def play_track(track):
    # Return the track from the music folder
    track_path = os.path.join(music_folder, track)
    if os.path.exists(track_path):
        return send_from_directory(music_folder, track)
    else:
        return jsonify({'error': 'Track not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
