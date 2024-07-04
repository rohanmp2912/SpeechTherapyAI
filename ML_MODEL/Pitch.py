import librosa
import numpy as np

# Path to the audio file
audio_path = 'SpeechTherapyAI/ML_MODEL/star_stare.mp3'

# Load the audio file
y, sr = librosa.load(audio_path)

# Extract the pitch (fundamental frequency) using librosa's piptrack function
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

# Find the maximum magnitude index for each frame
pitches = [pitches[:, i][magnitudes[:, i].argmax()] if magnitudes[:, i].any() else 0 for i in range(magnitudes.shape[1])]

# Convert the list to a numpy array
pitches = np.array(pitches)

# Remove zeros (unvoiced frames)
pitches = pitches[pitches > 0]

# Compute the average pitch
average_pitch = np.mean(pitches)

# Print the average pitch
print(f"Average Pitch: {average_pitch:.2f} Hz")