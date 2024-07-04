from pydub import AudioSegment
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

# Print the current working directory
print("Current working directory:", os.getcwd())

# Path to the MP3 file
mp3_audio_path = 'SpeechTherapyAI/ML_MODEL/star_stare.mp3'


# Load the MP3 file
y, sr = librosa.load(mp3_audio_path)

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Display the MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()

# Save the plot as an image file
image_path = 'SpeechTherapyAI/ML_MODEL/mfcc_plot.png'
plt.savefig(image_path)
print(f"MFCC plot saved as {image_path}")

# Show the plot
plt.show()

# Print MFCCs
print("MFCCs shape:", mfccs.shape)
print(mfccs)