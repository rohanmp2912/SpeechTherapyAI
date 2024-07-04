import parselmouth
import numpy as np
import matplotlib.pyplot as plt

# Path to the audio file
audio_path = 'SpeechTherapyAI/ML_MODEL/star_stare.mp3'

# Load the audio file
snd = parselmouth.Sound(audio_path)

# Extract formants
formant = snd.to_formant_burg()

# Extract time and formant frequencies
times = formant.ts()
f1 = [formant.get_value_at_time(1, t) for t in times]
f2 = [formant.get_value_at_time(2, t) for t in times]

# Remove None values (where formants could not be estimated)
f1 = [x if x else 0 for x in f1]
f2 = [x if x else 0 for x in f2]

# Convert to numpy arrays for easier manipulation
f1 = np.array(f1)
f2 = np.array(f2)

# Remove zeros (unvoiced frames)
f1 = f1[f1 > 0]
f2 = f2[f2 > 0]

# Compute the average formants
average_f1 = np.mean(f1)
average_f2 = np.mean(f2)

# Print the average formants
print(f"Average F1: {average_f1:.2f} Hz")
print(f"Average F2: {average_f2:.2f} Hz")

# Plot the formants
plt.figure(figsize=(10, 6))
plt.plot(times, f1, label='F1')
plt.plot(times, f2, label='F2')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Formants F1 and F2 over time')
plt.legend()

# Save the image
image_path = 'SpeechTherapyAI/ML_MODEL/F1_F2.png'
plt.savefig(image_path)

# Show the plot
plt.show()