from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import numpy as np

# Function to compare two embeddings
def is_same_speaker(embed1, embed2, threshold=0.8832492232322693):
    similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
    return similarity, similarity > threshold

# Initialize the encoder
encoder = VoiceEncoder()

# Paths to first audio file
audio_file1 = Path("misc_samples/benpass7.wav")

#Path to second audio file
audio_file2 = Path("misc_samples/fakepass1.wav")

# Preprocess the audio files
wav1 = preprocess_wav(audio_file1)
wav2 = preprocess_wav(audio_file2)

# Embed the audio files
embed1 = encoder.embed_utterance(wav1)
embed2 = encoder.embed_utterance(wav2)

# Compare the embeddings
similarity, same_speaker = is_same_speaker(embed1, embed2)

# Print the results
print(f"Similarity score: {similarity}")
print("Voice Verification Passed!" if same_speaker else "Voice Verification Failed")
