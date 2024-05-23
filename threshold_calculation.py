from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Function to compute similarity between two embeddings
def compute_similarity(embed1, embed2):
    return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))

# Initialize the encoder
encoder = VoiceEncoder()

# Paths to your validation audio files
speaker1_files = list(Path("validation_data/speaker_1").glob("*.wav"))
speaker2_files = list(Path("validation_data/speaker_2").glob("*.wav"))

# Create pairs of same-speaker and different-speaker audio files
same_speaker_pairs = [(f1, f2) for i, f1 in enumerate(speaker1_files) for f2 in speaker1_files[i+1:]] + \
                     [(f1, f2) for i, f1 in enumerate(speaker2_files) for f2 in speaker2_files[i+1:]]
different_speaker_pairs = [(f1, f2) for f1 in speaker1_files for f2 in speaker2_files]

# Function to process pairs and compute similarity scores
def process_pairs(pairs):
    similarities = []
    for file1, file2 in pairs:
        wav1 = preprocess_wav(Path(file1))
        wav2 = preprocess_wav(Path(file2))
        embed1 = encoder.embed_utterance(wav1)
        embed2 = encoder.embed_utterance(wav2)
        similarity = compute_similarity(embed1, embed2)
        similarities.append(similarity)
    return similarities

# Compute similarity scores for same and different speaker pairs
same_speaker_scores = process_pairs(same_speaker_pairs)
different_speaker_scores = process_pairs(different_speaker_pairs)

# Combine scores and labels
all_scores = np.array(same_speaker_scores + different_speaker_scores)
labels = np.array([1] * len(same_speaker_scores) + [0] * len(different_speaker_scores))

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(labels, all_scores)
roc_auc = auc(fpr, tpr)

# Find the EER
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

print(f"EER threshold: {eer_threshold}")
print(f"EER: {eer}")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
