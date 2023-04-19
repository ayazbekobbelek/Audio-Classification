import numpy as np
import librosa
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_audio(audio_file, sample_rate=22050, hop_length=512):
    y, sr = librosa.load(audio_file, sr=sample_rate)
    return y, sr


def extract_features(y, sr, hop_length=512, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack((mfcc, mfcc_delta, mfcc_delta2))
    return features.T


def train_model(X_train, y_train, X_test, y_test, C=1, kernel='rbf', gamma='scale'):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    return clf, scaler, accuracy_score(y_test, y_pred)


def predict_segments(clf, scaler, features, threshold=0.5):
    scaled_features = scaler.transform(features)
    probs = clf.predict_proba(scaled_features)
    preds = (probs[:, 1] > threshold).astype(int)
    return preds


def identify_changes(preds):
    change_points = []
    for i in range(1, len(preds)):
        if preds[i] != preds[i - 1]:
            change_points.append(i)
    return change_points


# Replace these with appropriate paths and labels
logging.info("Loading audio files and labels from folders...")
speech_folder = "musan/music/fma"
music_folder = "musan/speech/us-gov"

speech_files = [os.path.join(speech_folder, f) for f in os.listdir(speech_folder) if f.endswith('.wav')][:25]
music_files = [os.path.join(music_folder, f) for f in os.listdir(music_folder) if f.endswith('.wav')][:25]

audio_files = speech_files + music_files
labels = [0] * len(speech_files) + [1] * len(music_files)
logging.info("Loaded %d speech files and %d music files.", len(speech_files), len(music_files))

logging.info("Preprocessing and extracting features from audio files...")
all_features = []
for idx, audio_file in enumerate(audio_files):
    y, sr = preprocess_audio(audio_file)
    features = extract_features(y, sr)
    all_features.append(features)
    progress_percentage = (idx + 1) / len(audio_files) * 100
    logging.info("Processed file %d/%d (%.2f%%): %s", idx + 1, len(audio_files), progress_percentage, audio_file)

X = np.vstack(all_features)
y = np.concatenate([np.full(features.shape[0], label) for features, label in zip(all_features, labels)])

# Split data into train and test sets
logging.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
logging.info("Training the model...")
clf, scaler, accuracy = train_model(X_train, y_train, X_test, y_test)
logging.info("Model accuracy: %.2f", accuracy)

# Predict segments in a new audio file
logging.info("Predicting segments in a new audio file...")
new_audio_file = "mergedFiles/music-fma-0054.wav_6.wav"
y_new, sr_new = preprocess_audio(new_audio_file)
features_new = extract_features(y_new, sr_new)
preds_new = predict_segments(clf, scaler, features_new)

# Identify changes between speech and music
logging.info("Identifying changes between speech and music...")
change_points = identify_changes(preds_new)


def change_points_to_seconds(change_points, sr, hop_length):
    return [point * hop_length / sr for point in change_points]


# Convert change points to seconds
change_points_seconds = change_points_to_seconds(change_points, sr_new, 512)

logging.info("Change points in seconds:")
for point in change_points_seconds:
    logging.info("%.2f", point)
