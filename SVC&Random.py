import os
import random

import numpy as np
import librosa
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
)
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_features(mfcc, chroma, spec_contrast, tonnetz, audio, sr):
    plt.figure(figsize=(15, 15))

    plt.subplot(5, 1, 1)
    librosa.display.specshow(mfcc, x_axis='time', cmap='viridis')
    plt.colorbar()
    plt.title('MFCC')

    plt.subplot(5, 1, 2)
    librosa.display.specshow(spec_contrast, x_axis='time', cmap='viridis')
    plt.colorbar()
    plt.title('spec_contrast')

    plt.subplot(5, 1, 3)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', cmap='viridis')
    plt.colorbar()
    plt.title('Chroma')

    plt.subplot(5, 1, 4)
    librosa.display.specshow(tonnetz, x_axis='time', cmap='viridis')
    plt.colorbar()
    plt.title('Tonnetz')

    plt.subplot(5, 1, 5)
    librosa.display.waveshow(audio, sr=sr, x_axis="time")
    plt.colorbar()
    plt.show()

    plt.tight_layout()
    plt.show()



def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

    features = np.concatenate((
        mfcc.mean(axis=1), mfcc.std(axis=1), np.median(mfcc, axis=1),
        spec_contrast.mean(axis=1), spec_contrast.std(axis=1), np.median(spec_contrast, axis=1),
        chroma.mean(axis=1), chroma.std(axis=1), np.median(chroma, axis=1),
        tonnetz.mean(axis=1), tonnetz.std(axis=1), np.median(tonnetz, axis=1),
        mel_spectrogram.mean(axis=1), mel_spectrogram.std(axis=1), np.median(mel_spectrogram, axis=1)
    ))
    #plot_features(mfcc, chroma, spec_contrast, tonnetz, audio, sr)

    mfcc_features = [f'MFCC_{i}' for i in range(mfcc.shape[0])]
    spec_contrast_features = [f'SpectralContrast_{i}' for i in range(spec_contrast.shape[0])]
    chroma_features = [f'Chroma_{i}' for i in range(chroma.shape[0])]
    tonnetz_features = [f'Tonnetz_{i}' for i in range(tonnetz.shape[0])]
    mel_spectrogram_features = [f'MelSpectrogram_{i}' for i in range(mel_spectrogram.shape[0])]
    feature_groups = [mfcc_features, spec_contrast_features, chroma_features, tonnetz_features,
                      mel_spectrogram_features]
    return features, feature_groups


def augment_audio(audio, sr, pitch_shift_range=(-2, 2), time_stretch_range=(0.8, 1.2),
                  noise_factor_range=(0.005, 0.015)):
    # Pitch shifting
    pitch_shift = random.randint(pitch_shift_range[0], pitch_shift_range[1])
    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)

    # Time stretching
    time_stretch = random.uniform(time_stretch_range[0], time_stretch_range[1])
    audio = librosa.effects.time_stretch(audio, rate=time_stretch)

    # Adding noise
    noise_factor = random.uniform(noise_factor_range[0], noise_factor_range[1])
    noise = np.random.randn(len(audio))
    audio = audio + noise_factor * noise
    audio = np.clip(audio, -1, 1)

    return audio


def load_dataset(speech_folder, music_folder, max_files=None, augment_data=True, augmentations_per_file=2):
    logger.info('Loading the dataset from the speech and music folders.')
    labels, feature_list = [], []

    for folder, label in [(speech_folder, 0), (music_folder, 1)]:
        folder_name = "Speech" if label == 0 else "Music"
        logger.info(f'Processing {folder_name} folder.')
        files_loaded = 0

        for file in tqdm(os.listdir(folder), desc=f'{folder_name} folder'):
            if max_files is not None and files_loaded >= max_files:
                break

            file_path = os.path.join(folder, file)
            if file_path.endswith('.wav'):
                logger.info(f'Loading file {file_path}')
                audio, sr = librosa.load(file_path, sr=None)
                features, feature_groups = extract_features(audio, sr)
                feature_list.append(features)
                labels.append(label)
                files_loaded += 1
                logger.info(f'File {file_path} loaded successfully.')
                if augment_data:
                    for _ in range(augmentations_per_file):
                        augmented_audio = augment_audio(audio, sr)
                        features, feature_groups = extract_features(augmented_audio, sr)
                        feature_list.append(features)
                        labels.append(label)

    logger.info('Dataset loaded successfully.')
    return np.array(feature_list), np.array(labels), feature_groups


def split_dataset(X, y, test_size=0.2):
    logging.info('Splitting the dataset into training and testing sets.')
    data = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    logging.info('Dataset split successfully.')
    return data


def train_classifier(X_train, y_train):
    logger.info('Training the classifier.')
    classifiers = [
        ('Random Forest', RandomForestClassifier()),
        ('SVM', SVC(probability=True))
    ]

    best_classifier, best_score = None, 0
    for name, classifier in classifiers:
        logger.info(f'Training {name} classifier.')
        # TODO read about this
        grid = GridSearchCV(classifier, param_grid={}, cv=5)
        grid.fit(X_train, y_train)
        score = grid.best_score_
        if score > best_score:
            best_classifier, best_score = grid.best_estimator_, score
        logger.info(f'{name} classifier score: {score:.2f}')

    logger.info(f'Best classifier: {best_classifier.__class__.__name__} with score: {best_score:.2f}')
    logger.info('Classifier trained successfully.')
    return best_classifier


def evaluate_classifier(classifier, X_test, y_test):
    logging.info('Evaluating the classifier on the testing set.')
    y_pred = classifier.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    }
    logging.info('Classifier evaluation metrics computed.')
    return metrics


def detect_change_points(audio_path, classifier, window_size=3, hop_length=1, smoothing_window=5):
    logger.info('Loading the input audio file.')
    audio, sr = librosa.load(audio_path, sr=None)
    logger.info(f'Audio file {audio_path} loaded successfully.')

    change_points = []
    num_windows = int((len(audio) - window_size * sr) / (hop_length * sr))

    logger.info('Extracting features from the input audio file and detecting change points.')
    preds = []
    for i in tqdm(range(num_windows), desc='Detecting change points'):
        start = i * hop_length * sr
        end = start + window_size * sr
        window = audio[start:end]
        features, feature_groups = extract_features(window, sr)
        feature = features.reshape(1, -1)
        pred = classifier.predict(feature)[0]
        preds.append(pred)

    # Apply smoothing to the predictions
    smoothed_preds = uniform_filter1d(preds, size=smoothing_window, mode='nearest', output=np.int32)

    for i, (pred, prev_pred) in enumerate(zip(smoothed_preds[1:], smoothed_preds[:-1])):
        if pred != prev_pred:
            change_point_seconds = (i * hop_length * sr) / sr
            change_points.append(change_point_seconds)
            change_type = "Speech to Music" if prev_pred == 0 else "Music to Speech"
            logger.info(f'Change point detected at {change_point_seconds:.2f} seconds: {change_type}')

    logger.info('Change points detected successfully.')
    return change_points


def plot_true_vs_predicted_labels(y_true, y_pred):
    plt.figure()
    plt.plot(y_true, label='True Labels', linestyle='-', marker='o')
    plt.plot(y_pred, label='Predicted Labels', linestyle='-', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.legend()
    plt.title('True vs. Predicted Labels')
    plt.show()


def plot_roc_curve(classifier, X_test, y_test):
    y_prob = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def plot_feature_importances(importances, feature_names):
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=['Speech', 'Music']).plot(cmap=plt.cm.Blues,
                                                                                     values_format='.2f')
    plt.show()


speech_folder = "archive/speech_wav"
music_folder = "archive/music_wav"
input_audio = "merged/merged_audio_21s.wav"

X, y, feature_groups = load_dataset(speech_folder, music_folder, 10)
X_train, X_test, y_train, y_test = split_dataset(X, y)
classifier = train_classifier(X_train, y_train)
metrics = evaluate_classifier(classifier, X_test, y_test)
change_points = detect_change_points(input_audio, classifier)

logging.info('Classifier evaluation metrics:')
for name, value in metrics.items():
    logging.info(f'{name}: {value:.2f}')

logging.info(f'Change points: {change_points}')

y_pred = classifier.predict(X_test)
plot_true_vs_predicted_labels(y_test, y_pred)
plot_roc_curve(classifier, X_test, y_test)
plot_confusion_matrix(y_test, y_pred)

if hasattr(classifier, 'feature_importances_'):
    feature_names = []
    for feature_group in feature_groups:
        feature_names.extend([f'{f}_mean' for f in feature_group])
        feature_names.extend([f'{f}_std' for f in feature_group])
        feature_names.extend([f'{f}_median' for f in feature_group])
    plot_feature_importances(classifier.feature_importances_, feature_names)
