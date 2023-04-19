import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def music_segmentation(hop_length=512, threshold=0.5):
    y, sr = librosa.load(librosa.ex("nutcracker"), duration=30)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length, win_length=400)

    # Compute the derivative of the tempogram
    tempogram_diff = np.diff(tempogram, axis=1)

    spectrogram = librosa.stft(y)
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))

    # Find segments by analyzing the tempogram difference
    segments = []
    start = 0
    for i, frame in enumerate(tempogram_diff.T):
        if np.linalg.norm(frame) > threshold:
            segments.append((beat_times[start], beat_times[i]))
            start = i

    segments.append((beat_times[start], beat_times[-1]))
    return segments, tempogram, spectrogram_db

def plot_tempogram(tempogram, sr, hop_length, segments):
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo')
    plt.title('Tempogram with Segmentation')

    # Add vertical lines for the segments
    for segment in segments:
        plt.axvline(x=segment[0], color='red', linestyle='--', alpha=0.8)
        plt.axvline(x=segment[1], color='red', linestyle='--', alpha=0.8)

    plt.xlabel('Time (s)')
    plt.ylabel('Tempo (BPM)')
    plt.show()

def plot_waveform(y, sr, segments):
    plt.figure(figsize=(12, 6))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform with Segmentation')

    # Add vertical lines for the segments
    for segment in segments:
        plt.axvline(x=segment[0], color='red', linestyle='--', alpha=0.8)
        plt.axvline(x=segment[1], color='red', linestyle='--', alpha=0.8)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def display_spectrogram(spectrogram_db, sr):
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='log', sr=sr)
    plt.title('Spectrogram')
    plt.show()

if __name__ == "__main__":
    audio_file = "StarWars60.wav"
    segments, tempogram, spectogram = music_segmentation()
    print("Segments:", segments)

    y, sr = librosa.load(librosa.ex("nutcracker"), duration=30)
    hop_length = 512
    plot_tempogram(tempogram, sr, hop_length, segments)
    plot_waveform(y, sr, segments)
    display_spectrogram(spectogram, sr)
