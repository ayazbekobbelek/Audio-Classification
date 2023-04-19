import os
import random
from pydub import AudioSegment


def merge_files(speech_folder, music_folder, output_folder, output_filename):
    # Select random files from the speech and music folders
    speech_file = random.choice(os.listdir(speech_folder))
    music_file = random.choice(os.listdir(music_folder))

    # Load audio files
    speech_audio = AudioSegment.from_wav(os.path.join(speech_folder, speech_file))
    music_audio = AudioSegment.from_wav(os.path.join(music_folder, music_file))

    # Choose a random change point in seconds
    change_point = random.randint(0, len(speech_audio) // 1000)

    # Adjust volumes
    speech_audio = speech_audio - 3  # Slightly reduce the volume of speech
    music_audio = music_audio - 3  # Slightly reduce the volume of music

    # Calculate the crossfade duration based on the audio segments' lengths
    crossfade_duration = min(2000, change_point * 1000, (len(speech_audio) - change_point * 1000))

    # Apply crossfade for a smooth transition
    speech_audio_fade_out = speech_audio[change_point * 1000: (change_point * 1000) + crossfade_duration].fade_out(
        crossfade_duration)
    music_audio_fade_in = music_audio[:crossfade_duration].fade_in(crossfade_duration)

    # Merge the audio segments
    merged_audio = speech_audio[:change_point * 1000].append(music_audio_fade_in, crossfade=crossfade_duration).append(
        music_audio[crossfade_duration:])

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save merged audio file
    merged_audio.export(os.path.join(output_folder, f"{output_filename}_{change_point}s.wav"), format="wav")


# Usage
speech_folder = "archive/speech_wav"
music_folder = "archive/music_wav"
output_folder = "merged"
output_filename = "merged_audio"
merge_files(speech_folder, music_folder, output_folder, output_filename)
