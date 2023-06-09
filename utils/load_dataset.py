import os
import soundfile as sf
import numpy as np


def load_trainset(dataset_path, sample_rate, sample_width):

    audio_male = []  # 남성 음성 데이터
    audio_female = []  # 여성 음성 데이터

    # Iterate through the dataset directory
    for root, dirs, files in os.walk(dataset_path):

        for file in files:

            # Extract the label from the filename
            label = file.split("_")[0]

            # Full path to the audio file
            pcm_file_path = os.path.join(root, file)

            # Load the audio file
            audio, sr = sf.read(pcm_file_path, frames=16384, start=800, fill_value=0, samplerate=sample_rate, channels=1,
                                subtype='PCM_' + str(sample_width * 8))

            # Store the audio sample and its corresponding label
            if label[0] == 'M':
                audio_male.append(audio)

            else:   # label[0] == 'F'
                audio_female.append(audio)

    audio_male = np.array(audio_male)
    audio_female = np.array(audio_female)
    
    return audio_male, audio_female


def load_testset(dataset_path, sample_rate, sample_width):

    audio_test = []  # 음성 데이터

    for root, dirs, files in os.walk(dataset_path):

        for file in files:
            pcm_file_path = os.path.join(root, file)

            # Load the audio file
            audio, sr = sf.read(pcm_file_path, frames=16384, start=800, fill_value=0, samplerate=sample_rate, channels=1,
                                subtype='PCM_' + str(sample_width * 8))

            audio_test.append(audio)

    audio_test = np.array(audio_test)
    
    return audio_test
