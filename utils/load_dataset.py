import os
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

def load(dataset_path):
    audio_male = []  # 남성 음성 데이터
    audio_female = []  # 여성 음성 데이터

    # Iterate through the dataset directory
    for root, dirs, files in os.walk(dataset_path):

        for file in files:

            # Assuming each file is labeled with the class in its filename
            # Extract the label from the filename
            label = file.split("_")[0]
            # Full path to the audio file
            pcm_file_path = os.path.join(root, file)

            # Load the audio file
            audio, sr = sf.read(pcm_file_path, channels=1, samplerate=SAMPLE_RATE,
                            subtype='PCM_' + str(SAMPLE_WIDTH * 8))

            # Store the audio sample and its corresponding label
            if label[0] == 'M':
                audio_male.append(audio)
            else:   # label[0] == 'F'
                audio_female.append(audio)

    return audio_male, audio_female