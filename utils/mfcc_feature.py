import librosa
import numpy as np


def extract(audio, sample_rate):

    mfcc_feature = np.array([])
    
    for y in audio:
        mfcc_feature = np.append(
            mfcc_feature, librosa.feature.mfcc(y=y, sr=sample_rate))

    return mfcc_feature
