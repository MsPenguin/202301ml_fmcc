import librosa
import numpy as np


def extract(audio, sample_rate):

    mfcc_feature = []
    
    for y in audio:
        mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)
        mfcc = np.mean(mfcc.T, axis=0)
        mfcc_feature.append(mfcc)

    mfcc_feature = np.array(mfcc_feature)

    return mfcc_feature
