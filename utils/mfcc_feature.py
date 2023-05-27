import librosa

def extract(audio, sample_rate):
    mfcc_feature = []
    for y in audio:
        mfcc_feature.append(librosa.feature.mfcc(y=y, sr=sample_rate))
    
    return mfcc_feature