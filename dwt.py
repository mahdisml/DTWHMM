import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import librosa


class DWT:

    def __init__(self, using_property):
        self.using_property = using_property

    def get_array(self, file):
        if not self.using_property:
            y, sr = librosa.load(file, sr=8000)
            return y
        else:
            y, sr = librosa.load(file)
            hop_length = 512
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc)
            beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)
            chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
            beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)
            beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
            return beat_features.ravel()

    def against(self, file1, file2):
        if file1 != file2:
            x = self.get_array(file1)
            y = self.get_array(file2)
            distance, path = fastdtw(x, y, dist=euclidean)
        else:
            distance = 0
        return distance
