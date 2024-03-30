import librosa
from librosa import display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt

sr = 16000
file_dir = "../data/"
audio_dir = f"{file_dir}audio_samples/"

fmin=librosa.note_to_hz('C2') #recommended for pyin
fmax=librosa.note_to_hz('C7')

class Audio:
    def __init__(self, filename="", y=sr, sr=sr, title="", file_dir=audio_dir):
        self.win_length = 2048
        self.hop_length = self.win_length // 4
        self.n_mfcc = 128
        
        if filename:
            y, sr = librosa.load(file_dir + filename, sr=sr)
            if y.ndim > 1: # convert stereo to mono
                y = y[:,1]
        self.y = y
        self.sr = sr
        self.n_samples = len(y)
        self.duration = self.n_samples / self.sr
        self.time_vector = np.linspace(0, self.duration, self.n_samples) 

        if title:
            self.title = title
        else:
            self.title = filename   
    
    def get_player(self):
        self.player = ipd.Audio(self.y, rate=self.sr)
        return self.player
    
    def get_f0(self):
        self.f0, voiced_flag, voiced_probs = librosa.pyin(self.y, sr=self.sr,
                                                          fmin=fmin, fmax=fmax)
        return self.f0
    
    def get_chromagram(self):
        self.chromagram = librosa.feature.chroma_cens(C=self.cqt, sr=self.sr,
                                                     bins_per_octave=12)
        return self.chromagram
    
    def extract_features(self):
        self.get_player()
        self.get_f0()
        self.get_spectrogram()
        self.get_melspectrogram()
        self.get_mfcc()
        self.get_cqt()
        self.get_chromagram()
        print("extracted: f0, spectrogram, melspectrogram, mfcc, cqt, chromagram")
        
    def show_spectrogram(self, spectrogram, convert_db=True, 
                         y_axis='log', ax=None, fig=None, title="", colorbar=False):
        if not ax:
            fig, ax = plt.subplots(ncols=1, figsize=(6, 4))
        if convert_db:
            spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
        img = display.specshow(spectrogram, sr=self.sr, y_axis=y_axis, x_axis='time', ax=ax)
        ax.set_title(title)
        if colorbar:
            fig.colorbar(img, ax=ax, format="%+2.0f")
        return img

class AudioLibrosa(Audio):
    def __init__(self, filename="", y=sr, sr="", title="", file_dir=audio_dir):
        super().__init__(filename, y, sr, title, file_dir)

    def get_spectrogram(self):
        self.spectrogram = np.abs(librosa.stft(y=self.y))
        return self.spectrogram

    def get_melspectrogram(self):
        self.melspectrogram = librosa.feature.melspectrogram(S=self.spectrogram**2, sr=self.sr)
        return self.melspectrogram

    def get_mfcc(self):
        S = librosa.power_to_db(self.melspectrogram)
        self.mfcc = librosa.feature.mfcc(S=S, sr=self.sr, n_mfcc=self.n_mfcc)
        return self.mfcc

    def get_cqt(self):
        self.cqt = librosa.hybrid_cqt(y=self.y, sr=self.sr, bins_per_octave=12) #hybrid_cqt
        return self.cqt
    