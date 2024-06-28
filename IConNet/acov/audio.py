import librosa
from librosa import display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal, Iterable

sr = 16000
file_dir = "../data/"
audio_dir = f"{file_dir}audio_samples/"

fmin=librosa.note_to_hz('C2') #recommended for pyin
fmax=librosa.note_to_hz('C7')

class Audio:
    def __init__(
            self, 
            y: Optional[Iterable[float]]=None, 
            sr: int=sr, 
            title: str="",
            win_length: int=2048,
            hop_length: Optional[int]=None,
            n_fft: Optional[int]=None,
            n_mels: int=128,
            n_mfcc: int=128,
            features_player: bool=True,
            features_f0: bool=True,
            features_fft: bool=True,
            features_cqt: bool=True):
        self.win_length = win_length
        if hop_length is None:
            self.hop_length = self.win_length // 4
        else:
            self.hop_length = hop_length
        if n_fft is None:
            self.n_fft = self.win_length
        else:
            self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.features_player = features_player
        self.features_f0 = features_f0
        self.features_fft = features_fft
        self.features_cqt = features_cqt
        
        self.y = y
        self.sr = sr
        self.title = title

        if y is not None and len(y) > 0:
            self.init_time_vector()

    def load(self, filename="", file_dir=audio_dir, sr=sr):
        if filename:
            y, sr = librosa.load(file_dir + filename, sr=sr)
            if y.ndim > 1: # convert stereo to mono
                y = y[:,1]
        self.y = y
        self.sr = sr
        if self.title is None or len(self.title) == 0:
            self.title = filename  
        self.init_time_vector() 

    def init_time_vector(self):
        self.n_samples = len(self.y)
        self.duration = self.n_samples / self.sr
        self.time_vector = np.linspace(
            0, self.duration, self.n_samples) 

    def get_player(self):
        self.player = ipd.Audio(self.y, rate=self.sr)
        return self.player
    
    def get_f0(self):
        self.f0, voiced_flag, voiced_probs = librosa.pyin(
            self.y, sr=self.sr,
            fmin=fmin, fmax=fmax,
            win_length=self.win_length
            )
        return self.f0
    
    def get_chromagram(self):
        self.chromagram = librosa.feature.chroma_cens(
            C=self.cqt, sr=self.sr,
            bins_per_octave=12)
        return self.chromagram
    
    def extract_features(self):
        if self.features_player:
            self.get_player()
        if self.features_f0:
            self.get_f0()
        if self.features_fft:
            self.get_spectrogram()
            self.get_melspectrogram()
            self.get_mfcc()
        if self.features_cqt:
            self.get_cqt()
            self.get_chromagram()
        
    def show_spectrogram(
            self, 
            spectrogram,
            convert_db: Optional[Literal['power', 'amplitude']]='power', 
            y_axis='log', ax=None, fig=None, 
            title="", colorbar=False):
        if not ax:
            fig, ax = plt.subplots(ncols=1, figsize=(6, 4))
        if convert_db=='power':
            spectrogram = librosa.power_to_db(
                spectrogram, ref=np.max)
        elif convert_db=='amplitude':
            spectrogram = librosa.amplitude_to_db(
                spectrogram, ref=np.max)
        img = display.specshow(
            spectrogram, sr=self.sr, 
            y_axis=y_axis, x_axis='time', ax=ax)
        ax.set_title(title)
        if colorbar:
            fig.colorbar(img, ax=ax, format="%+2.0f")
        return img
    
    def get_spectrogram(self):
        raise NotImplementedError()

    def get_melspectrogram(self):
        raise NotImplementedError()

    def get_mfcc(self):
        raise NotImplementedError()

    def get_cqt(self):
        raise NotImplementedError()


class AudioLibrosa(Audio):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_spectrogram(self):
        self.spectrogram = np.abs(librosa.stft(
            y=self.y, n_fft=self.n_fft, 
            win_length=self.win_length, 
            hop_length=self.hop_length))
        return self.spectrogram

    def get_melspectrogram(self):
        self.melspectrogram = librosa.feature.melspectrogram(
            S=self.spectrogram**2, sr=self.sr, n_mels=self.n_mels)
        return self.melspectrogram

    def get_mfcc(self):
        S = librosa.power_to_db(self.melspectrogram)
        self.mfcc = librosa.feature.mfcc(
            S=S, sr=self.sr, n_mfcc=self.n_mfcc)
        return self.mfcc

    def get_cqt(self):
        self.cqt = librosa.hybrid_cqt(
            y=self.y, sr=self.sr, bins_per_octave=12) #hybrid_cqt
        return self.cqt
    