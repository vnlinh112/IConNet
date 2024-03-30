import librosa
from librosa import display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from .audio import AudioLibrosa
import warnings
warnings.filterwarnings('ignore')

sr = 16000
file_dir = "../data/"
audio_dir = f"{file_dir}audio_samples/"

def play_and_visualize(filename="", y="", sr=sr, title=""):
    print(title)
    if filename:
        y, sr = librosa.load(filename, sr=sr) # read and resampling to sr
    
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(15, 3))
    ax.set(title='Full waveform')
    display.waveshow(y, sr=sr, ax=ax, color="blue")
    
    ax2.set(title='Sample view', xlim=[0.05, 0.1])
    display.waveshow(y, sr=sr, ax=ax2, marker='.', color="blue")
    plt.show()
    return ipd.Audio(y, rate=sr)

notes = {}

def add_note(note):
    n = librosa.note_to_hz(note)
    signal = librosa.tone(n, sr=sr, length=sr)
    notes[note] = signal
    return play_and_visualize(y=signal, sr=sr, title=f"{note} {n:0.2f} Hz signal")

def show_feature(audio, title="", expand=False):
    print(title)
    
    n = 8
    if expand:
        fig, ax = plt.subplots(ncols=2, nrows=n//2, figsize=(8*2, 5*n//2))
    else:
        fig, ax = plt.subplots(ncols=n//2, nrows=2, figsize=(6*n//2, 5*2))
        
    axi = ax.ravel()    
    
    i = 0
    img = display.waveshow(audio.y, sr=sr, ax=axi[i], color="blue")
    axi[i].set(title='raw waveform signal')
    
    i += 1
    img = display.waveshow(audio.y, sr=sr, ax=axi[i], marker='.', color="blue")
    axi[i].set(title='Sample view', xlim=[0.05, 0.1])
    
    i += 1
    audio.show_spectrogram(audio.mfcc, convert_db=False, 
                           y_axis='mel', ax=axi[i], fig=fig, title='MFCC', colorbar=expand) 

    i += 1
    audio.show_spectrogram(audio.chromagram, convert_db=False, 
                           y_axis='chroma', ax=axi[i], fig=fig, title='Chroma CENS', colorbar=expand)
    
    i += 1
    audio.show_spectrogram(audio.spectrogram, y_axis='linear', ax=axi[i], fig=fig, 
                           title='Linear-frequency power spectrogram', colorbar=expand)
    
    i += 1
    audio.show_spectrogram(audio.spectrogram, y_axis='log', ax=axi[i], fig=fig, 
                           title='Log-frequency power spectrogram & F0', colorbar=expand)
    times = librosa.times_like(audio.f0, sr=audio.sr)
    axi[i].plot(times, audio.f0, label='F0', color='cyan', linewidth=2)
    axi[i].legend(loc='upper right')
    
    i += 1
    mel_db = librosa.power_to_db(audio.melspectrogram, ref=np.max)
    audio.show_spectrogram(mel_db, convert_db=False, y_axis='mel', ax=axi[i], fig=fig, 
                           title='Melspectrogram power spectrum', colorbar=expand)  
    
    i += 1
    audio.show_spectrogram(audio.cqt, y_axis='cqt_note', ax=axi[i], fig=fig, 
                           title='Constant-Q power spectrum', colorbar=expand) 

def feature_librosa(filename="", y="", sr=sr, title="", file_dir=audio_dir, expand=False):
    audio = AudioLibrosa(filename,y,sr,title,file_dir)
    sr = audio.sr
    audio.extract_features()
    show_feature(audio, title, expand)
    return audio.player

