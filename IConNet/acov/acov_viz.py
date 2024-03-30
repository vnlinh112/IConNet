import soundfile as sf
import numpy as np
from scipy import fft, signal
import matplotlib.pyplot as plt
from .audio_viz import feature_librosa

import warnings
warnings.filterwarnings('ignore')

def compute_autocovariance(data, n_win=None, win_pos=None, autocovariance=None, windowed_data=None):
    if autocovariance is None:
        x = np.linspace(0.5, n_win-0.5, num=n_win)
        # windowing function
        windowing_fn = np.sin(np.pi*x / n_win)**2 # sine-window
        if not win_pos:
            win_pos = np.random.randint(len(data) - n_win)
        windowed_data = data[win_pos:(win_pos + n_win)]
        windowed_data = windowed_data * windowing_fn / np.max(np.abs(windowed_data)) # normalize
        n_win = len(windowed_data)
        spectrum = fft.rfft(windowed_data, n=2*n_win)
        autocov_raw = fft.irfft(np.abs(spectrum**2))
        autocovariance = np.concatenate((autocov_raw[n_win:], 
                                         autocov_raw[0:n_win]))
    filtered_data = signal.convolve(data, autocovariance,
                                     mode='same') / sum(autocovariance)

    return autocovariance, filtered_data, win_pos, windowed_data

def show_autocovariance(sig, win_data, win_acov, filtered, win_pos=0, zoom=False):
    fig, (ax_orig, ax_win, ax_acov, ax_filt) = plt.subplots(4, 1, sharex=True)
    n_sig, n_win = len(sig), len(win_data)
    start_pos, end_pos = 0, n_sig
    if zoom: 
        n1 = win_pos - len(win_data)
        start_pos = np.max((start_pos, n1))
        n2 = win_pos + len(win_acov)
        end_pos = np.min((end_pos, n2))
        ticks = ax_orig.get_xticks()
        tick_labels = np.linspace(start_pos, end_pos, 5)
        ax_orig.set_xticks(tick_labels, tick_labels)
        ax_orig.set_xlim(start_pos, end_pos)
        
    ax_orig.plot(sig)
    ax_orig.set_title('Original pulse')
    ax_orig.margins(0, 0.1)
    
    n = len(sig) - len(win_data) - win_pos
    win_data = np.concatenate((np.zeros(win_pos), win_data, np.zeros(n)))
    ax_win.plot(win_data)
    ax_win.set_title(f'Filter impulse response: windowed data at position {win_pos}-th')
    ax_win.margins(0, 0.1)

    n2 = len(sig) - len(win_acov) - win_pos
    win_acov = np.concatenate((np.zeros(win_pos), win_acov, np.zeros(n2)))
    ax_acov.plot(win_acov)
    ax_acov.set_title(f'Filter impulse response: autocovariance of windowed data with length={n_win}')
    ax_acov.margins(0, 0.1)
    
    ax_filt.plot(filtered)
    ax_filt.set_title('Filtered signal')
    ax_filt.margins(0, 0.1)
    
    fig.tight_layout()
    fig.show()

def test(filename="1001_IEO_ANG_MD.wav"):
    win_length = 512*4
    data, sr = sf.read(filename)
    acov, filtered, windowpos, windata = compute_autocovariance(data, win_length, win_pos=15980)
    show_autocovariance(data, windata, acov, filtered, windowpos, zoom=True)
    feature_librosa(y=data, title=filename, sr=sr)
    feature_librosa(y=filtered, title=filename, sr=sr)
    feature_librosa(y=acov, title=filename, sr=sr)

    # acov_ravdess6, filtered_ravdess6, windowpos_ravdess6, windata_ravdess6 = compute_autocovariance(
    #     data_ravdess6, autocovariance = acov_ravdess4)
    # feature_librosa(y=filtered_ravdess6, title=fname_ravdess6, sr=audio_ravdess6.sr)