from ..nn.signal import get_window_freq_response
import numpy as np
from . import DEFAULT_SAMPLE_RATE
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def visualize_waveform(filename="", audio_dir="./", 
        y="", sr=DEFAULT_SAMPLE_RATE, title="", 
        zoom_xlim=[0.05,0.1]):
    if filename:
        import soundfile as sf
        y, sr = sf.read(audio_dir + filename)
    if not title:
        title = filename
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(15, 3))
    ax.set(title='Full waveform: ' + title)
    ax.plot(y)
    ax2.set(title='Sample view: ' + title, 
            xlim=np.multiply(zoom_xlim, sr))
    ax2.plot(y, marker='.')

def visualize_window(window, window_name="", 
                     f_xlim=None, f_ylim=None, 
                     f_xhighlight=-20, fs=2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,4))
    axes[0].plot(window)
    axes[0].set_title(f"{window_name} window")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Sample")

    response = get_window_freq_response(window)
    freq = np.fft.fftfreq(len(response), d=1/fs) 
    if not f_xlim:
        f_xlim = [0,0.5]
    if not f_ylim:
        f_ylim = [-110, 10]
    axes[1].set_xlim(f_xlim)
    axes[1].set_ylim(f_ylim)
    axes[1].plot(freq, response)
    axes[1].set_title("Frequency response of the window")
    axes[1].set_ylabel("Normalized magnitude [dB]")
    axes[1].set_xlabel("Normalized frequency [cycles per sample]")
    axes[1].axhline(f_xhighlight, color='red')