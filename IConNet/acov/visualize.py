import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import umap
import torch 
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial

import librosa
from librosa import display
import matplotlib.pyplot as plt
import seaborn as sns
import umap
sns.set_theme(style="dark")

from .zero_crossing import zero_crossings
from .audio import AudioLibrosa

from matplotlib import ticker
sr = 16000

def visualize_speech_codebook(
        waves, colors="blue", title="", 
        expand=False, n=16):
    print(title)
    if expand:
        ncols = 4
        fig, ax = plt.subplots(
            ncols=ncols, nrows=n//ncols, 
            figsize=(8*ncols, 5*n//ncols))
    else:
        ncols = 8
        fig, ax = plt.subplots(
            ncols=ncols, nrows=n//ncols, 
            figsize=(6*ncols, 5*n//ncols))
    axi = ax.ravel()    
    i = 0
    for x, y in enumerate(waves[:n//2]):
        if type(colors) == str:
            color = colors
        else:
            color = colors[x]
        audio = AudioLibrosa(
            y=y, sr=sr, title=f"Embedding {x}",
            win_length=256,
            n_mels=64, n_mfcc=13,
            features_f0=True,
            features_fft=True,
            features_player=False, 
            features_cqt=False)
        audio.extract_features()
        
        img = display.waveshow(
            audio.y, sr=sr, ax=axi[i], color=color)
        axi[i].set(title=audio.title)
        axi[i].set(xlabel=None)
        axi[i].xaxis.set_major_formatter(
            ticker.StrMethodFormatter("{x:.2f}"))
    
        i += 1
        audio.show_spectrogram(
            audio.melspectrogram, 
            convert_db='power', y_axis='mel', 
            ax=axi[i], fig=fig, 
            title='Melspectrogram & F0', 
            colorbar=expand)  
        times = librosa.times_like(audio.f0, sr=audio.sr)
        axi[i].plot(
            times, audio.f0, 
            label='F0', color='cyan', linewidth=4)
        axi[i].legend(loc='upper right')
        axi[i].set(xlabel=None)
        axi[i].xaxis.set_major_formatter(
            ticker.StrMethodFormatter("{x:.2f}"))
        i += 1
    axi[i-1].set(xlabel="Time")

def get_embedding_color(    
    y: Tensor,
    n_fft: int = 1024,
    stride: int = 256
) -> Tensor:
    """Compute the zero-crossing score of the codebook embedding.

    Returns
    -------
    zcr : np.ndarray [shape=(..., 1, t)]
        ``zcr[..., 0, i]`` is the zero crossing score in frame ``i`` 
        with score in range [0,1]
    """
    crossings = zero_crossings(y, dtype=torch.float)
    zcrate = crossings.mean(dim=-1)
    nonzero_mean = torch.where(
        crossings>0, crossings, 0.0).mean(dim=-1)
    score = F.sigmoid(1 - 8*zcrate + 2*nonzero_mean)
    color = torch.stack(
        [torch.full_like(score, 0.5), score, score], 
        dim=-1).numpy()
    return color, score

def visualize_embedding_umap(
    data, colors, edgecolors="black",
    n_neighbors=15, min_dist=0.1, 
    n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data)
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(
            u[:,0], range(len(u)), 
            c=colors, edgecolors= edgecolors)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(
            u[:,0], u[:,1], 
            c=colors, edgecolors=edgecolors)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            u[:,0], u[:,1], u[:,2], 
            s=100, c=colors, edgecolors=edgecolors)
    plt.title(title, fontsize=18)

def visualize_training_curves(
        test_accuracy, train_cls_loss,
        train_recon_loss, train_perplexity,
        train_smooth_params = (301, 7),
        test_smooth_params = (3, 2)
):
    train_smooth_fn = partial(
        savgol_filter, 
        window_length=train_smooth_params[0], 
        polyorder=train_smooth_params[1])
    test_smooth_fn = partial(
        savgol_filter, 
        window_length=test_smooth_params[0], 
        polyorder=test_smooth_params[1])
    try:
        test_accuracy = test_smooth_fn(test_accuracy)
        train_cls_loss = train_smooth_fn(train_cls_loss)
        train_recon_loss = train_smooth_fn(train_recon_loss)
        train_perplexity = train_smooth_fn(train_perplexity)
    except Exception:
        pass
    
    f = plt.figure(figsize=(12,3))
    ax = f.add_subplot(1,4,1)
    ax.plot(test_accuracy)
    ax.set_title('Test accuracy')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1,4,2)
    ax.plot(train_cls_loss)
    ax.set_title('Train cls loss')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1,4,3)
    ax.plot(train_recon_loss)
    ax.set_title('SNR-based loss')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1,4,4)
    ax.plot(train_perplexity)
    ax.set_title('Avg. codebook usage')
    ax.set_xlabel('iteration')