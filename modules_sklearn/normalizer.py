from modules_sklearn import *
import librosa
import scipy

def normalize(data, norm=None, norm_value=np.inf, trim=False):
    """
    Arguments:
        data: (dim, seq_len)
    """
    # norm_value = 2
    # trim=True
    if trim:
        data = [d[:,3:-3] for d in data]
    if norm == 'l2':
        data = [librosa.util.normalize(d, norm=2) for d in data]
    if norm == 'norm':
        data = [librosa.util.normalize(d, norm=norm_value) for d in data]
    elif norm == 'lognorm': # only melspectrogram
        data = [librosa.power_to_db(d) for d in data] #, ref=np.mean
        data = [librosa.util.normalize(d, norm=norm_value) for d in data]
    elif norm == 'cmvn': # only mfcc
        data = [cmvn(d) for d in data]
        data = [librosa.util.normalize(d, norm=norm_value) for d in data]
    elif norm == 'cens': # only chroma
        data = [cens(d) for d in data]
    elif norm == 'nlmn':
        data = [nonlocalmean_norm(d) for d in data]
    elif norm == 'perceptual': # only cqt
        data = [perceptualize(d) for d in data]

    return data


def cmvn(vec, variance_normalization=True):
    """ This function is aimed to perform global cepstral mean and
        variance normalization (CMVN) on input feature vector "vec".
        The code assumes that there is one observation per row.
    Args:
        vec (array): input feature matrix
            (size:(dim, seq_len))
        variance_normalization (bool): If the variance
            normilization should be performed or not.
    Return:
          array: The mean(or mean+variance) normalized feature vector.
    """
    vec = vec.transpose(1,0)

    eps = 2**-30
    rows, cols = vec.shape

    # Mean calculation
    norm = np.mean(vec, axis=0)
    norm_vec = np.tile(norm, (rows, 1))

    # Mean subtraction
    mean_subtracted = vec - norm_vec

    # Variance normalization
    if variance_normalization:
        stdev = np.std(mean_subtracted, axis=0)
        stdev_vec = np.tile(stdev, (rows, 1))
        output = mean_subtracted / (stdev_vec + eps)
    else:
        output = mean_subtracted

    return output.transpose(1,0)


def perceptualize(C):
    fmin=librosa.note_to_hz('C2')
    freqs = librosa.cqt_frequencies(
        C.shape[0], fmin=fmin)
    perceptual_C = librosa.perceptual_weighting(
        C**2, freqs, ref=np.max)
    return perceptual_C


def nonlocalmean_norm(S, to_db=False):
    """
    Do not apply this to Chroma and MFCC
    """
    if to_db:
        S = librosa.power_to_db(S) #, ref=np.max
    rec = librosa.segment.recurrence_matrix(S, mode='affinity',
                                    metric='cosine', sparse=True)
    S_nlm = librosa.decompose.nn_filter(S, rec=rec,
                                     aggregate=np.average)
    return S_nlm


def cens(chroma):
    win_len_smooth=41
    smoothing_window='hann'

    # L1-Normalization
    chroma = librosa.util.normalize(chroma, norm=1, axis=0)

    # Quantize amplitudes
    QUANT_STEPS = [0.4, 0.2, 0.1, 0.05]
    QUANT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]

    chroma_quant = np.zeros_like(chroma)

    for cur_quant_step_idx, cur_quant_step in enumerate(QUANT_STEPS):
        chroma_quant += (chroma > cur_quant_step) * QUANT_WEIGHTS[cur_quant_step_idx]

    if win_len_smooth:
        # Apply temporal smoothing
        win = librosa.filters.get_window(
            smoothing_window, win_len_smooth + 2, fftbins=False)
        win /= np.sum(win)
        win = np.atleast_2d(win)

        cens = scipy.signal.convolve2d(
            chroma_quant, win, mode="same", boundary="fill")
    else:
        cens = chroma_quant

    # L2-Normalization
    return librosa.util.normalize(cens, norm=2, axis=0)
