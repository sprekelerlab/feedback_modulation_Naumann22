"""
Functions to generate input data to model (sources and contexts).
"""

# import packages
import numpy as np
import torch
from scipy.signal import sawtooth


def gen_sines(dur=2, fs=8000, freqs_rel=[1, 1.8], base_freq=100):
    """
    Generate sine waves as source signals.
    Output is a numpy float32 array of size (dur*fs) x (len(freqs_rel)).
    dur:        duration of signal in seconds
    fs:         sampling frequency in Hz
    freqs_rel:  list of frequencies for sine waves (relative to a baseline frequency)
    base_freq:  baseline frequency for list of frequencies (in Hz)
    """

    t = np.arange(0, dur, 1/fs)

    s = []
    for ff in freqs_rel:
        s.append(np.sin(base_freq*ff * t * 2*np.pi))

    return np.array(s).astype(np.float32).T


def gen_chords(dur=2, fs=8000, freqs_rel=[1, 1.25, 1.5, 2.1], base_freq=100):
    """
    Generate "chords" (compositions of sines) as source signals.
    Output is a numpy float32 array of size (dur*fs) x 2, normalised to range (-1, 1).
    dur:        duration of signal in seconds
    fs:         sampling frequency in Hz
    freqs_rel:  list of frequencies for chords; first two for first chord, second two for second chord (rel. to base)
    base_freq:  baseline frequency for list of frequencies (in Hz)
    """

    f11 = freqs_rel[0]*base_freq
    f12 = freqs_rel[1]*base_freq
    f21 = freqs_rel[2]*base_freq
    f22 = freqs_rel[3]*base_freq

    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    t = np.arange(0, dur, 1/fs)
    chord1 = np.sin(f11 * t * 2 * np.pi) + np.sin(f12 * t * 2 * np.pi)
    chord2 = np.sin(f21 * t * 2 * np.pi) + np.sin(f22 * t * 2 * np.pi)

    return np.array([chord1 / max(abs(chord1)), chord2 / max(abs(chord2))]).astype(np.float32).T


def gen_ica_style(dur=2, fs=8000, freqs_rel=[1.4, 0.8, 1.2], base_freq=100):
    """
    Generate signals often used as example in ICA: one sawtooth, one box signal and one sine wave
    Output is a numpy float32 array of size (dur*fs) x 2, normalised to range (-1, 1).
    dur:        duration of signal in seconds
    fs:         sampling frequency in Hz
    freqs_rel:  list of frequencies for the three signals (relative to baseline frequency)
    base_freq:  baseline frequency for list of frequencies (in Hz)
    """

    t = np.arange(0, dur, 1/fs)

    # extract frequencies (only 3 possible)
    f1, f2, f3 = freqs_rel

    # generate sawtooth, box signal and sine
    st = sawtooth(2 * np.pi * base_freq * f1 * t)
    box = np.sign(np.sin(2 * np.pi * base_freq * f2 * t))
    sine = np.sin(2 * np.pi * base_freq * f3 * t)

    return np.array([sine, box, st]).astype(np.float32).T


def gen_mixing_mats(n_mixing=10, mat_size=(2, 2), det_threshold=0.2, mix_type='random_norm'):
    """
    Generate mixing matrices that determine the mixing of the sources (=contexts).
    n_mixing:      number of mixing matrices to generate
    mat_size:      shape of mixing matrices as tuple
    det_threshold: threshold of determinant below which matrices are discarded (too close to singularity)
    mix_type:      type of mixing matrix; options: 'scale', 'scale_both', 'interpolate', 'interpolate_both',
                                                   'constant', 'random', 'random_norm'

    """

    if mat_size != (2, 2) and 'random' not in mix_type:
        print('WARNING: The requested mix type is not implemented for shapes other than (2, 2)!')

    # empty array for mixing matrices and their inverses
    mixing_mats = []
    mixing_inv = []

    # scale the first signal (or both if 'scale_both')
    if 'scale' in mix_type:
        while len(mixing_mats) < n_mixing:
            alpha = np.random.uniform(0.3, 3)
            beta = np.random.uniform(1, 2) if 'both' in mix_type else 1
            a = np.array([[alpha, 0], [0, beta]]).astype(np.float32)
            if np.abs(np.linalg.det(a)) > det_threshold:
                mixing_mats.append(torch.from_numpy(a))
                mixing_inv.append(np.linalg.pinv(a).flatten())

    # mixing are an interpolation between the two signals, if not 'both', then the second mixed signal = second source
    elif 'interpolate' in mix_type:
        while len(mixing_mats) < n_mixing:
            alpha = np.random.uniform(0.2, 1)
            beta = np.random.uniform(0.2, 1) if 'both' in mix_type else 1
            a = np.array([[alpha, 1-alpha], [1-beta, beta]]).astype(np.float32)
            # a = a/np.linalg.norm(a)
            if np.abs(np.linalg.det(a)) > det_threshold:
                mixing_mats.append(torch.from_numpy(a))
                mixing_inv.append(np.linalg.pinv(a).flatten())

    # for testing: mixing matrix is always the same (hard coded below)
    elif 'constant' in mix_type:

        a = np.array([[0.8, 0.2], [0.5, 0.5]]).astype(np.float32)
        ainv = np.linalg.pinv(a).flatten()
        for _ in range(n_mixing):
            mixing_mats.append(a)
            mixing_inv.append(ainv)

    # values of mixing matrix are randomly uniformly sampled between 0 and 1
    # if 'norm', then the rows are normalised to sum 1
    elif 'random':  # arbitrary positive invertible mixing
        while len(mixing_mats) < n_mixing:
            a = np.random.uniform(0, 1, size=mat_size).astype(np.float32)
            if 'norm' in mix_type:
                a = a/np.sum(a, axis=1)[:, np.newaxis]
            if mat_size[0] != mat_size[1]:  # matrix not square, use pseudo inverse
                mixing_mats.append(torch.from_numpy(a))
                mixing_inv.append(np.linalg.pinv(a).flatten())
            elif np.abs(np.linalg.det(a)) > det_threshold:  # make sure matrix is not too close to singularity
                mixing_mats.append(torch.from_numpy(a))
                mixing_inv.append(np.linalg.pinv(a).flatten())

    return mixing_mats, mixing_inv


def gen_train_test_data(n_train, n_test, mix_type='random_norm', batch_size=32, mat_size=(2, 2), det_threshold=0.2):
    """
    Generate mixing matrices for training and testing of model in pytorch. Training data is made up of batches.
    Output are two tuples of pytorch dataloaders: zipped mixing matrices and their inverses for training and testing.
    n_train:     number of mixing matrices for training
    n_test:      number of mixing matrices for testing
    mix_type:    type of mixing matrix generation
    batch_size:  number of mixings in a batch
    mat_size:    shape of mixing matrices as tuple
    """

    # training data
    mixing_mats_train, mixing_inv_train = gen_mixing_mats(n_mixing=n_train, mix_type=mix_type, mat_size=mat_size,
                                                          det_threshold=det_threshold)
    trainloader_A = torch.utils.data.DataLoader(mixing_mats_train, batch_size=batch_size)
    trainloader_Ainv = torch.utils.data.DataLoader(mixing_inv_train, batch_size=batch_size)

    # testing data
    mixing_mats_test, mixing_inv_test = gen_mixing_mats(n_mixing=n_test, mix_type=mix_type, mat_size=mat_size,
                                                        det_threshold=det_threshold)
    testloader_A = torch.utils.data.DataLoader(mixing_mats_test, batch_size=1)
    testloader_Ainv = torch.utils.data.DataLoader(mixing_inv_test, batch_size=1)

    return list(zip(trainloader_A, trainloader_Ainv)), list(zip(testloader_A, testloader_Ainv))


if __name__ in "__main__":

    # generate sources
    s = gen_chords()  # other opt: gen_sines(), gen_ica_style()
    n_sig = s.shape[1]

    # generate mixings (contexts)
    test_mats = gen_mixing_mats(mat_size=(n_sig, n_sig), n_mixing=1, mix_type='random_norm')
    A = test_mats[0][0].numpy()

    # apply mixing to sources to get sensory stimuli
    x = A @ s.T

    # plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(5, 3), dpi=150)
    ax[0].plot(s[:1000])
    ax[1].plot(x.T[:1000], c='gray')
    ax[0].set(ylabel='sources s', title='example sources and mixing')
    ax[1].set(xlabel='time (samples)', ylabel='sensory stim. x')
    plt.tight_layout()
    plt.show()
