import numpy
from matplotlib import pyplot as pl

from mlp_backprop_momentum import MLP
import k_fold_cross_validation as cv

import scipy.io.wavfile as wav
from scikits.talkbox.features import mfcc

def read_wav_files(files):
    """
    Get the sample rate and all data for each files
    :param files: an array of file paths (must be .wav files)
    :return: (rates, data)
    """
    n = len(files)
    data = []
    rates = numpy.zeros(n)
    for i in xrange(len(files)):
        rate, d = wav.read(files[i])
        rates[i] = rate
        data.append(d)
    return rates, data


def compute_mfcc(files, nceps=13, mode='mean'):
    """
    Calculate Mel-frequency cepstral coefficients (MFCCs) for each files
    and use the coefficients mean to summarize each file. So each file gets
    a vector of 13 coefficients instead of having a matrice containing coefficients
    for all windows.

    :param files: an array of file paths (must be .wav files)
    :return: an array of shape (num_files, 13)
    """
    num_files = len(files)
    sample_rates, data = read_wav_files(files)
    ceps_mean = numpy.zeros((num_files, nceps))

    for i in xrange(0, num_files):
        ceps_i, _, _ = mfcc(data[i], fs=sample_rates[i], nceps=nceps)
        ceps_mean[i] = getattr(numpy, mode)(ceps_i, axis=0)

    return ceps_mean


def create_dataset(gender_classes, FILES, nceps=13):
    """
    Creates a dataset for training.
    Note that the returned dataset is shuffled to prevent issues during
    training.

    :param gender_classes: an array of tuples [(gender_key, output_class),]
    :return: dataset: a 2D-array which has a shape of (num_files, num_coeffs + 1)
        num_files: is the total number of files
        num_coeffs: is the number of MFCC coefficient (see MFCC_COEFFS)
        and finally the output class is added a the end of each element of the dataset
    """

    # use the same number of files for all class
    size = min([len(FILES[gender]) for gender, _ in gender_classes])

    # create dataset
    dataset = []
    for input_gender, output_class in gender_classes:
        ceps = compute_mfcc(FILES[input_gender][:size], nceps)
        for input_ceps in ceps:
            dataset.append(numpy.append(input_ceps, output_class))
    dataset = numpy.array(dataset)

    # shuffle dataset
    numpy.random.shuffle(dataset)
    return dataset
