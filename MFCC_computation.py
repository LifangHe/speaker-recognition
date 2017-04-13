import numpy
from matplotlib import pyplot as pl
import math

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


def create_dataset(gender_classes, FILES, nceps=13, mode='mean'):
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
        ceps = compute_mfcc(FILES[input_gender][:size], nceps, mode)
        for input_ceps in ceps:
            dataset.append(numpy.append(input_ceps, output_class))
    dataset = numpy.array(dataset)

    # shuffle dataset
    numpy.random.shuffle(dataset)
    return dataset


def print_mse(mse, n_neurons, figsize=(15, 4), ylim=(0, 1)):
    pl.figure(figsize=figsize)

    for n in numpy.arange(mse.shape[0]):
        pl.subplot(1, mse.shape[0], n + 1)

        for i in numpy.arange(mse.shape[1]):
            pl.plot(mse[n, i, :], c='b')

        pl.ylim(ylim)
        pl.xlabel('Epochs')
        pl.ylabel('mse')
        pl.title(str(n_neurons[n]) + ' neurons')
        pl.grid()

    pl.tight_layout()


def print_mse_train_test(mse_train, mse_test, n_neurons, figsize=(16, 8), aspect=20):
    pl.figure(figsize=figsize)

    # plot training
    pl.subplot(2, 1, 1)
    pl.imshow(mse_train, vmin=numpy.min(mse_train), vmax=numpy.percentile(mse_train, 90), aspect=aspect,
              interpolation='nearest')
    pl.yticks(numpy.arange(len(n_neurons)), n_neurons)
    pl.xlabel('Epochs')
    pl.ylabel('Number of hidden Neurons')
    pl.title('Training')
    pl.colorbar()

    # plot tests
    pl.subplot(2, 1, 2)
    pl.imshow(mse_test, vmin=numpy.min(mse_test), vmax=numpy.percentile(mse_test, 90), aspect=aspect,
              interpolation='nearest')
    pl.yticks(numpy.arange(len(n_neurons)), n_neurons)
    pl.xlabel('Epochs')
    pl.ylabel('Number of hidden Neurons')
    pl.title('Test')
    pl.colorbar()

    pl.tight_layout()


def print_coeff_boxplot(keys, FILES, nceps=13, ylim=(-3.5, 20)):
    NUM_PLOTS = len(keys)
    NUM_COLS = 3
    NUM_ROWS = math.ceil(NUM_PLOTS / float(NUM_COLS))

    pl.figure(figsize=(15, 5 * NUM_ROWS))

    X_LABEL = 'coefficients'

    for plot_index, s_class in enumerate(sorted(keys)):
        values = compute_mfcc(FILES[s_class], nceps)
        pl.subplot(NUM_ROWS, NUM_COLS, plot_index + 1)
        pl.ylim(ylim)
        pl.boxplot(values)
        pl.title(s_class)
        pl.xlabel(X_LABEL)
        pl.grid()

    pl.tight_layout()


def conf_mat_stats(matrix):
    # true_positive + false_positive for each class
    tp_fp = numpy.sum(matrix, axis=0)

    # true_positive + false_negative for each class
    tp_fn = numpy.sum(matrix, axis=1)

    # init precision, recalls, and f1 scores for each class
    precisions = numpy.zeros(matrix.shape[0])
    recalls = numpy.zeros(matrix.shape[0])
    f1_scores = numpy.zeros(matrix.shape[0])

    for i in numpy.arange(matrix.shape[0]):
        tp = matrix[i][i]
        p = tp / float(tp_fp[i])  # precision = tp/(tp + fp)
        r = tp / float(tp_fn[i])  # recall = tp/(tp + fn)
        precisions[i] = p
        recalls[i] = r

        # f1-score = 2 x precision x recall / ( precision + recall)
        f1_scores[i] = 2 * p * r / float(p + r)

    return numpy.mean(precisions), numpy.mean(recalls), numpy.mean(f1_scores)