# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import itertools
import scipy
import math
import numpy as np
import pandas as pd
from math import floor, log2, sqrt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from numpy.random import randn, rand
from scipy.signal import lfilter
from numpy.fft import ifft, fft
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def nchoosek(n, k):
    if k == 0:
        r = 1
    else:
        r = n / k * nchoosek(n - 1, k - 1)
    return round(r)


# sqrt(L0*(d^(-1*c))).*g
def ofdmim_data_generator(mode, dist, PLexp, M, nSym, EbNo_dB, N, Ncp, n, k, L, SJR_dB, rho):
    g = int(N / n)  # number of subblocks
    K = int(k * g)  # total number of active subcarriers
    Total_length = (Ncp + N)  # total length of each frame
    p1 = floor(log2(nchoosek(n, k)))  # number of bits carried over indices (in a subblock)
    p2 = k * log2(M)  # number of bits carried over modulated signals (in a subblock)
    p = int(p1 + p2)  # total number of bits carried in a subblock
    m = int(g * p)  # number of bits carried in a symbol
    c = int(2 ** p1)  # number of active subcarrier combinations in a subblock
    Eb = (N + Ncp) / m
    Nj_t = (10 ** (-SJR_dB / 10)) * Eb
    # Nj_f = Nj_t * (K / N / rho)
    ##################################################
    if math.floor(g) != g:
        print('g must be integer.')
    if floor(g * rho) != g * rho:
        print('g*rho must be integer.')
    ##################################################
    # constellation maps
    if M == 2:
        constmap = sqrt(1) * np.array([-1, 1])
    elif M == 4:  # Gray mapping
        constmap = sqrt(1 / 2) * np.array([1 + 1j, -1 + 1j, 1 - 1j, - 1 - 1j])
    elif M == 8:
        constmap = sqrt(1 / 5) * np.array([-3 / sqrt(2) + 1j * 3 / sqrt(2), 3 / sqrt(2) + 1j * 3 / sqrt(2),
                                           -3 / sqrt(2) - 1j * 3 / sqrt(2), 3 / sqrt(2) - 1j * 3 / sqrt(2), 1j, 1, -1,
                                           -1j])
    elif M == 16:  # Gray mapping
        constmap = sqrt(1 / 10) * np.array([-3 - 3j, - 3 - 1j, - 3 + 3j, - 3 + 1j, - 1 - 3j,
                                            - 1 - 1j, - 1 + 3j, - 1 + 1j, 3 - 3j, 3 - 1j, 3 + 3j, 3 + 1j, 1 - 3j,
                                            1 - 1j, 1 + 3j, 1 + 1j])
    ##################################################
    # subcarrier combinations
    if k == 2 and n == 4:
        scComb = np.array([[1, 2],
                           [2, 3],
                           [3, 4],
                           [1, 4]])
    elif k == 3 and n == 4:
        scComb = np.array([[1, 2, 3],
                           [2, 3, 4],
                           [1, 3, 4],
                           [1, 2, 4]])
    elif k == 1 and n == 1:
        scComb = np.array([[1]])
    else:
        scComb = list(itertools.combinations(range(1, n + 1), k))
        scComb = scComb[0:c, :]
    ##################################################
    nErr = np.zeros((1, np.size(EbNo_dB)))
    if mode == "train":
        features = np.zeros((len(dist)-1, int(nSym), (m + Ncp) * 2))
        targets = np.ones((len(dist)-1, int(nSym) * (len(dist)-1)))
        targets_tst = np.ones((len(dist), int(nSym)))
    else:
        features = np.zeros((len(dist), int(nSym), (m + Ncp) * 2))
        targets_tst = np.ones((len(dist), int(nSym)))
        targets = np.ones((len(dist), int(nSym)*len(dist)))

    d = np.ones((1, N))
    if rho != 0:
        D = sqrt(K * Nj_t) * np.diag(d) / sqrt(np.sum(d ** 2, 1))
    else:
        D = 0

    ##################################################
    # symbol space
    symbol_space = np.zeros((c * M ** k, n))
    x = np.array(list(itertools.combinations(np.tile(range(1, M + 1), (1, k))[0], k)))
    perm = np.unique(x, axis=0)
    for ll in range(1, c + 1):
        symbol_space[(ll - 1) * M ** k: ll * M ** k, scComb[ll - 1, :] - 1] = constmap[perm - 1].copy()
    # % ==========================================================
    for txInd, chParam in enumerate(zip(dist, PLexp)):
        for symInd in range(1, int(nSym) + 1):
            data = rand(1, m) > 0.5  # random data generation
            #data = np.ones((1, m)).astype(int)  # pilot generation
            X_Freq = np.zeros((1, N))  # signal to be inverse fast fourrier transformed
            # data is treated to be in frequency domain

            # subblock generation
            for jj in range(1, g + 1):
                temp = data[0, p * (jj - 1): p * jj]
                X_Freq[0, n * (jj - 1): n * jj] = symbol_space[int(str(temp * 1)[1:-1].replace(" ", ""), 2), :]
            #  -----------------------IFFT block - -------------------------
            x_time = N / sqrt(K) * ifft(X_Freq, n=N, axis=-1)
            #  adding cyclic prefix
            x_time_cp = np.hstack((x_time[:, N - Ncp: N], x_time))  # [x_time[:, N - Ncp : N], x_time]
            #  ------------------ Channel Modeling - ---------------------
            noise = sqrt(0.5) * (rand(1, Total_length) + 1j * rand(1, Total_length))
            No_t = (10 ** (-EbNo_dB / 10)) * Eb
            No_f = K / N * No_t
            h_time = sqrt(chParam[0] ** -chParam[1]) * sqrt(0.5 / (L + 1)) * (rand(L + 1, 1) + 1j * rand(L + 1, 1))  # Rayleigh channel
            # jamming signal can be added in frequency domain
            jam_f = sqrt(0.5) * (randn(1, N) + 1j * randn(1, N)) * D
            y_time = lfilter(h_time.flatten(), 1, x_time_cp) + sqrt(No_t) * noise
            features[txInd, symInd - 1, :] = np.hstack((np.real(y_time), np.imag(y_time)))
        targets_tst[txInd, :] = txInd
        targets[txInd, int(nSym)*txInd: int(nSym)*(txInd+1)] = 0
        #print(txInd)
        if mode == "train" and txInd == len(dist)-2:
            return features, targets
    return features, targets_tst.reshape(-1)


numSym = 100000  # number of OFDM Symbols to transmit
EbNot_dB = 15  # bit to noise ratio
EbNo_lin = 10 ** (EbNot_dB / 10)
M = 2  # modulation order (2,4,8,16 etc.)
N = 256  # FFT size or total number of subcarriers
Ncp = 30  # number of symbols allocated to cyclic prefix
n = 4  # number of subcarriers in a subblock
k = 2  # number of active subcarriers in a subblock
v = 6  # channel order
ICSI = 0  # 1--> imperfect CSI, 0--> perfect CSI
rho = 0 # jamming parameter
L = v - 1  # channel order - 1, number of side paths
SJR_dB = 10  # signal to jamming ratio
distance = [9.5, 2, 3.5, 8, 5.5, 7]  # node distances
PLexp = [2, 2, 2, 2, 2, 2]  # node path loss exponents
###########################


###########################  Train data

print("Started train data generation")
features_train, targets_train = ofdmim_data_generator("train", distance, PLexp, M, numSym, EbNot_dB, N, Ncp, n, k, L, SJR_dB, rho=0)
print("Started validation data generation")
features_val, targets_val = ofdmim_data_generator("train", distance, PLexp, M, numSym/5, EbNot_dB, N, Ncp, n, k, L, SJR_dB, rho=0)
print("Started test data generation")
features_test, targets_test = ofdmim_data_generator("test", distance, PLexp, M, numSym/5, EbNot_dB, N, Ncp, n, k, L, SJR_dB, rho=0)

with open('features_train.npy', 'wb') as f:
    np.save(f, features_train)
with open('targets_train.npy', 'wb') as f:
    np.save(f, targets_train)
with open('features_val.npy', 'wb') as f:
    np.save(f, features_val)
with open('targets_val.npy', 'wb') as f:
    np.save(f, targets_val)
with open('features_test.npy', 'wb') as f:
    np.save(f, features_test)
with open('targets_test.npy', 'wb') as f:
    np.save(f, targets_test)



print("hello")




