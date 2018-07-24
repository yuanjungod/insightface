import numpy as np
import cv2
import os


def batchnorm_forward(x, gamma, beta, eps):
    N, D = x.shape

    # step1: calculate mean
    mu = 1. / N * np.sum(x, axis=0)
    print(mu.shape)
    # step2: subtract mean vector of every trainings example
    xmu = x - mu
    # step3: following the lower branch - calculation denominator
    sq = xmu ** 2
    # step4: calculate variance
    var = 1. / N * np.sum(sq, axis=0)
    # step5: add eps for numerical stability, then sqrt
    sqrtvar = np.sqrt(var + eps)
    # step6: invert sqrtwar
    ivar = 1. / sqrtvar
    # step7: execute normalization
    xhat = xmu * ivar
    # step8: Nor the two transformation steps
    gammax = gamma * xhat
    # step9
    out = gammax + beta
    # store intermediate
    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)
    return out, cache


def batchnorm_backward(dout, cache):
    # unfold the variables stored in cache
    xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache
    # get the dimensions of the input/output
    N, D = dout.shape
    # step9
    dbeta = np.sum(dout, axis=0)
    dgammax = dout  # not necessary, but more understandable
    # step8
    dgamma = np.sum(dgammax * xhat, axis=0)
    dxhat = dgammax * gamma
    # step7
    divar = np.sum(dxhat * xmu, axis=0)
    dxmu1 = dxhat * ivar
    # step6
    dsqrtvar = -1. / (sqrtvar ** 2) * divar
    # step5
    dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar
    # step4
    dsq = 1. / N * np.ones((N, D)) * dvar
    # step3
    dxmu2 = 2 * xmu * dsq
    # step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
    # step1
    dx2 = 1. / N * np.ones((N, D)) * dmu
    # step0
    dx = dx1 + dx2
    return dx, dgamma, dbeta


if __name__ == "__main__":
    x = np.ndarray((32, 100))
    pic_root = "/Users/happy/Downloads/train_data/train/529/"
    count = 0
    for i in os.listdir(pic_root):
        image = cv2.imread(pic_root + i)
        image = image.reshape(-1)
        # print(image.shape)

        # print(dir(image))
        # print(image[:100])
        # print()
        x[count] = image[:100]
        count += 1
        if count == 32:
            break
    out, cache = batchnorm_forward(x, 1, 1, 0.00001)
    # print out
    # print cache
