# -*- coding: utf-8 -*-

import math
import numpy as np
import numpy.ma as ma

__version__ = '0.4'
__revision__ = '$Rev$'
__all__ = ['biweightMean', 'mean', 'mode', 'std', 'checkfit', 'linefit', 'polyfit',
           '__version__', '__revision__', '__all__']

__iterMax = 25
__delta = 5.0e-7
__epsilon = 1.0e-20


def biweightMean(inputData):
    y = inputData.ravel()
    if isinstance(y, ma.MaskedArray):
        y = y.compressed()

    n = len(y)
    closeEnough = 0.03 * np.sqrt(0.5 / (n - 1))

    diff = 1.0e30
    nIter = 0

    y0 = np.median(y)
    deviation = y - y0
    sigma = std(deviation)

    if sigma < __epsilon:
        diff = 0

    while diff > closeEnough:
        nIter += 1
        if nIter > __iterMax:
            break

        uu = ((y - y0) / (6.0 * sigma)) ** 2.0
        uu = np.where(uu > 1.0, 1.0, uu)
        weights = (1.0 - uu) ** 2.0
        weights /= weights.sum()

        y0 = (weights * y).sum()
        deviation = y - y0
        prevSigma = sigma
        sigma = std(deviation, Zero=True)

        if sigma > __epsilon:
            diff = abs(prevSigma - sigma) / prevSigma
        else:
            diff = 0.0

    return y0


def mean(inputData, Cut=3.0):
    data = inputData.ravel()
    if isinstance(data, ma.MaskedArray):
        data = data.compressed()

    data0 = np.median(data)
    maxAbsDev = np.median(np.abs(data - data0)) / 0.6745

    if maxAbsDev < __epsilon:
        maxAbsDev = np.abs(data - data0).mean() / 0.8000

    cutOff = Cut * maxAbsDev
    good = np.where(np.abs(data - data0) <= cutOff)[0]

    dataMean = data[good].mean()
    dataSigma = math.sqrt(((data[good] - dataMean) ** 2.0).sum() / len(good))

    sigmaCut = max(Cut, 1.0)
    if sigmaCut <= 4.5:
        dataSigma /= (-0.15405 + 0.90723 * sigmaCut -
                      0.23584 * sigmaCut ** 2.0 +
                      0.020142 * sigmaCut ** 3.0)

    cutOff = Cut * dataSigma
    good = np.where(np.abs(data - data0) <= cutOff)[0]
    dataMean = data[good].mean()

    if len(good) > 3:
        dataSigma = math.sqrt(((data[good] - dataMean) ** 2.0).sum() / len(good))

    if sigmaCut <= 4.5:
        dataSigma /= (-0.15405 + 0.90723 * sigmaCut -
                      0.23584 * sigmaCut ** 2.0 +
                      0.020142 * sigmaCut ** 3.0)

    dataSigma /= math.sqrt(len(good) - 1)
    return [dataMean, dataSigma]


def mode(inputData):
    def _hsm(data):
        if data.size == 1:
            return data[0]
        elif data.size == 2:
            return data.mean()
        elif data.size == 3:
            i1 = data[1] - data[0]
            i2 = data[2] - data[1]
            if i1 < i2:
                return data[:2].mean()
            elif i2 < i1:
                return data[1:].mean()
            else:
                return data[1]
        else:
            wMin = data[-1] - data[0]
            N = data.size // 2 + data.size % 2  # PY3 FIX
            j = 0
            for i in range(N):  # PY3 FIX
                w = data[i + N - 1] - data[i]
                if w < wMin:
                    wMin = w
                    j = i
            return _hsm(data[j:j + N])

    data = inputData.ravel()
    if isinstance(data, ma.MaskedArray):
        data = data.compressed()

    data = np.sort(data)
    return _hsm(data)


def std(inputData, Zero=False):
    data = inputData.ravel()
    if isinstance(data, ma.MaskedArray):
        data = data.compressed()

    data0 = 0.0 if Zero else np.median(data)
    maxAbsDev = np.median(np.abs(data - data0)) / 0.6745

    if maxAbsDev < __epsilon:
        maxAbsDev = np.abs(data - data0).mean() / 0.8000

    if maxAbsDev < __epsilon:
        return 0.0

    u = (data - data0) / (6.0 * maxAbsDev)
    u2 = u ** 2.0
    good = np.where(u2 <= 1.0)[0]

    if len(good) < 3:
        print("WARNING: Distribution is too strange to compute standard deviation")  # PY3 FIX
        return -1.0

    numerator = ((data[good] - data0) ** 2.0 * (1.0 - u2[good]) ** 2.0).sum()
    nElements = data.size
    denominator = ((1.0 - u2[good]) * (1.0 - 5.0 * u2[good])).sum()

    sigma = nElements * numerator / (denominator * (denominator - 1.0))
    return math.sqrt(sigma) if sigma > 0 else 0.0

