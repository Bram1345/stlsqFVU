import numpy as np
import time
import math

## To generate the points used for integral formulation
# Sample subset of points randomly, uniformly, in phase space, while keeping boundaries into account like that points too much to the boundary can not be chosen because then an integral around these can not be taken
def samplePointsForIntegralFormRandom(samples, rngSeed, intervalLengths, shape, d = 0):
    rng = np.random.default_rng(rngSeed)
    points = np.zeros((samples, len(intervalLengths)))
    for i in range(len(intervalLengths)):
        points[:, i] = rng.choice(
            range(math.floor(intervalLengths[i] / 2 + d), math.ceil(shape[i] - intervalLengths[i] / 2 - d)),
            size= samples, replace=True)
    return points

# Sample subset of points forced uniform in phase space(thus the same space between all points), while keeping boundaries into account like that points too much to the boundary can not be chosen because then an integral around these can not be taken
def samplePointsForIntegralFormUniform(intervalLengths, spaceBetweenPoints, shape, d= 0):
    length = len(shape)
    if spaceBetweenPoints == None:
        spaceBetweenPoints = intervalLengths
    pointsAlongDims = [None] * length
    for i in range(length):
        pointsAlongDims[i] = np.arange(math.floor(intervalLengths[i] / 2 + d),
                                       math.ceil(shape[i] - intervalLengths[i] / 2 - d), spaceBetweenPoints[i])
    mesh = np.meshgrid(*pointsAlongDims, indexing="ij")
    for i in range(len(mesh)):
        mesh[i] = mesh[i].flatten()
    mesh = np.array(mesh).T
    return mesh

## Calculate the integrals for the specified points and terms
# Main Function
# Computes the integralforms for the specified points and terms
def integralForm(terms, intervalLengths, dxs, rangesList):
    matrix = []
    dxs = np.array(dxs) * np.array(intervalLengths) # Don't know yet why I have to do this to make it work
    for term in terms:
        df = np.zeros((len(rangesList), 1))
        for i in range(len(rangesList)):
            df[i] = integrateMultiDimTrapz(term, rangesList[i], dxs)
        matrix.append(df)
    return matrix

# Help function of "integralForm"
# Computes the ranges to integrate over
def computeRanges(shape, points, intervalLengths):
    length = len(shape)
    rangesList = []
    for i in range(len(points)):
        ranges = [None] * length
        for j in range(length):
            ranges[j] = range(math.ceil(points[i][j] - intervalLengths[j] / 2),
                              math.ceil(points[i][j] + intervalLengths[j] / 2))
        rangesList.append(ranges)
    return rangesList

# Help function of "integralForm"
# Compute integrals by the trapeziumrule
def integrateMultiDimTrapz(f, ranges, dxs):
    length = len(f.shape)
    if length != len(ranges) or length != len(dxs):
        raise ValueError("f and ranges have different lens")
    result = f[tuple(np.meshgrid(*ranges))]
    for i in range(length):
        result = np.trapz(result, axis= 0, dx= dxs[i])
    return result

# Help function of "integralForm"
# Compute integrals by rectangles
def integrateMultiDimLeftRectangles(f, ranges, dxs):
    length = len(f.shape)
    if length != len(ranges) or length != len(dxs):
        raise ValueError("f and ranges have different lens")
    result = f[tuple(np.meshgrid(*ranges))]
    for i in range(length):
        result = np.sum(result, axis= 0)*dxs[i]
    return result
