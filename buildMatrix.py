import numpy as np
import scipy.linalg
from itertools import product, compress, combinations
import time
import integralForm

## In this code I assume t is the first dimension

## Sample points
# Main Function
# Sample subset of points randomly, uniformly, in phase space
def samplePoints(lenDomains, samples, frontBoundaries, endBoundaries, rngSeed= None):
    rng = np.random.default_rng(rngSeed)
    points = np.zeros((samples, len(lenDomains)), dtype= int)
    for i in range(len(lenDomains)):
        points[:, i] = rng.choice(
            range(frontBoundaries[i], lenDomains[i] - endBoundaries[i]),
            size=samples, replace=True)
    return points

# Main Function
# Sample all points in phase space
def sampleAllPoints(lenDomains):
    domainPoints = []
    for i in range(len(lenDomains)):
        domain = range(0, lenDomains[i])
        domainPoints.append(domain)
    points = {}
    counter = 0
    for item in product(*domainPoints):
        points[counter] = item
        counter += 1
    return points

## Main functions to calculate library
# Main Function
# Create the time derivative of a given quantity using the derivative and noise mitigation method supplied
def derivToTime(f, dt, timer= False, method= "AllPoints", points= None, intervalLengths= None, dxsIntegral= None, rangesList= None, derivMethod= "CFD", degOfPoly= None, widthOfPoly= None):
    if timer:
        timeStart = time.time()
    if method not in ["AllPoints", "IntegralForm", "SamplePoints"]:
        raise ValueError("Method Unknown")
    ft = deriv(f, 0, dt, 1, derivMethod= derivMethod, degOfPoly= degOfPoly, widthOfPoly= widthOfPoly)
    if method == "IntegralForm":
        ft = integralForm.integralForm([ft], intervalLengths, dxsIntegral, rangesList= rangesList)[0]
        if timer:
            print("df/dt created after ", time.time() - timeStart, " seconds")
        return ft
    elif method == "SamplePoints":
        ft = convertToSamplePoints([ft], points)[0]
        if timer:
            print("df/dt created after ", time.time() - timeStart, " seconds")
        return ft
    else:
        if timer:
            print("df/dt created after ", time.time() - timeStart, " seconds")
        return ft.reshape(-1, 1)

# Main Function
# Create given quantities using the noise mitigation method supplied
def buildMatrixFunctions(matrix, matrixNames, fs, shapeOfData, maxP, masksOfFs,
                    namesOfFs, timer= False, method= "AllPoints", points= None, intervalLengths= None, dxsIntegral= None, rangesList= None, derivMethod= "CFD", degOfPoly= None, widthOfPoly= None):
    if method not in ["AllPoints", "IntegralForm", "SamplePoints"]:
        raise ValueError("Method Unknown")
    if timer:
        timeStart = time.time()

    dims = len(shapeOfData)

    # Create constant
    ones = np.ones(shapeOfData)
    if method == "IntegralForm":
        ones = integralForm.integralForm([ones], intervalLengths, dxsIntegral, rangesList)[0]
    elif method == "SamplePoints":
        ones = convertToSamplePoints([ones], points)[0]
    elif method == "AllPoints":
        ones = ones.reshape(-1, 1)
    else:
        raise ValueError("Method unknown")
    matrix.append(ones)
    matrixNames.append("1")
    if timer:
        print(matrixNames[-1], " created after ", time.time() - timeStart, " seconds")
    del ones

    # Create functions
    for f, maskOfF, nameOfF in zip(fs, masksOfFs, namesOfFs):
        fMatrix = expandDimensions(f, maskOfF, shapeOfData)
        if method == "IntegralForm":
            matrix.append(integralForm.integralForm([fMatrix], intervalLengths, dxsIntegral, rangesList)[0])
        elif method == "SamplePoints":
            matrix.append(convertToSamplePoints([fMatrix], points)[0])
        elif method == "AllPoints":
            matrix.append(fMatrix.reshape(-1, 1))
        else:
            raise ValueError("Method unknown")
        matrixNames.append(nameOfF)
        if timer:
            print(matrixNames[-1], " created after ", time.time() - timeStart, " seconds")
        newF = None
        for P in range(2, maxP + 1):
            newF = fMatrix ** P
            if method == "IntegralForm":
                matrix.append(integralForm.integralForm([newF], intervalLengths, dxsIntegral, rangesList)[0])
            elif method == "SamplePoints":
                matrix.append(convertToSamplePoints([newF], points)[0])
            elif method == "AllPoints":
                matrix.append(newF.reshape(-1, 1))
            else:
                raise ValueError("Method unknown")
            matrixNames.append(nameOfF + "^" + str(P))
            if timer:
                print(matrixNames[-1], " created after ", time.time() - timeStart, " seconds")

# Main Function
# Create the derivative to non-time dimensions of a given quantity multiplied by given prefactors using the derivative and noise mitigation method supplied
# Also more options are supplied, for example take powers, multiply derivatives with eachother, choose a maximum amount of derivative degree, choose if the derivative to multiple dimensions should be taken...
def buildMatrixDerivatives(matrix, matrixNames, f, maskOfF, nameOfF, dxs, maxPDer, maxD, namesOfDxs, shapeOfData, prefactors= None, masksOfPrefactors= None, namesOfPrefactors= None, maxPPre= 1, firstTimesFirst= False, mixDerivatives= False, multiplyDerivatives= False, multiplyPrefactors= False, timer= False, method= "AllPoints", points= None, intervalLengths= None, dxsIntegral= None, rangesList= None, derivMethod= "CFD", degOfPoly= None, widthOfPoly= None):
    if method not in ["AllPoints", "IntegralForm", "SamplePoints"]:
        raise ValueError("Method Unknown")
    if timer:
        timeStart = time.time()
    else:
        timeStart = None
    if multiplyDerivatives and not firstTimesFirst:
        print("Warning! if you multiply derivatives by each other and give f as a prefactor you get some duplications of terms!")
    dims = len(shapeOfData)
    # Make f the right shape
    if sum(maskOfF) != len(shapeOfData):
        f = expandDimensions(f, maskOfF, shapeOfData)
    # Multiply prefactors with eachother
    # Be aware, this takes a lot of space. If something like cross product should be made it is better to use the crossproduct function first
    if multiplyPrefactors:
        newPrefactors = []
        masksOfNewPrefactors = []
        namesOfNewPrefactors = []
        for terms, names, masks in zip(combinations(prefactors, 2), combinations(namesOfPrefactors, 2), combinations(masksOfPrefactors, 2)):
            newPrefactors.append(expandDimensions(terms[0], masks[0], shapeOfData) * expandDimensions(terms[1], masks[1], shapeOfData))
            masksOfNewPrefactors.append([True]*len(shapeOfData))
            namesOfNewPrefactors.append(names[0] + names[1])
            #if masks[0] == masks[1]:
            #    newPrefactors.append(terms[0] * terms[1])
            #    masksOfNewPrefactors.append(masks[0])
            #    namesOfNewPrefactors.append(names[0] + names[1])
        prefactors += newPrefactors
        masksOfPrefactors += masksOfNewPrefactors
        namesOfPrefactors += namesOfNewPrefactors
    # Create arrays for taking derivatives
    DsList, dxsNames, indexDerives = createArraysForDerivatives(dims, maxD, namesOfDxs, mixDerivatives, maskOfF, deleteF= False)
    # Create derivatives
    if multiplyDerivatives:
        derivatives = []
        derivativeNames = []
    for Ds, dxsName in zip(DsList, dxsNames):
        tempDerivative = f
        for d, indexDerive, dx in zip(Ds, indexDerives, dxs):
            if d != 0:
                tempDerivative = deriv(tempDerivative, indexDerive, dx, d, derivMethod= derivMethod, degOfPoly=degOfPoly,
                                       widthOfPoly=widthOfPoly)
        if np.all(Ds == [0]*(dims - 1)):
            derivativeName = nameOfF
            if not firstTimesFirst:
                generateTermWithPrefactorsAndPowers(matrix, matrixNames, tempDerivative, derivativeName, maxPPre,
                                                    maxPDer,
                                                    prefactors=prefactors[1:],
                                                    masksOfPrefactors=masksOfPrefactors[1:],
                                                    namesOfPrefactors=namesOfPrefactors[1:], method=method,
                                                    points=points, intervalLengths=intervalLengths,
                                                    dxsIntegral=dxsIntegral, rangesList= rangesList, timeStart=timeStart)
            else:
                generateTermWithPrefactorsAndPowers(matrix, matrixNames, tempDerivative, derivativeName, maxPPre,
                                                    maxPDer,
                                                    prefactors=prefactors,
                                                    masksOfPrefactors=masksOfPrefactors,
                                                    namesOfPrefactors=namesOfPrefactors, method=method,
                                                    points=points, intervalLengths=intervalLengths,
                                                    dxsIntegral=dxsIntegral, rangesList= rangesList, timeStart=timeStart)
        else:
            derivativeName = "d" + nameOfF + "/" + dxsName
            generateTermWithPrefactorsAndPowers(matrix, matrixNames, tempDerivative, derivativeName, maxPPre, maxPDer,
                                            prefactors=prefactors,
                                            masksOfPrefactors=masksOfPrefactors, namesOfPrefactors=namesOfPrefactors, method=method,
                                            points=points, intervalLengths=intervalLengths, dxsIntegral=dxsIntegral, rangesList= rangesList, timeStart=timeStart)
        if multiplyDerivatives:
            derivatives.append(tempDerivative)
            derivativeNames.append(derivativeName)
    if multiplyDerivatives:  # If space would become a problem but time not I can still calculate derivatives 2 at a time here, it will just take a lot more time to compute
        for terms, names in zip(combinations(derivatives, 2), combinations(derivativeNames, 2)):
            if method == "IntegralForm":
                matrix.append(integralForm.integralForm([terms[0] * terms[1]], intervalLengths, dxsIntegral, rangesList)[0])
            elif method == "SamplePoints":
                matrix.append(convertToSamplePoints([terms[0] * terms[1]], points)[0])
            elif method == "AllPoints":
                matrix.append((terms[0] * terms[1]).reshape(-1, 1))
            else:
                raise ValueError("Method unknown")
            matrixNames.append("(" + names[0] + ")" + "(" + names[1] + ")")
            if timer:
                print(matrixNames[-1], " created after ", time.time() - timeStart, " seconds")

# Main Function
# Creates the crossproduct of two 3D quantities
def crossProduct(a1, a2, a3, b1, b2, b3, a1Name, a2Name, a3Name, b1Name, b2Name, b3Name, masks, shapeOfData): # Change such that using the 3 different v dimensions does not make the crossproduct all these dimensions
    crossProductValues = []
    crossProductNames = []
    crossProductMasks = []

    newMask = [False for _ in range(len(shapeOfData))]
    newShapeOfData = []
    tempMask1 = []
    tempMask2 = []
    for i in range(len(shapeOfData)):
        if masks[1][i] or masks[5][i]:
            newMask[i] = True
            newShapeOfData.append(shapeOfData[i])
            tempMask1.append(masks[1][i])
            tempMask2.append(masks[5][i])
    crossProductValues.append(expandDimensions(a2, tempMask1, newShapeOfData) * expandDimensions(b3, tempMask2, newShapeOfData))
    crossProductNames.append(a2Name + b3Name)
    crossProductMasks.append(newMask)

    newMask = [False for _ in range(len(shapeOfData))]
    newShapeOfData = []
    tempMask1 = []
    tempMask2 = []
    for i in range(len(shapeOfData)):
        if masks[2][i] or masks[4][i]:
            newMask[i] = True
            newShapeOfData.append(shapeOfData[i])
            tempMask1.append(masks[2][i])
            tempMask2.append(masks[4][i])
    crossProductValues.append(-expandDimensions(a3, tempMask1, newShapeOfData) * expandDimensions(b2, tempMask2, newShapeOfData))
    crossProductNames.append("-" + a3Name + b2Name)
    crossProductMasks.append(newMask)

    newMask = [False for _ in range(len(shapeOfData))]
    newShapeOfData = []
    tempMask1 = []
    tempMask2 = []
    for i in range(len(shapeOfData)):
        if masks[2][i] or masks[3][i]:
            newMask[i] = True
            newShapeOfData.append(shapeOfData[i])
            tempMask1.append(masks[2][i])
            tempMask2.append(masks[3][i])
    crossProductValues.append(expandDimensions(a3, tempMask1, newShapeOfData) * expandDimensions(b1, tempMask2, newShapeOfData))
    crossProductNames.append(a3Name + b1Name)
    crossProductMasks.append(newMask)

    newMask = [False for _ in range(len(shapeOfData))]
    newShapeOfData = []
    tempMask1 = []
    tempMask2 = []
    for i in range(len(shapeOfData)):
        if masks[0][i] or masks[5][i]:
            newMask[i] = True
            newShapeOfData.append(shapeOfData[i])
            tempMask1.append(masks[0][i])
            tempMask2.append(masks[5][i])
    crossProductValues.append(-expandDimensions(a1, tempMask1, newShapeOfData) * expandDimensions(b3, tempMask2, newShapeOfData))
    crossProductNames.append("-" + a1Name + b3Name)
    crossProductMasks.append(newMask)

    newMask = [False for _ in range(len(shapeOfData))]
    newShapeOfData = []
    tempMask1 = []
    tempMask2 = []
    for i in range(len(shapeOfData)):
        if masks[0][i] or masks[4][i]:
            newMask[i] = True
            newShapeOfData.append(shapeOfData[i])
            tempMask1.append(masks[0][i])
            tempMask2.append(masks[4][i])
    crossProductValues.append(expandDimensions(a1, tempMask1, newShapeOfData) * expandDimensions(b2, tempMask2, newShapeOfData))
    crossProductNames.append(a1Name + b2Name)
    crossProductMasks.append(newMask)

    newMask = [False for _ in range(len(shapeOfData))]
    newShapeOfData = []
    tempMask1 = []
    tempMask2 = []
    for i in range(len(shapeOfData)):
        if masks[1][i] or masks[3][i]:
            newMask[i] = True
            newShapeOfData.append(shapeOfData[i])
            tempMask1.append(masks[1][i])
            tempMask2.append(masks[3][i])
    crossProductValues.append(-expandDimensions(a2, tempMask1, newShapeOfData) * expandDimensions(b1, tempMask2, newShapeOfData))
    crossProductNames.append("-" + a2Name + b1Name)
    crossProductMasks.append(newMask)

    return crossProductValues, crossProductNames, crossProductMasks

# Help function of "buildMatrixDerivatives"
# Creates the wanted quantities by multiplying two calculated quantities using the noise mitigation technique provided
def generateTermWithPrefactorsAndPowers(matrix, matrixNames, term, termName, maxPPre, maxPTerm, prefactors= None, masksOfPrefactors= None, namesOfPrefactors= None, method= "AllPoints", points= None, intervalLengths= None, dxsIntegral= None, rangesList= None, timeStart= None):
    for p in range(1, maxPTerm + 1):
        newTerm = term**p
        if method == "IntegralForm":
            matrix.append(integralForm.integralForm([newTerm], intervalLengths, dxsIntegral, rangesList)[0])
        elif method == "SamplePoints":
            matrix.append(convertToSamplePoints([newTerm], points)[0])
        elif method == "AllPoints":
            matrix.append(newTerm.reshape(-1, 1))
        else:
            raise ValueError("Method unknown")
        if p == 1:
            tempTermName = termName
        else:
            tempTermName = "(" + termName + ")" + "^" + str(p)
        matrixNames.append(tempTermName)

        if timeStart != None:
            print(matrixNames[-1], " created after ", time.time() - timeStart, " seconds")

        if np.all(prefactors != None):
            for prefactor, maskOfPrefactor, prename in zip(prefactors, masksOfPrefactors, namesOfPrefactors):
                for pPre in range(1, maxPPre + 1):
                    newColumn = newTerm * expandDimensions(prefactor**pPre, maskOfPrefactor, term.shape)
                    if method == "IntegralForm":
                        matrix.append(integralForm.integralForm([newColumn], intervalLengths, dxsIntegral, rangesList)[0])
                    elif method == "SamplePoints":
                        matrix.append(convertToSamplePoints([newColumn], points)[0])
                    elif method == "AllPoints":
                        matrix.append(newColumn.reshape(-1, 1))
                    else:
                        raise ValueError("Method unknown")
                    del newColumn
                    if pPre == 1:
                        tempPreName = prename
                    else:
                        tempPreName = prename + "^" + str(pPre)
                    matrixNames.append(tempPreName + tempTermName)
                    if timeStart != None:
                        print(matrixNames[-1], " created after ", time.time() - timeStart, " seconds")

# Help function of "buildMatrixDerivatives"
# Creates dataset of the full dimensionality of the system from the essential dimensions for a quantity
def expandDimensions(f, maskOfF, shapeOfData):
    maskOfPrefactorInv = tuple([np.array(maskOfF) == False])
    idxs = np.arange(len(shapeOfData))[maskOfPrefactorInv]
    expandedF = f
    for axis in idxs:
        expandedF = [np.expand_dims(expandedF, axis= axis)]*shapeOfData[axis]
        expandedF = np.concatenate(expandedF, axis= axis)
    return expandedF

# Help function of "buildMatrixDerivatives"
def createArraysForDerivatives(dims, maxD, namesOfDxs, mixDerivatives, maskOfF= None, deleteF= False):
    if maskOfF != None:
        maskOfF = maskOfF[1:]
    indexDerives = range(1, dims)
    if mixDerivatives:
        DsListTemp = range(0, maxD + 1)
        DsListTemp = [DsListTemp for _ in range(dims - 1)]
        DsListTemp = np.meshgrid(*DsListTemp, indexing="ij")
        DsListTemp = np.array(DsListTemp, dtype= int).T.reshape(-1, dims - 1)
        if deleteF:
            DsListTemp = np.delete(DsListTemp, DsListTemp[DsListTemp == [0] * (dims - 1)], axis=0)
        if maskOfF != None:
            for i in range(len(maskOfF)):
                if not maskOfF[i]:
                    DsListTemp = np.delete(DsListTemp, DsListTemp[:, i] != 0, axis=0)
        DsList = np.delete(DsListTemp, np.sum(DsListTemp, axis=1) > maxD, axis=0)
    else:
        DsList = []
        if not deleteF:
            DsList.append([0]*(dims - 1))
        for i in range(dims - 1):
            if maskOfF[i]:
                for j in range(1, maxD + 1):
                    dZeros = np.zeros(dims - 1, dtype= int)
                    dZeros[i] = j
                    DsList.append(dZeros)

    dxsNames = []
    for orders in DsList:
        newName = str()
        for dx, order in zip(namesOfDxs, orders):
            if order == 0:
                continue
            elif order == 1:
                newName += dx
            else:
                newName += dx + "^" + str(int(order))
        dxsNames.append(newName)

    return DsList, dxsNames, indexDerives

# Help function of "buildMatrixDerivatives"
def convertToSamplePoints(terms, points):
    matrix = []
    for term in terms:
        newColumn = np.zeros((len(points), 1))
        for p in range(len(points)):
            newColumn[p] = term[tuple(points[p])]
        matrix.append(newColumn)
    return matrix

## Functions to create derivatives
# Main function
# Creates derivative based on input method for a given order
def deriv(f, indexDerive, dx, deg, maxDeg= 4, derivMethod= "CFD", degOfPoly= 3, widthOfPoly= 5):
    shapeOfData = f.shape
    df = np.zeros(shapeOfData)

    helpArray = [-1]
    for _ in range(len(shapeOfData) - indexDerive - 1):
        helpArray = [helpArray]
    helpArray = helpArray * shapeOfData[indexDerive]
    for _ in range(indexDerive):
        helpArray = [helpArray]
    helpArray = np.array(helpArray)

    mask = np.zeros(shapeOfData)
    maskOnes = np.ones(shapeOfData[-1]) # Don't ask me why this works but it does
    np.put_along_axis(mask, helpArray, maskOnes, axis= indexDerive)
    if derivMethod == "CFD":
        if deg > maxDeg:
            f = deriv(f, indexDerive, dx, deg - maxDeg, derivMethod= "CFD", degOfPoly= None, widthOfPoly= None)
            deg = maxDeg

        for p in range(shapeOfData[indexDerive]):
            temp = finiteDiff(f, dx, deg, indexDerive, p)
            mask = np.roll(mask, 1, axis= indexDerive)
            np.place(df, mask, temp)

    elif derivMethod == "Poly":
        X = createRegPolys(max(deg + 1, degOfPoly), dx, widthOfPoly)
        for p in range(shapeOfData[indexDerive]):
            temp = polyDiff(f, dx, deg, indexDerive, p, X, degOfPoly, widthOfPoly)
            mask = np.roll(mask, 1, axis=indexDerive)
            np.place(df, mask, temp)

    elif derivMethod == "PolyCheby":
        Tns, chebyCoefs = createChebyPolys(max(deg + 1, degOfPoly), dx, widthOfPoly)
        for p in range(shapeOfData[indexDerive]):
            temp = polyChebyDiff(f, deg, indexDerive, p, Tns, chebyCoefs, degOfPoly, widthOfPoly)
            mask = np.roll(mask, 1, axis=indexDerive)
            np.place(df, mask, temp)
    else:
        raise ValueError("Unkown method to take derivative")
    return df

# Help function of "deriv"
# Implements finite centered difference for a given order
def finiteDiff(u, dx, deg, indexDerive, x):
    if deg == 1:
        if x == 0:
            temp = (-3.0 / 2 * np.take(u, 0, axis=indexDerive) + 2 * np.take(u, 1, axis=indexDerive) - np.take(u, 2,
                                                                                                               axis=indexDerive) / 2) / dx
        elif x == u.shape[indexDerive] - 1:
            n = u.shape[indexDerive]
            temp = (3.0 / 2 * np.take(u, n - 1, axis=indexDerive) - 2 * np.take(u, n - 2, axis=indexDerive) + np.take(u,
                                                                                                                      n - 3,
                                                                                                                      axis=indexDerive) / 2) / dx
        else:
            temp = (np.take(u, x + 1, axis=indexDerive) - np.take(u, x - 1, axis=indexDerive)) / (2 * dx)
        return temp

    if deg == 2:
        if x == 0:
            temp = (2 * np.take(u, 0, axis=indexDerive) - 5 * np.take(u, 1, axis=indexDerive) + 4 * np.take(u, 2,
                                                                                                            axis=indexDerive) - np.take(
                u, 3, axis=indexDerive)) / dx ** 2
        elif x == u.shape[indexDerive] - 1:
            n = u.shape[indexDerive]
            temp = (2 * np.take(u, n - 1, axis=indexDerive) - 5 * np.take(u, n - 2, axis=indexDerive) + 4 * np.take(u,
                                                                                                                    n - 3,
                                                                                                                    axis=indexDerive) - np.take(
                u, n - 4, axis=indexDerive)) / dx ** 2
        else:
            temp = (np.take(u, x + 1, axis=indexDerive) - 2 * np.take(u, x, axis=indexDerive) + np.take(u, x - 1,
                                                                                                        axis=indexDerive)) / dx ** 2
        return temp

    if deg == 3:
        if x == 0:
            temp = (-2.5 * np.take(u, 0, axis=indexDerive) + 9 * np.take(u, 1, axis=indexDerive) - 12 * np.take(u, 2,
                                                                                                                axis=indexDerive) + 7 * np.take(
                u, 3, axis=indexDerive) - 1.5 * np.take(u, 4, axis=indexDerive)) / dx ** 3
        elif x == 1:
            temp = (-2.5 * np.take(u, 1, axis=indexDerive) + 9 * np.take(u, 2, axis=indexDerive) - 12 * np.take(u, 3,
                                                                                                                axis=indexDerive) + 7 * np.take(
                u, 4, axis=indexDerive) - 1.5 * np.take(u, 5, axis=indexDerive)) / dx ** 3
        elif x == u.shape[indexDerive] - 1:
            n = u.shape[indexDerive]
            temp = (2.5 * np.take(u, n - 1, axis=indexDerive) - 9 * np.take(u, n - 2, axis=indexDerive) + 12 * np.take(
                u, n - 3, axis=indexDerive) - 7 * np.take(u, n - 4, axis=indexDerive) + 1.5 * np.take(u, n - 5,
                                                                                                      axis=indexDerive)) / dx ** 3
        elif x == u.shape[indexDerive] - 2:
            n = u.shape[indexDerive]
            temp = (2.5 * np.take(u, n - 2, axis=indexDerive) - 9 * np.take(u, n - 3, axis=indexDerive) + 12 * np.take(
                u, n - 4, axis=indexDerive) - 7 * np.take(u, n - 5, axis=indexDerive) + 1.5 * np.take(u, n - 6,
                                                                                                      axis=indexDerive)) / dx ** 3
        else:
            temp = (np.take(u, x + 2, axis=indexDerive) / 2 - np.take(u, x + 1, axis=indexDerive) + np.take(u, x - 1,
                                                                                                            axis=indexDerive) - np.take(
                u, x - 2, axis=indexDerive) / 2) / dx ** 3
        return temp

    if deg == 4:
        if x == 0:
            temp = (3 * np.take(u, 0, axis=indexDerive) - 14 * np.take(u, 1, axis=indexDerive) + 26 * np.take(u, 2,
                                                                                                                axis=indexDerive) - 24 * np.take(
                u, 3, axis=indexDerive) + 11 * np.take(u, 4, axis=indexDerive) - 2 * np.take(u, 5, axis=indexDerive)) / dx ** 4
        elif x == 1:
            temp = (3 * np.take(u, 1, axis=indexDerive) - 14 * np.take(u, 2, axis=indexDerive) + 26 * np.take(u, 3,
                                                                                                                axis=indexDerive) - 24 * np.take(
                u, 4, axis=indexDerive) + 11 * np.take(u, 5, axis=indexDerive) - 2 * np.take(u, 6, axis=indexDerive)) / dx ** 4
        elif x == u.shape[indexDerive] - 1:
            n = u.shape[indexDerive]
            temp = (3 * np.take(u, n - 1, axis=indexDerive) - 14 * np.take(u, n - 2, axis=indexDerive) + 26 * np.take(u, n - 3,
                                                                                                                axis=indexDerive) - 24 * np.take(
                u, n - 4, axis=indexDerive) + 11 * np.take(u, n - 5, axis=indexDerive) - 2 * np.take(u, n - 6, axis=indexDerive)) / dx ** 4
        elif x == u.shape[indexDerive] - 2:
            n = u.shape[indexDerive]
            temp = (3 * np.take(u, n - 2, axis=indexDerive) - 14 * np.take(u, n - 3, axis=indexDerive) + 26 * np.take(u, n - 4,
                                                                                                                axis=indexDerive) - 24 * np.take(
                u, n - 5, axis=indexDerive) + 11 * np.take(u, n - 6, axis=indexDerive) - 2 * np.take(u, n - 7, axis=indexDerive)) / dx ** 4
        else:
            temp = (np.take(u, x + 2, axis=indexDerive) - 4*np.take(u, x + 1, axis=indexDerive) + 6*np.take(u, x, axis=indexDerive) - 4*np.take(
                u, x - 1, axis=indexDerive) + np.take(
                u, x - 2, axis=indexDerive)) / dx ** 4
        return temp
    raise ValueError("The degree is only implemented up to four, use the deriv function to combine these derivatives to get higher degrees")

# Help function of "deriv"
# Implements polynomial differentiation for a given order
def polyDiff(u, dx, deg, indexDerive, x, X, degOfPoly= 3, widthOfPoly= 5):
    degOfPoly = min(max(deg + 1, degOfPoly), x + 1, u.shape[indexDerive] - x)
    X = X[:, :degOfPoly + 1]
    if degOfPoly == 1:
        if x == 0:
            xArray = np.arange(0, 2, 1)
            X = X[widthOfPoly:widthOfPoly + 2, :]
        elif x == u.shape[indexDerive] - 1:
            xArray = np.arange(x - 1, x + 1, 1)
            X = X[widthOfPoly - 1:widthOfPoly + 1, :]
        else:
            raise RuntimeError("This should be impossible to get into")
    else:
        widthOfPolyNew = min(widthOfPoly, x, u.shape[indexDerive] - x - 1)
        X = X[widthOfPoly - widthOfPolyNew:widthOfPoly + widthOfPolyNew + 1, :]
        xArray = np.arange(x - widthOfPolyNew, x + widthOfPolyNew + 1, 1)
    XT = X.T
    A = XT.dot(X)
    Y = np.tensordot(XT, np.take(u, xArray, axis= indexDerive), axes= [1, indexDerive])
    coefs = scipy.linalg.solve(A, Y) # fit[0, ...] gives the constant, fit[1, ...] the first order coefficients...
    degsOfPoly = np.arange(0, degOfPoly + 1, 1)
    values = polyDiffHelp(x, coefs, degsOfPoly, deg)
    return values

# Help function of "polyDiff"
def polyDiffHelp(x, coefs, degsOfPoly, deg):
    if deg >= len(degsOfPoly):
        return np.zeros(coefs.shape[1:])
    else:
        return coefs[deg]*np.math.factorial(deg)

# Implements polynomial differentiation with chebyshev polynomials for a given order
def polyChebyDiff(u, deg, indexDerive, x, Tns, chebyCoefs, degOfPoly, widthOfPoly):
    degOfPoly = min(max(deg + 1, degOfPoly), x + 1, u.shape[indexDerive] - x)
    X = Tns[:, :degOfPoly + 1]
    chebyCoefs = chebyCoefs[:degOfPoly + 1, :degOfPoly + 1]
    if degOfPoly == 1:
        if x == 0:
            xArray = np.arange(0, 2, 1)
            X = X[widthOfPoly:widthOfPoly + 2, :]
        elif x == u.shape[indexDerive] - 1:
            xArray = np.arange(x - 1, x + 1, 1)
            X = X[widthOfPoly - 1:widthOfPoly + 1, :]
        else:
            raise RuntimeError("This should be impossible to get into")
    else:
        widthOfPolyNew = min(widthOfPoly, x, u.shape[indexDerive] - x - 1)
        X = X[widthOfPoly - widthOfPolyNew:widthOfPoly + widthOfPolyNew + 1, :]
        xArray = np.arange(x - widthOfPolyNew, x + widthOfPolyNew + 1, 1)
    XT = X.T
    A = XT.dot(X)
    Y = np.tensordot(XT, np.take(u, xArray, axis=indexDerive), axes=[1, indexDerive])
    coefs = scipy.linalg.solve(A, Y)  # fit[0, ...] gives the constant, fit[1, ...] the first order coefficients...
    degsOfPoly = np.arange(0, degOfPoly + 1, 1)
    values = polyDiffChebyHelp(x, coefs, degsOfPoly, deg, chebyCoefs)
    return values

# Help function of "polyChebyDiff"
def polyDiffChebyHelp(x, coefs, degsOfPoly, deg, chebyCoefs):
    if deg >= len(degsOfPoly):
        return np.zeros(coefs.shape[1:])
    else:
        return np.tensordot(coefs, chebyCoefs[deg], axes= [0, 0])*np.math.factorial(deg)

# Help function of "deriv" for "polyDiff"
def createRegPolys(maxDeg, dx, widthOfPoly):
    xValuesArray = np.arange(-widthOfPoly*dx, widthOfPoly*dx + dx, dx)
    X = np.column_stack([xValuesArray ** p for p in range(0, maxDeg + 1)])
    return X

# Help function of "deriv" for "polyChebyDiff"
def createChebyPolys(maxDeg, dx, widthOfPoly):
    xValuesArray = np.arange(-widthOfPoly*dx, widthOfPoly*dx + dx, dx)
    Tns = np.zeros((2*widthOfPoly + 1, maxDeg + 1))
    Tns[:, 0] = np.ones_like(xValuesArray)
    Tns[:, 1] = xValuesArray
    chebyCoefs = np.zeros((maxDeg + 1, maxDeg + 1))
    chebyCoefs[0, 0] = 1
    chebyCoefs[1, 1] = 1
    for i in range(2, maxDeg + 1):
        Tns[:, i] = 2*xValuesArray*Tns[:, i - 1] - Tns[:, i - 2]
        chebyCoefs[:, i] = 2*np.roll(chebyCoefs[:, i - 1], 1, axis= 0) - chebyCoefs[:, i - 2]
    return Tns, chebyCoefs

## Manipulate data afterwards
# Remove edges of the calculated data, for example because high derivatives tend to blow up on the edge
def removeEdges(f, ft, matrix, domains, removeRanges):
    shapeOfData = f.shape
    maskWithoutEdges = []
    for j in range(len(shapeOfData)):
        maskWithoutEdges.append(np.arange(removeRanges[j], shapeOfData[j] - removeRanges[j]))
        domains[j] = domains[j][maskWithoutEdges[j]]

    for i in range(len(matrix)):
        matrix[i] = matrix[i].reshape(shapeOfData)
        for j in range(len(shapeOfData)):
            matrix[i] = np.take(matrix[i], maskWithoutEdges[j], axis= j)
        matrix[i] = matrix[i].reshape(-1, 1)

    ft = ft.reshape(shapeOfData)
    for j in range(len(shapeOfData)):
        ft = np.take(ft, maskWithoutEdges[j], axis= j)
    ft = ft.reshape(-1, 1)

    for j in range(len(shapeOfData)):
        f = np.take(f, maskWithoutEdges[j], axis= j)
    return f, ft, matrix, domains

## Create arrays to be plotted
# Computes the fitted function and takes the time integral to calculate the prediction of the original data when given the needed matrices as input
def computeFittedFunctionAndData(fittedMatrices, f, coefs, dt):
    ftFit = np.zeros(f.shape)
    if len(coefs) != len(fittedMatrices):
        raise ValueError("length of coefs and matrices are not equal")
    for i in range(len(fittedMatrices)):
        ftFit += coefs[i]*fittedMatrices[i]

    ftFitNew = np.roll(ftFit, 1, axis= 0)
    ftFitNew[0] = np.zeros(f.shape[1:])
    fFit = f[0] + np.cumsum(ftFitNew, axis= 0) * dt
    return ftFit, fFit

# Computes one term (derivative times prefactor) with given noise mitigation technique, mostly used to compute this term for all points to plot afterwards
def computeDerivativeWithPrefactor(f, Ds, dxs, powerOfTerm= 1, prefactor= None, maskOfPrefactor= None, powerOfPrefactor= None, derivMethod= "CFD", degOfPoly= None, widthOfPoly= None):
    shapeOfData = f.shape
    tempDerivative = f
    indexDerives = range(1, len(shapeOfData))
    for d, indexDerive, dx in zip(Ds, indexDerives, dxs):
        if d != 0:
            tempDerivative = deriv(tempDerivative, indexDerive, dx, d, derivMethod= derivMethod, degOfPoly= degOfPoly, widthOfPoly= widthOfPoly)
    tempDerivative = tempDerivative**powerOfTerm
    # Add prefactors to derivatives
    if np.all(prefactor != None):
        return tempDerivative*expandDimensions(prefactor, maskOfPrefactor, shapeOfData)**powerOfPrefactor
    else:
        return tempDerivative

# Computes the fitted function and takes the time integral to calculate the prediction of the original data when given the whole library and the found solution as input
def computeFittedFunctionAndDataFromFullMatrixAndMask(matrix, f, maskOfCoefs, coefs, dt):
    shape = f.shape
    ftFit = np.zeros(shape)
    coefs = coefs.flatten()
    for i in range(len(maskOfCoefs)):
        ftFit += coefs[i]*matrix[maskOfCoefs[i]].reshape(shape)
    fFit = integrateFt(ftFit, f, dt)
    return ftFit, fFit

# Integrates the input function over time to create original function, the values at initial time of the original function are used as initial condition
def integrateFt(ftFit, f, dt):
    shape = f.shape
    ftFitNew = np.roll(ftFit, 1, axis=0)
    ftFitNew[0] = np.zeros(shape[1:])
    fFit = f[0] + np.cumsum(ftFitNew, axis=0) * dt
    return fFit
