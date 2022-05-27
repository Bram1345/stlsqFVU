import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
import matplotlib.pyplot as plt
import warnings
import os
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

plt.rc('figure', figsize=[12, 8])
plt.rc('font',   size=16)
plt.rc('axes',   titlesize=18)
plt.rc('axes',   labelsize=18)
plt.rc('xtick',  labelsize=18)
plt.rc('ytick',  labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=18)

class Stlsq:
    def __init__(self,
             model="Ridge",
             discardCriterium="Value+FVUOnePerIt",
             stopCriterium="AmountLeft",
             names=None,
             threshold= 0.1,
             alpha= 0.1,
             beta= 0.1,
             amountLeft= 1,
             testSize= 0.2,
             sampleFVU= False,
             crossValidate= True,
             switchAtAmountOfTerms= None,
             pareto= False,
             paretoPlot= False,
             weights= None,
             weightsThreshold= None,
             fForReducing= None,
             suppressWarnings= False):

        if discardCriterium not in ["Threshold", "OnePerIt", "FVUOnePerIt", "Value+FVUOnePerIt"]:
            raise ValueError("discard criterium not known")
        self.discardCriterium = discardCriterium
        if stopCriterium not in ["Threshold", "NoChange", "AmountLeft"]:
            raise ValueError("Stop criterium not known")
        self.stopCriterium = stopCriterium

        if (self.discardCriterium, self.stopCriterium) == ("Threshold", "AmountLeft"):
            raise ValueError("(self.discardCriterium, self.stopCriterium) == (Threshold, AmountLeft) is no valid choice")

        if (self.discardCriterium, self.stopCriterium) == ("OnePerIt", "NoChange"):
            raise ValueError("(self.discardCriterium, self.stopCriterium) == (OnePerIt, NoChange) is no valid choice")

        if (self.discardCriterium, self.stopCriterium) == ("FVUOnePerIt", "NoChange"):
            raise ValueError("(self.discardCriterium, self.stopCriterium) == (FVUOnePerIt, NoChange) is no valid choice")

        if (self.discardCriterium, self.stopCriterium) == ("Value+FVUOnePerIt", "NoChange"):
            raise ValueError("(self.discardCriterium, self.stopCriterium) == (FVUOnePerIt, NoChange) is no valid choice")

        self.names = np.array(names)
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.amountLeft = amountLeft
        self.testSize = testSize
        self.sampleFVU = sampleFVU
        self.crossValidate = crossValidate
        self.switchAtAmountOfTerms = switchAtAmountOfTerms

        self.model = self.initModel(model)

        self.historyCoefs = []
        self.historyMasks = []
        self.thrownAwayNames = [" "]
        self.coefs = None
        self.mask = None
        self.nextItMask = None
        self.lhs = None
        self.X = None
        self.it = None
        self.predictedFitIt = None

        self.pareto = pareto
        self.paretoPlot = paretoPlot
        if pareto:
            self.nTerms = []
            self.FVUs = []
        else:
            self.nTerms = None
            self.FVUs = None
        self.weights = weights
        self.weightsThreshold = weightsThreshold
        self.fForReducing = fForReducing

        self.initialized = False
        self.finalFit = False

        if suppressWarnings: # Also other warnings in the session will be supressed by this
            warnings.filterwarnings("ignore")

    def mainIt(self, forceContinue= False):
        self.it += 1
        self.mask = self.nextItMask
        self.model.fit(np.take(self.X, self.mask, axis= 1), self.lhs, sample_weight= self.weightsVector)
        self.coefs = self.model.coef_
        self.historyCoefs.append(self.coefs.flatten())
        self.historyMasks.append(self.mask)

        if self.pareto or (self.discardCriterium == "FVUOnePerIt" or self.discardCriterium == "Value+FVUOnePerIt"):
            self.paretoIt()

        if self.stopCriterium == "Threshold" and not forceContinue:
            if np.all(abs(self.coefs) > self.threshold):
                self.change = False

        if self.change:
            if self.discardCriterium == "OnePerIt":
                idx = np.argmin(abs(self.coefs))
                self.nextItMask = np.delete(self.mask, idx, axis=0)
            elif self.discardCriterium == "FVUOnePerIt":
                idx = self.FVUOnePerIt()
                self.nextItMask = np.delete(self.mask, idx, axis=0)
            elif self.discardCriterium == "Threshold":
                idx = (abs(self.coefs) < self.threshold).flatten()
                self.nextItMask = np.delete(self.mask, idx, axis=0)
            elif self.discardCriterium == "Value+FVUOnePerIt":
                if len(self.mask) > self.switchAtAmountOfTerms:
                    idx = np.argmin(abs(self.coefs))
                    self.nextItMask = np.delete(self.mask, idx, axis=0)
                else:
                    idx = self.FVUOnePerIt()
                    self.nextItMask = np.delete(self.mask, idx, axis=0)
            self.thrownAwayNames.append(self.names[self.mask][idx])

        if self.stopCriterium == "NoChange":
            if np.all(self.nextItMask == self.mask):
                self.change = False

        if len(self.mask) <= self.amountLeft: # AmountLeft stopping criterion
            self.change = False

    def FVUOnePerIt(self):
        worstIdx = None
        worstScore = np.inf
        tempMask = self.mask
        if len(tempMask) == 1:
            return 0
        if self.sampleFVU and len(tempMask) > self.sampleFVU:
            rng = np.random.default_rng()
            removedListIdxs = rng.integers(0, len(tempMask), size= self.sampleFVU)
            for i in range(self.sampleFVU):
                removedIdx = tempMask[removedListIdxs[i]]
                tempTempMask = np.delete(tempMask, removedListIdxs[i])
                self.model.fit(np.take(self.X, tempTempMask, axis=1), self.lhs, sample_weight= self.weightsVector)
                score = 1 - self.model.score(self.XTest[:, tempTempMask], self.lhsTest, self.weightsVectorVal)
                if score <= worstScore:
                    worstIdx = i
                    worstScore = score
        else:
            for i in range(len(tempMask)):
                removedIdx = tempMask[0]
                tempMask = tempMask[1:]
                self.model.fit(np.take(self.X, tempMask, axis=1), self.lhs, sample_weight= self.weightsVector)
                score = 1 - self.model.score(self.XTest[:, tempMask], self.lhsTest, self.weightsVectorVal)
                if score <= worstScore:
                    worstIdx = i
                    worstScore = score
                tempMask = np.append(tempMask, removedIdx)
        return worstIdx

    def stlsq(self, lhs, # np.array of shape (samples, 1) with the lhs of the differential equation
              X # list with len equal to amount of terms and every element has shape (samples, 1)
              ):
        if self.initialized:
            raise ValueError("This object was initialised thus it is not possible to use stlsq, use stlsqPerIt instead")
        self.nextItMask = np.arange(len(X))
        self.lhs = lhs
        self.X = np.hstack(X)

        if self.discardCriterium == "FVUOnePerIt" or self.discardCriterium == "Value+FVUOnePerIt":
            if self.crossValidate:
                if self.weights == "Relative":
                    self.X, self.XTest, self.lhs, self.lhsTest, self.fForReducing, self.fForReducingTest = train_test_split(self.X, self.lhs, self.fForReducing, test_size= self.testSize)
                else:
                    self.X, self.XTest, self.lhs, self.lhsTest = train_test_split(self.X, self.lhs, test_size= self.testSize)
            else:
                self.XTest = self.X
                self.lhsTest = self.lhs

        if self.weights == "Threshold":
            self.weightsVector = np.where(self.lhs > self.weightsThreshold, 1, 0).flatten()
        elif self.weights == "Relative":
            self.weightsVector = self.fForReducing #self.lhs.flatten()
        elif self.weights == None:
            self.weightsVector = None
        else:
            raise ValueError("Weights argument invalid")

        if self.discardCriterium == "FVUOnePerIt" or self.discardCriterium == "Value+FVUOnePerIt":
            if self.weights == "Threshold":
                self.weightsVectorVal = np.where(self.lhsTest > self.weightsThreshold, 1, 0).flatten()
            elif self.weights == "Relative":
                self.weightsVectorVal = self.fForReducingTest #self.lhsTest.flatten()
            elif self.weights == None:
                self.weightsVectorVal = None
            else:
                raise ValueError("Weights argument invalid")

        self.change = True
        self.it = -1
        while self.change:
            self.mainIt()
        self.finalIt()
        print("Sequential fitting terminated, stopping condition is satisfied")

    def initStlsqPerIt(self,
        lhs,  # np.array of shape (samples, 1) with the lhs of the differential equation
        X  # list with len equal to amount of terms and every element has shape (samples, 1)
        ):
        self.nextItMask = np.arange(len(X))
        self.mask = self.nextItMask
        self.lhs = lhs
        self.X = np.hstack(X)
        self.it = -1
        self.change = True

        if self.discardCriterium == "FVUOnePerIt" or self.discardCriterium == "Value+FVUOnePerIt":
            if self.crossValidate:
                self.X, self.XTest, self.lhs, self.lhsTest = train_test_split(self.X, self.lhs, test_size=self.testSize)
            else:
                self.XTest = self.X
                self.lhsTest = self.lhs

        if self.weights == "Threshold":
            self.weightsVector = np.where(self.lhs > self.weightsThreshold, 1, 0).flatten()
        elif self.weights == "Relative":
            self.weightsVector = self.lhs.flatten()
        elif self.weights == None:
            self.weightsVector = None
        else:
            raise ValueError("Weights argument invalid")

        if self.discardCriterium == "FVUOnePerIt" or self.discardCriterium == "Value+FVUOnePerIt":
            if self.weights == "Threshold":
                self.weightsVectorVal = np.where(self.lhsTest > self.weightsThreshold, 1, 0).flatten()
            elif self.weights == "Relative":
                self.weightsVectorVal = self.lhsTest.flatten()
            elif self.weights == None:
                self.weightsVectorVal = None
            else:
                raise ValueError("Weights argument invalid")

        self.initialized = True

    def stlsqPerIt(self, its= 1, forceContinue= False):
        # forceContinue only does something when stopping criterium is threshold and discard criterium is one per iteration
        if not self.initialized:
            raise ValueError("Use InitStlsqPerIt first to initialise this procedure")
        if forceContinue:
            self.change= True
        if self.change:
            for _ in range(its):
                if len(self.mask) <= 1 and not forceContinue:
                    print("One 1 term left, cannot reduce any further")
                    break
                self.mainIt(forceContinue= forceContinue)
        elif not self.finalFit:
            self.finalIt()
        else:
            print("Sequential fitting terminated, stopping condition satisfied")

    def initModel(self, model):
        if model == "Ridge":
            return Ridge(alpha= self.alpha, fit_intercept= False)
        elif model == "Lasso":
            return Lasso(alpha= self.beta, fit_intercept= False)
        elif model == "Elastic-Net":
            return ElasticNet(alpha= self.beta + 2*self.alpha, l1_ratio= self.beta/(self.beta + 2*self.alpha), fit_intercept= False)
        else:
            raise ValueError("Only Ridge, Lasso and Elastic-Net are implemented")

    def printSolution(self, history = None):
        # Give history = int to get the situation at that iteration
        if np.all(self.names == None):
            raise ValueError("Model did not receive names")
        if history == None:
            mask = self.mask
            coefs = self.coefs
        else:
            mask = self.historyMasks[history]
            coefs = self.historyCoefs[history]
        return list(zip(self.names[mask], coefs.flatten()))

    def FVU(self):
        return 1 - self.model.score(self.X[:, self.mask], self.lhs)

    def externalFVU(self, X, lhs):
        return 1 - r2_score(lhs, X)

    def paretoIt(self):
        self.nTerms.append(len(self.mask))
        self.FVUs.append(self.FVU())
        if self.paretoPlot:
            self.plotPareto()

    def plotPareto(self, extra= None, saveLoc= None, xAxis= "# Non-zero Terms", yAxis= r"$FVU(\frac{\partial f}{\partial t}$)", noLims= False, label= "FVU of found terms", labelExtra= ""):
        if not self.pareto:
            raise ValueError("Pareto should be set to True so the values are kept and can be plotted")
        plt.ion()
        plt.semilogy(self.nTerms, self.FVUs, '.', label= label, marker= "o", markersize= "8")
        if extra != None:
            plt.semilogy(*extra, '.', c= "red", label= labelExtra, marker= "o", markersize= "8")
        if not noLims:
            plt.xlim(0, self.nTerms[0] + 1)
            plt.ylim(min(self.FVUs) * 0.9, max(self.FVUs) * 1.1)
        plt.xlabel(xAxis)
        plt.ylabel(yAxis)
        plt.legend()
        if saveLoc != None:
            plt.savefig(os.path.join(saveLoc, 'paretoPlot.png'), bbox_inches="tight")
        plt.show()

    def finalIt(self):
        newModel = LinearRegression(fit_intercept= False)
        if self.weights != None:
            weightsVectorFinal = np.concatenate((self.weightsVector, self.weightsVectorVal))
        else:
            weightsVectorFinal = None
        if (self.discardCriterium == "FVUOnePerIt" or self.discardCriterium == "Value+FVUOnePerIt") and self.crossValidate:
            self.predictedFitIt = np.argmax(np.array(self.FVUs)/np.roll(np.array(self.FVUs), 1)) - 1 # useful to add penalty for amount of terms? -> no, we want to see where an important term was thrown away
            newModel.fit(np.take(np.vstack((self.X, self.XTest)), self.historyMasks[self.predictedFitIt], axis=1), np.vstack((self.lhs, self.lhsTest)), sample_weight= weightsVectorFinal)
            self.mask = self.historyMasks[self.predictedFitIt]
        elif (self.discardCriterium == "FVUOnePerIt" or self.discardCriterium == "Value+FVUOnePerIt"):
            self.predictedFitIt = np.argmax(np.array(self.FVUs)/np.roll(np.array(self.FVUs), 1)) - 1 # useful to add penalty for amount of terms? -> no, we want to see where an important term was thrown away
            newModel.fit(np.take(self.X, self.historyMasks[self.predictedFitIt], axis=1), self.lhs, sample_weight= weightsVectorFinal)
            self.mask = self.historyMasks[self.predictedFitIt]
        else:
            newModel.fit(np.take(self.X, self.mask, axis=1), self.lhs, sample_weight= weightsVectorFinal)
        self.historyMasks.append(self.mask)
        self.coefs = newModel.coef_
        self.historyCoefs.append(self.coefs.flatten())
        self.finalFit = True
