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

class StlsqFVUPerIt:
    def __init__(self,
             model= "Ridge",
             names= None,
             alpha= 0.1,
             beta= 0.1,
             amountLeft= 1,
             valSize= 0.2,
             switchAtAmountOfTerms= 8,
             paretoPlot= False,
             suppressWarnings= False):

        self.names = np.array(names)
        self.alpha = alpha
        self.beta = beta
        self.amountLeft = amountLeft
        self.valSize = valSize
        self.switchAtAmountOfTerms = switchAtAmountOfTerms

        self.model = self.initModel(model)

        self.historyCoefs = [] #
        self.historyMasks = [] #
        self.thrownAwayNames = [" "] #
        self.coefs = None
        self.mask = None
        self.nextItMask = None
        self.lhs = None
        self.X = None
        self.it = None
        self.predictedFitIt = None

        self.paretoPlot = paretoPlot
        self.nTerms = []
        self.FVUs = []

        self.initialized = False
        self.finalFit = False

        if suppressWarnings: # Also other warnings in the session will be suppressed by this
            warnings.filterwarnings("ignore")

    # The main loop of the algorithm, executes all other important functions every iteration
    def mainIt(self, forceContinue= False):
        self.it += 1
        self.mask = self.nextItMask
        self.model.fit(np.take(self.X, self.mask, axis= 1), self.lhs)
        self.coefs = self.model.coef_
        self.historyCoefs.append(self.coefs.flatten())
        self.historyMasks.append(self.mask)
        self.paretoIt()

        if self.change:
            if len(self.mask) > self.switchAtAmountOfTerms:
                idx = np.argmin(abs(self.coefs))
                self.nextItMask = np.delete(self.mask, idx, axis=0)
            else:
                idx = self.FVUOnePerIt()
                self.nextItMask = np.delete(self.mask, idx, axis=0)
            self.thrownAwayNames.append(self.names[self.mask][idx])

        if len(self.mask) <= self.amountLeft and not forceContinue:
            self.change = False

    # The implementation of the throwing away terms based on FVU value
    def FVUOnePerIt(self):
        worstIdx = None
        worstScore = np.inf
        tempMask = self.mask
        if len(tempMask) == 1:
            return 0
        for i in range(len(tempMask)):
            removedIdx = tempMask[0]
            tempMask = tempMask[1:]
            self.model.fit(np.take(self.X, tempMask, axis=1), self.lhs)
            score = 1 - self.model.score(self.XVal[:, tempMask], self.lhsVal)
            if score <= worstScore:
                worstIdx = i
                worstScore = score
            tempMask = np.append(tempMask, removedIdx)
        return worstIdx

    # The function to start the algorithm to run all iterations at once
    def stlsq(self, lhs, # np.array of shape (samples, 1) with the lhs of the differential equation
              X # list with len equal to amount of terms and every element has shape (samples, 1)
              ):
        if self.initialized:
            raise ValueError("This object was initialised thus it is not possible to use stlsq, use stlsqPerIt instead")
        self.nextItMask = np.arange(len(X))
        self.lhs = lhs
        self.X = np.hstack(X)

        self.X, self.XVal, self.lhs, self.lhsVal = train_test_split(self.X, self.lhs, test_size= self.valSize)

        self.change = True
        self.it = -1
        while self.change:
            self.mainIt()
        self.finalIt()
        print("Sequential fitting terminated, stopping condition is satisfied")

    # The function to start the algorithm to run iterations on command by the user
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

        self.X, self.XVal, self.lhs, self.lhsVal = train_test_split(self.X, self.lhs, test_size=self.valSize)

        self.initialized = True

    # The function to run iterations on command
    def stlsqPerIt(self, its= 1, forceContinue= False):
        if not self.initialized:
            raise ValueError("Use InitStlsqPerIt first to initialise this procedure")
        if forceContinue:
            self.change = True
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
            print("Run one more time to obtain predicted solution")

    # Initialize the fitting model used
    def initModel(self, model):
        if model == "Ridge":
            return Ridge(alpha= self.alpha, fit_intercept= False)
        elif model == "Lasso":
            return Lasso(alpha= self.beta, fit_intercept= False)
        elif model == "Elastic-Net":
            return ElasticNet(alpha= self.beta + 2*self.alpha, l1_ratio= self.beta/(self.beta + 2*self.alpha), fit_intercept= False)
        else:
            raise ValueError("Only Ridge, Lasso and Elastic-Net are implemented")

    # Print the found solution
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

    # Calculate the FVU of the current iteration
    def FVU(self):
        return 1 - self.model.score(self.XVal[:, self.mask], self.lhsVal)

    # Calculate the FVU of given data
    def externalFVU(self, X, lhs):
        return 1 - r2_score(lhs, X)

    # Add the relevant values for the Pareto plot to the lists
    def paretoIt(self):
        self.nTerms.append(len(self.mask))
        self.FVUs.append(self.FVU())
        if self.paretoPlot:
            self.plotPareto()

    # Make the Pareto plot
    def plotPareto(self, saveLoc= None, title= "FVU of terms per iteration", xAxis= "# Non-zero Terms", yAxis= r"$FVU(\frac{\partial f}{\partial t}$)", noLims= False, label= "FVU of terms per iteration", labelBest= "FVU of predicted terms"):
        plt.ion()
        plt.semilogy(self.nTerms, self.FVUs, '.', label= label, marker= "o", markersize= "8")
        if self.finalIt:
            plt.semilogy(self.nTerms[self.predictedFitIt], self.FVUs[self.predictedFitIt], '.', label= labelBest, marker= "o", markersize= "8", color= "red")
        if not noLims:
            plt.xlim(0, self.nTerms[0] + 1)
            plt.ylim(min(self.FVUs) * 0.9, max(self.FVUs) * 1.1)
        plt.title(title)
        plt.xlabel(xAxis)
        plt.ylabel(yAxis)
        plt.legend()
        if saveLoc != None:
            plt.savefig(os.path.join(saveLoc, 'paretoPlot.png'), bbox_inches="tight")
        plt.show()

    # After the solution is found do one final regular fit
    def finalIt(self):
        newModel = LinearRegression(fit_intercept= False)
        temp = np.array(self.FVUs)/np.roll(np.array(self.FVUs), 1)
        temp[0] = 0
        self.predictedFitIt = np.argmax(temp) - 1
        newModel.fit(np.take(np.vstack((self.X, self.XVal)), self.historyMasks[self.predictedFitIt], axis=1), np.vstack((self.lhs, self.lhsVal)))
        self.mask = self.historyMasks[self.predictedFitIt]
        self.historyMasks.append(self.mask)
        self.coefs = newModel.coef_
        self.historyCoefs.append(self.coefs.flatten())
        self.finalFit = True
