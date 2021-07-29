import numpy as np
from kernel import *
from utils import *
import matplotlib.pyplot as plt


class KernelLogistic(object):
    def __init__(self, kernel=gaussian_kernel, iterations=100,eta=0.01,lamda=0.05,sigma=1):
        self.kernel = lambda x,y: kernel(x,y,sigma)
        self.iterations = iterations
        self.alpha = None
        self.eta = eta     # Step size for gradient descent
        self.lamda = lamda # Regularization term

    def fit(self, X, y):
        ''' find the alpha values here'''
        self.train_X = X
        self.train_y = y
        self.alpha = np.zeros((y.shape[0],1))
        kernel = self.kernel(self.train_X,self.train_X)
        # TODO
        y = np.reshape(y,(np.shape(y)[0],1))
        for i in range(self.iterations):
            gradient = -np.dot(kernel,y)-self.lamda*(np.dot(kernel, self.alpha)) + np.dot(kernel, 1/(1+np.exp(-1*(np.dot(kernel, self.alpha)))))
            self.alpha = self.alpha - self.eta*gradient
        # END TODO
    

    def predict(self, X):
        # TODO 
        kernel = self.kernel(X,self.train_X)
        ypred = 1/(1+np.exp(-1*np.dot(kernel,self.alpha)))
        return np.reshape(ypred, (np.shape(ypred)[0],))
        # END TODO

def k_fold_cv(X,y,k=10,sigma=1.0):
    '''Does k-fold cross validation given train set (X, y)
    Divide train set into k subsets, and train on (k-1) while testing on 1. 
    Do this process k times.
    Do Not randomize 
    
    Arguments:
        X  -- Train set
        y  -- Train set labels
    
    Keyword Arguments:
        k {number} -- k for the evaluation
        sigma {number} -- parameter for gaussian kernel
    
    Returns:
        error -- (sum of total mistakes for each fold)/(k)
    '''
    # TODO 
    print(np.shape(X),np.shape(y))
    mistakes = 0
    #y = np.reshape(y, (np.shape(y)[0], 1))
    n = np.shape(X)[0]
    for i in range(k):
        if i==0:
            xtrain = X[n//k:-1]
            xtest = X[:n//k]
            ytrain = y[n//k:-1]
            ytest = y[:n//k]
        else:
            xtrain = X[:(i*(n//k))]
            xtrain = np.concatenate((xtrain,X[(i+1)*(n//k):-1]))
            xtest = X[(i*(n//k)):(i+1)*(n//k)]
            ytrain = y[:(i*(n//k))]
            ytrain = np.concatenate((ytrain,y[(i+1)*(n//k):-1]))
            ytest = y[i*(n//k):(i+1)*(n//k)]
        clf = KernelLogistic(gaussian_kernel,sigma = sigma)
        clf.fit(xtrain,ytrain)
        y_predict = clf.predict(xtest) > 0.5
        wrong = np.sum(y_predict != ytest)
        mistakes = mistakes + wrong
    return mistakes/k
    # END TODO

if __name__ == '__main__':
    data = np.loadtxt("./data/dataset1.txt")
    X1 = data[:900,:2]
    Y1 = data[:900,2]

    clf = KernelLogistic(gaussian_kernel)
    clf.fit(X1, Y1)

    y_predict = clf.predict(data[900:,:2]) > 0.5
    correct = np.sum(y_predict == data[900:,2])
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    if correct > 92:
        marks = 1.0
    else:
        marks = 0
    print(f"You recieve {marks} for the fit function")

    errs = []
    sigmas = [0.5, 1, 2, 3, 4, 5, 6]
    for s in sigmas:  
      errs+=[(k_fold_cv(X1,Y1,sigma=s))]
    plt.plot(sigmas,errs)
    plt.xlabel('Sigma')
    plt.ylabel('Mistakes')
    plt.title('A plot of sigma v/s mistakes')
    plt.show()
