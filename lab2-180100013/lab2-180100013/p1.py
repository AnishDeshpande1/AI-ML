import numpy as np
import matplotlib.pyplot as plt
from utils import load_data2, split_data, preprocess, normalize

np.random.seed(337)


def mse(X, Y, W):
    """
    Compute mean squared error between predictions and true y values

    Args:
    X - numpy array of shape (n_samples, n_features)
    Y - numpy array of shape (n_samples, 1)
    W - numpy array of shape (n_features, 1)
    """

    # TODO
    mse = np.sum((Y-np.dot(X,W))**2)/(2*np.shape(X)[0])
    # END TODO

    return mse


def ista(X_train, Y_train, X_test, Y_test, _lambda=0.1, lr=0.015, max_iter=10000):
    """
    Iterative Soft-thresholding Algorithm for LASSO
    """
    train_mses = []
    test_mses = []

    # TODO: Initialize W using using random normal
    n_samples = np.shape(X_train)[0]
    n_features = np.shape(X_train)[1]
    W = np.array(np.random.normal(0,1,n_features))
    W = W.reshape((n_features,1))
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    # END TODO

    for i in range(max_iter):
        # TODO: Compute train and test MSE
        #print(i)
        train_mse = mse(X_train,Y_train,W) #+ _lambda*np.sum(abs(W))
        test_mse = mse(X_test,Y_test,W) #+ _lambda*np.sum(abs(W))
        # END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        # TODO: Update w and b using a single step of ISTA. You are not allowed to use loops here.
        Wold = W
        G = np.dot(np.transpose(X_train),np.matmul(X_train,W)-Y_train)/n_samples
        W = (W - lr*G)
        np.place(W,abs(W)<_lambda*lr,[0])
        np.where(W>_lambda*lr,W-_lambda*lr,W)
        np.where(W<-_lambda*lr,W+_lambda*lr,W)
        # END TODO

        # TODO: Stop the algorithm if the norm between previous W and current W falls below 1e-4
        if(np.linalg.norm(W-Wold)<0.0001):
            break
        # End TODO

    return W, train_mses, test_mses


if __name__ == '__main__':
    # Load and split data
    X, Y = load_data2('data2.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test)

    # TODO: Your code for plots required in Problem 1.2(b) and 1.2(c)
    # lamb = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,3,4,5,6]
    # x1 = []
    # x2 = []
    # for k in range(len(lamb)):
    # 	W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test,_lambda=lamb[k],lr=0.001, max_iter=10000)
    # 	x1.append(train_mses_ista[-1])
    # 	x2.append(test_mses_ista[-1])
    # print(x1,x2)
    # plt.figure(figsize=(4,4))
    # #print(train_mses_ista[-1], test_mses_ista[-1])
    # plt.plot(x1)
    # plt.plot(x2)
    # plt.legend(['Train MSE', 'Test MSE'])
    # plt.xlabel('lambda')
    # plt.ylabel('MSE')
    # plt.show()
    plt.figure(figsize=(4,4))
    plt.scatter(np.arange(len(W)),W)
    plt.show()
    # End TODO
