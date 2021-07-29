import numpy as np 
from matplotlib import pyplot as plt
import argparse

from utils import *
from p1 import mse

## ONLY CHANGE CODE BETWEEN TODO and END TODO
def prepare_data(X,degree):
    '''
    X is a numpy matrix of size (n x 1)
    return a numpy matrix of size (n x (degree+1)), which contains higher order terms
    '''
    # TODO
    n = np.shape(X)[0]
    X_new = np.ones((n,degree+1))
    for i in range(n):
        for j in range(degree+1):
            X_new[i][j]=X[i]**j
    return X_new
    # End TODO
    return X 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Problem 4')
    parser.add_argument('--degree', type=int, default=3,
                    help='Degree of polynomial to use')
    args = parser.parse_args()
    np.random.seed(42)
    degree = args.degree

    X_train, Y_train = load_data1('data3_train.csv')
    Y_train = Y_train/20
    X_test, Y_test   = load_data1('data3_test.csv')
    Y_test = Y_test/20

    X_train = prepare_data(X_train,degree)
    indices_0 = np.random.choice(np.arange(200),40,replace=False)
    indices_1 = np.random.choice(np.arange(200),40,replace=False)
    indices_2 = np.random.choice(np.arange(200),40,replace=False)
    indices_3 = np.random.choice(np.arange(200),40,replace=False)

    ## TODO - compute each fold using indices above, compute weights using OLS
    X_0 = X_train[indices_0]
    Y_0 = Y_train[indices_0]
    X_1 = X_train[indices_1]
    Y_1 = Y_train[indices_1]
    X_2 = X_train[indices_2]
    Y_2 = Y_train[indices_2]
    X_3 = X_train[indices_3]
    Y_3 = Y_train[indices_3]
    W_0 = np.dot(np.linalg.inv(X_0.T @ X_0),X_0.T @ Y_0)
    W_1 = np.dot(np.linalg.inv(np.matmul(X_1.T,X_1)),np.matmul(X_1.T,Y_1))
    W_2 = np.dot(np.linalg.inv(np.matmul(X_2.T,X_2)),np.matmul(X_2.T,Y_2))
    W_3 = np.dot(np.linalg.inv(np.matmul(X_3.T,X_3)),np.matmul(X_3.T,Y_3))

    # train_0=[]
    # test_0=[]
    # train_1=[]
    # test_1=[]
    # train_2=[]
    # test_2=[]
    # train_3=[]
    # test_3=[]
    # for i in range(1,7):
    #     X_test = prepare_data(X_test,degree)
    #     train_mse_0 = mse(X_0,Y_0,W_0)
    #     train_mse_1 = mse(X_1,Y_1,W_1)
    #     train_mse_2 = mse(X_2,Y_2,W_2)
    #     train_mse_3 = mse(X_3,Y_3,W_3)
    #     test_mse_0  = mse(X_test, Y_test, W_0)
    #     test_mse_1  = mse(X_test, Y_test, W_1)
    #     test_mse_2  = mse(X_test, Y_test, W_2)
    #     test_mse_3  = mse(X_test, Y_test, W_3)
    #     test_0.append(test_mse_0)
    #     train_0.append(train_mse_0)
    #     test_1.append(test_mse_1)
    #     train_1.append(train_mse_1)
    #     test_2.append(test_mse_2)
    #     train_2.append(train_mse_2)
    #     test_3.append(test_mse_3)
    #     train_3.append(train_mse_3)
    # plt.figure(figsize=(4,4))
    # plt.plot(train_0)
    # plt.plot(test_0)
    # plt.show()
    ## END TODO


    X_test = prepare_data(X_test,degree)

    train_mse_0 = mse(X_0,Y_0,W_0)
    train_mse_1 = mse(X_1,Y_1,W_1)
    train_mse_2 = mse(X_2,Y_2,W_2)
    train_mse_3 = mse(X_3,Y_3,W_3)
    test_mse_0  = mse(X_test, Y_test, W_0)
    test_mse_1  = mse(X_test, Y_test, W_1)
    test_mse_2  = mse(X_test, Y_test, W_2)
    test_mse_3  = mse(X_test, Y_test, W_3)

    X_lin = np.linspace(X_train[:,1].min(),X_train[:,1].max()).reshape((50,1))
    X_lin = prepare_data(X_lin,degree)
    print(f'Test Error 1: %.4f Test Error 2: %.4f Test Error 3: %.4f test E 4: %.4f'%(test_mse_0,test_mse_1,test_mse_2,test_mse_3))
    plt.scatter(X_train[:,1],Y_train,color='orange')
    plt.plot(X_lin[:,1],X_lin @ W_0, c='g')
    plt.plot(X_lin[:,1],X_lin @ W_1, c='r')
    plt.plot(X_lin[:,1],X_lin @ W_2, c='b')
    plt.plot(X_lin[:,1],X_lin @ W_3, color='purple')
    plt.plot(X_lin[:,1],X_lin @(W_1+W_2+W_3+W_0)/4, color='black')
    plt.show()
    plt.figure(figsize=(4,4))
    