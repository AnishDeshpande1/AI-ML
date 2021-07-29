'''File contains the trainer class

Complete the functions train() which will train the network given the dataset and hyperparams, and the function __init__ to set your network topology for each dataset
'''
import numpy as np
import sys
import pickle

import nn

from util import *
from layers import *

class Trainer:
	def __init__(self,dataset_name):
		self.save_model = False
		if dataset_name == 'MNIST':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readMNIST()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 50
			self.epochs = 10
			self.lr = 0.01
			self.nn = nn.NeuralNetwork(np.shape(self.YTest)[1],self.lr)
			self.nn.addLayer(FullyConnectedLayer(784,16,'relu'))
			self.nn.addLayer(FullyConnectedLayer(16,16,'relu'))
			self.nn.addLayer(FullyConnectedLayer(16,np.shape(self.YTest)[1],'softmax'))


		if dataset_name == 'CIFAR10':
			
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCIFAR10()
			self.XTrain = self.XTrain[0:5000,:,:,:]
			self.XVal = self.XVal[0:1000,:,:,:]
			self.XTest = self.XTest[0:1000,:,:,:]
			self.YVal = self.YVal[0:1000,:]
			self.YTest = self.YTest[0:1000,:]
			self.YTrain = self.YTrain[0:5000,:]

			self.save_model = True
			self.model_name = "model.p"
						# Add your network topology along with other hyperparameters here
			self.batch_size = 50
			self.epochs = 40
			self.lr = 0.01
			self.nn = nn.NeuralNetwork(np.shape(self.YTest)[1],self.lr)
			k1 = (5,5)
			k2 = (4,4)
			numfilt = 32
			s1 = 3
			s2 = 2
			c = 10
			self.nn.addLayer(ConvolutionLayer(np.shape(self.XTest)[1:],k1,numfilt,s1,'relu'))
			self.nn.addLayer(AvgPoolingLayer((numfilt, c, c), k2, s2))
			self.nn.addLayer(FlattenLayer())
			self.nn.addLayer(FullyConnectedLayer(512,self.YTest.shape[1],'softmax'))

			# self.nn.addLayer(FullyConnectedLayer(3072,16,'relu'))
			# self.nn.addLayer(FullyConnectedLayer(16,16,'relu'))
			# self.nn.addLayer(FullyConnectedLayer(16,np.shape(self.YTest)[1],'softmax'))

		if dataset_name == 'XOR':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readXOR()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 50
			self.epochs = 100
			self.lr = 0.01
			self.nn = nn.NeuralNetwork(np.shape(self.YTest)[1],self.lr)
			self.nn.addLayer(FullyConnectedLayer(2,4,'relu'))
			self.nn.addLayer(FullyConnectedLayer(4,np.shape(self.YTest)[1],'softmax'))


		if dataset_name == 'circle':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCircle()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 50
			self.epochs = 50
			self.lr = 0.01
			self.nn = nn.NeuralNetwork(np.shape(self.YTest)[1],self.lr)
			self.nn.addLayer(FullyConnectedLayer(2,2,'relu'))
			self.nn.addLayer(FullyConnectedLayer(2,np.shape(self.YTest)[1],'softmax'))        
	def train(self, verbose=True):
		# Method for training the Neural Network
		# Input
		# trainX - A list of training input data to the neural network
		# trainY - Corresponding list of training data labels
		# validX - A list of validation input data to the neural network
		# validY - Corresponding list of validation data labels
		# printTrainStats - Print training loss and accuracy for each epoch
		# printValStats - Prints validation set accuracy after each epoch of training
		# saveModel - True -> Saves model in "modelName" file after each epoch of training
		# loadModel - True -> Loads model from "modelName" file before training
		# modelName - Name of the model from which the funtion loads and/or saves the neural net
		
		# The methods trains the weights and baises using the training data(trainX, trainY)
		# and evaluates the validation set accuracy after each epoch of training

		for epoch in range(self.epochs):
			# A Training Epoch
			if verbose:
				print("Epoch: ", epoch)

			# TODO
			# Shuffle the training data for the current epoch
			n = np.shape(self.XTrain)[0]
			indices = np.arange(n)
			np.random.shuffle(indices)
			newX = self.XTrain[indices]
			newY = self.YTrain[indices]
			# Initializing training loss and accuracy
			trainLoss = 0
			trainAcc = 0
			# Divide the training data into mini-batches
			bsize = self.batch_size
			if((float(n)/bsize)%1==0):
				numBatches=int(float(n)/bsize)
			else:
				numBatches = int(np.ceil(float(n)/bsize))
			for i in range(numBatches):
				miniX = np.array(newX[i*bsize:(i+1)*bsize])
				miniY = np.array(newY[i*bsize:(i+1)*bsize])
				# Calculate the activations after the feedforward pass
				activs = self.nn.feedforward(miniX)
				# Compute the loss  
				loss = self.nn.computeLoss(miniY,activs[-1])
				trainLoss+=loss
				# Calculate the training accuracy for the current batch
				predictions = oneHotEncodeY(np.argmax(activs[-1], axis = 1), self.nn.out_nodes)
				mini_acc = self.nn.computeAccuracy(miniY,predictions)
				trainAcc+=mini_acc
				# Backpropagation Pass to adjust weights and biases of the neural network
				self.nn.backpropagate(activs, miniY)
			# END TODO
			# Print Training loss and accuracy statistics
			trainAcc /= numBatches
			if verbose:
				print("Epoch ", epoch, " Training Loss=", trainLoss, " Training Accuracy=", trainAcc)
			
			if self.save_model:
				model = []
				for l in self.nn.layers:
					# print(type(l).__name__)
					if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer": 
						model.append(l.weights) 
						model.append(l.biases)
				pickle.dump(model,open(self.model_name,"wb"))
				print("Model Saved... ")

			# Estimate the prediction accuracy over validation data set
			if self.XVal is not None and self.YVal is not None and verbose:
				_, validAcc = self.nn.validate(self.XVal, self.YVal)
				print("Validation Set Accuracy: ", validAcc, "%")

		pred, acc = self.nn.validate(self.XTest, self.YTest)
		print('Test Accuracy ',acc)

