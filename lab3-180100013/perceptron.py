import numpy as np
import argparse

def get_data(dataset):
	datasets = ['D1', 'D2']
	assert dataset in datasets, "Dataset {dataset} not supported. Supported datasets {datasets}"
	X_train = np.loadtxt(f'data/{dataset}/training_data')
	Y_train = np.loadtxt(f'data/{dataset}/training_labels', dtype=int)
	X_test = np.loadtxt(f'data/{dataset}/test_data')
	Y_test = np.loadtxt(f'data/{dataset}/test_labels', dtype=int)

	return X_train, Y_train, X_test, Y_test

def get_features(x):
	'''
	Input:
	x - numpy array of shape (2500, )

	Output:
	features - numpy array of shape (D, ) with D <= 5
	'''
	### TODO
	features = np.zeros((5,))
	is_star = False
	y = x
	
	y = y.reshape((50,50))
	for i in range(50):
		num_changes = 0
		for j in range(1,50):
			if(y[i,j] != y[i,j-1]):
				num_changes = num_changes + 1
			if(num_changes >= 3):
				is_star = True
				break

	counts = []
	on_edge = 0
	for i in range(5, 45):
		for j in range(5,45):
			patch = y[i-5:i+5,j-5:j+5]
			if(patch[5,5] == 0):
				continue
			if(np.sum(patch)>=30):
				continue
			lis = []
			lis.append(patch[3,3]==patch[7,7]==1)
			lis.append(patch[3,4]==patch[7,6]==1)
			lis.append(patch[3,5]==patch[7,5]==1)
			lis.append(patch[3,6]==patch[7,4]==1)
			lis.append(patch[3,7]==patch[7,3]==1)
			lis.append(patch[4,3]==patch[6,7]==1)
			lis.append(patch[5,3]==patch[5,7]==1)
			lis.append(patch[6,3]==patch[4,7]==1)
			if(True in lis):
				on_edge = on_edge + 1
				continue
			counts.append(np.sum(patch))
	if(len(counts)==0):
		features[0] = is_star
		features[1] = -1
		features[2] = -1
		features[3] = -1
		features[4] = -1
	else:
		features[0] = is_star
		features[1] = np.mean(counts)
		features[2] = np.max(counts)
		features[3] = on_edge
		features[4] = np.min(counts)
	if(is_star):
		features[1]=0

	#print(on_edge)
	return features
	### END TODO

class Perceptron():
	def __init__(self, C, D):
		'''
		C - number of classes
		D - number of features
		'''
		self.C = C
		self.weights = np.zeros((C, D))
		
	def pred(self, x):
		'''
		x - numpy array of shape (D,)
		'''
		### TODO: Return predicted class for x
		arr_pred = np.dot(self.weights, x)
		prediction = np.argmax(arr_pred)
		return prediction
		### END TODO

	def train(self, X, Y, max_iter=7):
		for iter in range(max_iter):
			for i in range(X.shape[0]):
				### TODO: Update weights
				pr = self.pred(X[i])
				if(pr != Y[i]):
				    self.weights[pr] = self.weights[pr] - X[i]
				    self.weights[Y[i]] = self.weights[Y[i]] + X[i]
				### END TODO
			#print(f'Train Accuracy at iter {iter} = {self.eval(X, Y)}')

	def eval(self, X, Y):
		n_samples = X.shape[0]
		correct = 0
		for i in range(X.shape[0]):
			if self.pred(X[i]) == Y[i]:
				correct += 1
		return correct/n_samples

if __name__ == '__main__':
	X_train, Y_train, X_test, Y_test = get_data('D2')
	X_train = np.array([get_features(x) for x in X_train])
	X_test = np.array([get_features(x) for x in X_test])

	C = max(np.max(Y_train), np.max(Y_test))+1
	D = X_train.shape[1]

	perceptron = Perceptron(C, D)

	perceptron.train(X_train, Y_train)
	acc = perceptron.eval(X_test, Y_test)
	print(f'Test Accuracy: {acc}')
