import numpy as np
def load_data1(file):
	'''
	Given a file, this function returns X, the regression features
	and Y, the output

	Args:
	filename - is a csv file with the format

	feature1,feature2, ... featureN,y
	0.12,0.31,1.33, ... ,5.32

	Returns:
	X - numpy array of shape (number of samples, number of features)
	Y - numpy array of shape (number of samples, 1)
	'''

	data = np.loadtxt(file, delimiter=',', skiprows=1)
	X = data[:, :-1]
	Y = data[:, -1:]

	return X, Y

def load_data2(file):
	'''
	Given a file, this function returns X, the features 
	and Y, the output

	Args:
	filename - is a csv file with the format

	feature1,feature2, ... featureN,y
	0.12,0.31,Yes, ... ,5.32

	Returns:
	X - numpy array of shape (number of samples, number of features)
	Y - numpy array of shape (number of samples, 1)
	'''
	data = np.loadtxt(file, delimiter=',', skiprows=1, dtype='str')
	X = data[:, :-1]
	Y = data[:, -1:].astype(float)

	return X, Y


def split_data(X, Y, train_ratio=0.8):
	'''
	Split data into train and test sets
	The first floor(train_ratio*n_sample) samples form the train set
	and the remaining the test set

	Args:
	X - numpy array of shape (n_samples, n_features)
	Y - numpy array of shape (n_samples, 1)
	train_ratio - fraction of samples to be used as training data

	Returns:
	X_train, Y_train, X_test, Y_test
	'''

	## TODO

	X_train = []
	Y_train = []
	X_test = []
	Y_test = []
	# test_seq = []
	# N = np.shape(X)[0]
	# train_len = int(np.floor(N * train_ratio))
	# random_seq = np.random.choice(N, train_len, replace=False)
	# test_seq_encode = np.zeros(N)
	# for a in random_seq:
	# 	X_train.append(X[a])
	# 	Y_train.append(Y[a])
	# 	test_seq_encode[a] = 1
	# for i in range(0,N):
	# 	if test_seq_encode[i] == 0:
	# 		test_seq.append(i)

	# for a in test_seq:
	# 	X_test.append(X[a])
	# 	Y_test.append(Y[a])
	# X_train = np.array(X_train)
	# X_test = np.array(X_test)
	# Y_train = np.array(Y_train)
	# Y_test = np.array(Y_test)
	N = np.shape(X)[0]
	train_len = int(np.floor(N * train_ratio))
	for i in range(N):
		if(i<train_len):
			X_train.append(X[i])
			Y_train.append(Y[i])
		else:
			X_test.append(X[i])
			Y_test.append(Y[i])
	## END TODO

	return X_train, Y_train, X_test, Y_test

def one_hot_encode(X, labels):
	'''
	Args:
	X - numpy array of shape (n_samples, 1) 
	labels - list of all possible labels for current category
	
	Returns:
	X in one hot encoded format (numpy array of shape (n_samples, n_labels))
	'''
	X.shape = (X.shape[0], 1)
	newX = np.zeros((X.shape[0], len(labels)))
	label_encoding = {}
	for i, l in enumerate(labels):
		label_encoding[l] = i
	for i in range(X.shape[0]):
		newX[i, label_encoding[X[i,0]]] = 1
	return newX

def normalize(X):
	'''
	Returns normalized X

	Args:
	X of shape (n_samples, 1)

	Returns:
	(X - mean(X))/std(X)
	'''
	## TODO
	# n_samples = np.shape(X)[0]
	# mean = 0.0
	# for i in range(n_samples):
	# 	mean = mean+float(X[i])
	# mean = mean / n_samples
	# # X = np.array(X)
	# # print(X)
	# # print(np.char.isnumeric(X))
	# # mean = np.mean(X)
	# std = 0.0
	# for i in range(n_samples):
	# 	std = std + (float(X[i])-mean)**2
	# std = std**0.5
	# #std = np.std(X)
	# for i in range(np.shape(X)[0]):
	# 	X[i] = (float(X[i])-mean)/std
	# X = np.array([float(x) for x in X])
	Y = np.zeros((np.shape(X)[0],1))
	for i in range(np.shape(X)[0]):
		Y[i]=float(X[i])
	Y = Y - np.mean(Y)
	Y = Y/np.std(Y)
	return Y
	## END TODO

def preprocess(X, Y):
	'''
	X - feature matrix; numpy array of shape (n_samples, n_features) 
	Y - outputs; numpy array of shape (n_samples, 1)

	Convert data X obtained from load_data2 to a usable format by gradient descent function
	Use one_hot_encode() to convert 

	NOTE 1: X has first column denote index of data point. Ignore that column
			and add constant 1 instead (for bias) 
	NOTE 2: For categorical string data, encode using one_hot_encode() and
			normalize the other features and Y using normalize()
	'''

	## TODO
	# final_X = []
	# for i in range(np.shape(X)[0]):
	# 	X[i][0]=1
	# #print(X)
	# temp = X[:,0]
	# final_X.append(temp)
	# for i in range(1,np.shape(X)[1]):
	# 	X_num = np.char.isnumeric(X[:,i])
	# 	flag = False
	# 	for j in range(np.shape(X)[0]):
	# 		if X_num[i] == False:
	# 			flag = True
	# 			break
	# 	if flag:
	# 	    label = []
	# 	    for j in range(np.shape(X)[0]):
	# 	        if X[j,i] not in label:
	# 	            label.append(X[j,i])
	# 	    new_X = one_hot_encode(X[:,i], label)
	# 	    for cols in new_X:
	# 	        final_X.append(cols)
	# 	else:
	# 	    new_X = normalize(X[:,i])
	# 	    final_X.append(new_X)
	# #Y = normalize(Y)   
	# #print(final_X)
	# final_X = np.transpose(final_X)
	# fin = pd.DataFrame(final_X)
	# print(fin.head())


	n_samples = np.shape(X)[0]
	n_features = np.shape(X)[1]
	num_cols_to_add = 0
	categorical_cols = []
	for i in range(n_features):
		# X_num = np.char.isnumeric(np.array(X[:,i]))
		# flag = False
		# for j in range(n_samples):
		# 	if X_num[j] == False:
		# 		flag = True
		# 		break
		flag = False
		for j in range(n_samples):
		    try:
    			val = int(X[j,i])
		    except:
    			flag = True
    			break


		if flag:
			categorical_cols.append(i)
			label = []
			for j in range(n_samples):
			    if X[j,i] not in label:
	        		label.append(X[j,i])
			t = len(label)
			num_cols_to_add = num_cols_to_add + t - 1
	final_X = np.zeros((n_samples,n_features+num_cols_to_add))
	col_pointer = 1
	final_X[:,0] = 1.0
	for i in range(1,n_features):
		if i not in categorical_cols:
			temp = normalize(X[:,i])
			for j in range(n_samples):
				final_X[j][col_pointer]=float(temp[j])
			col_pointer=col_pointer+1
		else:
			label = []
			for j in range(n_samples):
				if X[j][i] not in label:
					label.append(X[j][i])
			new_X = one_hot_encode(X[:,i], label)
			for k in range(len(label)):
				for p in range(n_samples):
					final_X[p][col_pointer]=float(new_X[p][k])
				col_pointer = col_pointer + 1
	#print(final_X.shape)
	Y = normalize(Y)
	return final_X,Y

	## END TODO

