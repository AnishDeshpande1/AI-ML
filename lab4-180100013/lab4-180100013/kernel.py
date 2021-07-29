import numpy as np 

def linear_kernel(X,Y,sigma=None):
	'''Returns the gram matrix for a linear kernel
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma - dummy argment, don't use
	Return:
		K - numpy array of size n x m
	''' 
	# TODO 
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE
	return np.array(np.matmul(X, np.transpose(Y)))
	pass
	# END TODO

def gaussian_kernel(X,Y,sigma=0.1):
	'''Returns the gram matrix for a rbf
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma - The sigma value for kernel
	Return:
		K - numpy array of size n x m
	'''
	# TODO
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE 
	n = np.shape(X)[0]
	m = np.shape(Y)[0]
	XY = np.array(np.matmul(X, np.transpose(Y)))
	X2 = np.array(np.matmul(X, np.transpose(X)))
	Y2 = np.array(np.matmul(Y, np.transpose(Y)))
	X2 = np.diag(X2).reshape((n,1))
	Y2 = np.diag(Y2).reshape((m,1))
	In = np.ones((n,1))
	Im = np.ones((m,1))
	X2 = np.hstack((X2, In))
	Y2 = np.hstack((Im, Y2))
	Y2 = np.transpose(Y2)
	X2Y2 = np.matmul(X2,Y2)
	ANS = X2Y2 - 2*XY
	ANS = np.exp((-1/(2*sigma*sigma))*ANS)
	return np.array(ANS)
	# END TODO

def my_kernel(X,Y,sigma):
	'''Returns the gram matrix for your designed kernel
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma- dummy argment, don't use
	Return:
		K - numpy array of size n x m
	''' 
	# TODO
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE 
	A = np.matmul(X, np.transpose(Y))+1
	B = np.multiply(A,A)
	return np.multiply(B,B)
	# END TODO
