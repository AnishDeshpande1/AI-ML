'''This file contains the implementations of the layers required by your neural network

For each layer you need to implement the forward and backward pass. You can add helper functions if you need, or have extra variables in the init function

Each layer is of the form - 
class Layer():
    def __init__(args):
        *Initializes stuff*

    def forward(self,X):
        # X is of shape n x (size), where (size) depends on layer
        
        # Do some computations
        # Store activations_current
        return X

    def backward(self, lr, activation_prev, delta):
        """
        # lr - learning rate
        # delta - del_error / del_activations_current
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        """
        # Compute gradients wrt trainable parameters
        # Update parameters
        # Compute gradient wrt input to this layer
        # Return del_error/del_activation_prev
'''
import numpy as np

class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation   # string having values 'relu' or 'softmax', activation function to use
        # Stores the outgoing summation of weights * feautres 
        self.data = None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))    
        self.biases = np.random.normal(0,0.1, (1, out_nodes))
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary 

    def forwardpass(self, X):
        '''
                
        Arguments:
            X  -- activation matrix       :[n X self.in_nodes]
        Return:
            activation matrix      :[n X self.out_nodes]
        '''
        # TODO
        Y = np.matmul(X,self.weights) + self.biases
        if self.activation == 'relu':
        	self.data = relu_of_X(Y)
        	return self.data
            #raise NotImplementedError
        elif self.activation == 'softmax':
        	self.data = softmax_of_X(Y)
        	return self.data
            #raise NotImplementedError

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        pass
        # END TODO      
    def backwardpass(self, lr, activation_prev, delta):
        '''
        # lr - learning rate
        # delta - del_error / del_activations_current  : 
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        '''

        # TODO 
        del_error = None
        n = activation_prev.shape[0]
        if self.activation == 'relu':
        	del_error = gradient_relu_of_X(self.data,delta)
            #raise NotImplementedError
        elif self.activation == 'softmax':
        	del_error = gradient_softmax_of_X(self.data,delta)
            #raise NotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        p_del = del_error@np.transpose(self.weights)
        dW = np.matmul(np.transpose(np.array(activation_prev)),del_error)
        #dW = np.mean(activation_prev[:,:,np.newaxis]@del_error[:,np.newaxis,:],axis = 0)
        dW = dW/np.shape(activation_prev)[0]
        #print(np.shape(dW), "dW")
        #print(np.shape(activation_prev), "Activ_prev")
        #print(np.shape(del_error), "grad_of_")
        self.weights = self.weights - lr*dW
        #l = np.shape(del_error)[0]
        dB = np.sum(del_error,axis = 0)
        dB = dB/np.shape(activation_prev)[0]
        dB = np.reshape(dB,np.shape(self.biases))
        self.biases = self.biases - lr*dB
        return p_del
        pass
        # END TODO
class ConvolutionLayer:
    def __init__(self, in_channels, filter_size, numfilters, stride, activation):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for convolution layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer
        # numfilters  - number of feature maps (denoting output depth)
        # stride      - stride to used during convolution forward pass
        # activation  - can be relu or None
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride
        self.activation = activation
        self.out_depth = numfilters
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

        # Stores the outgoing summation of weights * feautres 
        self.data = None
        
        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))   
        self.biases = np.random.normal(0,0.1,self.out_depth)
        

    def forwardpass(self, X):
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]

        # TODO
        if self.activation == 'relu':
        	op_r = self.out_row
        	op_c = self.out_col
        	f_r = self.filter_row
        	f_c = self.filter_col
        	n = np.shape(X)[0]
        	stride = self.stride 
        	OUTPUT = np.zeros([n,self.out_depth,self.out_row, self.out_col])
        	OUTPUT = OUTPUT + self.biases[np.newaxis,:,np.newaxis,np.newaxis]
        	for i in range(op_r):
        		for j in range(op_c):
        			patch = X[:,:,i*stride:i*stride+f_r,j*stride:j*stride+f_c]
        			patch = np.reshape(patch, (n,self.in_depth,-1))
        			wts = np.reshape(self.weights,(self.out_depth, self.in_depth,-1))
        			OUTPUT[:,:,i,j] = np.sum(np.einsum('nid,oid->nod',patch,wts),axis = -1)
        	shp = np.shape(OUTPUT)
        	self.data = np.reshape(relu_of_X(np.reshape(OUTPUT, (n,-1))),shp)
        	#print(np.shape(self.data))
        	return self.data
            #raise NotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        
        ###############################################
        # END TODO
    def backwardpass(self, lr, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # Update self.weights and self.biases for this layer by backpropagation
        # TODO
        n = np.shape(activation_prev)[0]
        stride = self.stride
        in_depth = self.in_depth
        in_row = self.in_row
        in_col = self.in_col
        op_r = self.out_row
       	op_c = self.out_col
       	f_r = self.filter_row
        f_c = self.filter_col
        wts = self.weights
        shp = np.shape(wts)
        o_del = np.zeros((n,in_depth,in_row,in_col))
        #print(np.shape(delta), " delta")
        ###############################################
        if self.activation == 'relu':
        	inp_delta = gradient_relu_of_X(self.data, delta)
        	for i in range(op_r):
        		for j in range(op_c):
        			patch = np.einsum('no,oirc->nirc', inp_delta[:, :, i, j], self.weights)
        			o_del[:,:,i*stride:i*stride+f_r, j*stride:j*stride+f_c] += patch 
        	#print(np.shape(self.data), "self.data in backpass CNN")
        	temp = inp_delta.reshape([n, self.out_depth, -1])
        	self.biases -= lr*np.sum(np.sum(temp, axis=-1), axis=0)/n
        	for i in range(op_r):
        		for j in range(op_c):
        			self.weights -= lr * np.sum(np.einsum('no,nirc->noirc', inp_delta[:, :, i, j], activation_prev[:, :, i*stride:i*stride+f_r, j*stride:j*stride+f_c]), axis=0)/n
        	return o_del
            # raise NotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################

        # END TODO
    
class AvgPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        stride = self.stride
        op_r = self.out_row
        op_c = self.out_col
        f_r = self.filter_row
        f_c = self.filter_col
        n = np.shape(X)[0]
        kernel = np.ones([f_r,f_c])/(f_r*f_c)
        activs = np.zeros([n,self.out_depth, op_r,op_c])
        for i in range(op_r):
        	for j in range(op_c):
        		patch = X[:,:,i*stride:i*stride+f_r,j*stride:j*stride+f_c]
        		activs[:,:,i,j] = np.sum(np.sum(np.multiply(patch,kernel[np.newaxis,np.newaxis,:,:]),axis=-1),axis=-1)
        return activs
        #pass
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        stride = self.stride
        op_r = self.out_row
        op_c = self.out_col
        f_r = self.filter_row
        f_c = self.filter_col
        n = np.shape(activation_prev)[0]
        kernel = np.ones([f_r,f_c])/(f_r*f_c)
        o_del = np.zeros([n, self.in_depth, self.in_row, self.in_col])
        for i in range(op_r):
        	for j in range(op_c):
        		layer = np.multiply(delta[:, :, i:i+1, j:j+1], kernel[np.newaxis, np.newaxis, :, :])
        		o_del[:, :, i*stride:i*stride+f_r, j*stride:j*stride+f_c] += layer
        return o_del
        # TODO
        #pass
        # END TODO
        ###############################################



class MaxPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        stride = self.stride
        op_r = self.out_row
        op_c = self.out_col
       	f_r = self.filter_row
       	f_c = self.filter_col
       	n = np.shape(X)[0]
       	activs = np.zeros([n,self.out_depth, op_r,op_c])
        for i in range(op_r):
        	for j in range(op_c):
        		patch = X[:,:,i*stride:i*stride+f_r,j*stride:j*stride+f_c]
        		activs[:,:,i,j] = np.max(np.max(patch,axis=-1),axis=-1)
        return activs
        #pass
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        stride = self.stride
        shp = np.shape(activation_prev)
        op_r = self.out_row
        op_c = self.out_col
       	f_r = self.filter_row
       	f_c = self.filter_col
       	n = np.shape(activation_prev)[0]
       	back_del = np.zeros(shp)
       	for i in range(op_r):
       		for j in range(op_c):
       			patch = activation_prev[:,:,i*stride:i*stride+f_r,j*stride:j*stride+f_c]
       			dep = np.shape(patch)[1]
       			in_r = np.shape(patch)[2]
       			in_c = np.shape(patch)[3]
       			kernel = np.zeros_like(patch)
       			index = np.argmax(np.reshape(patch,(n,dep,-1)),axis=2)
       			ni,depi = np.indices((n,dep))
       			kernel.reshape(n,dep,in_r*in_c)[ni,depi,index]=1
       			back_del[:,:,i*stride:i*stride+f_r,j*stride:j*stride+f_c]+=np.multiply(delta[:,:,i:i+1,j:j+1],kernel)
       	return back_del
        #pass
        # END TODO
        ###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
    	# TODO
    	s = np.shape(X)
    	Y = np.reshape(X,(s[0],s[1]*s[2]*s[3]))
    	self.in_batch = s[0]
    	self.r = s[1]
    	self.c = s[2]
    	self.k = s[3]
    	return Y
        # print(X.shape)
    def backwardpass(self, lr, activation_prev, delta):
    	Y = delta.reshape(self.in_batch, self.r, self.c, self.k)
    	return Y
        # END TODO

# Function for the activation and its derivative
def relu_of_X(X):

    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu
    # TODO
    return np.where(X<=0,0,X)
    #raise NotImplementedError
    # END TODO 
    
def gradient_relu_of_X(X, delta):
    # Input
    # Note that these shapes are specified for FullyConnectedLayers, the function also needs to work with ConvolutionalLayer
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu amd during backwardpass
    
    # TODO
    Y = np.where(X>0,1,0)
    return np.multiply(Y,delta)
    #raise NotImplementedError
    # END TODO

def softmax_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax
    
    # TODO
    n = np.shape(X)[0]
    Y = np.exp(X)
    Z = np.sum(Y,axis=1)
    Z = np.reshape(Z,(n,1))
    Y = Y/Z
    return Y
    #raise NotImplementedError
    # END TODO  
def gradient_softmax_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax amd during backwardpass
    # Hint: You might need to compute Jacobian first

    # TODO

    I = np.eye(X.shape[1])
    Y = softmax_of_X(X)
    Z1 = np.multiply(Y,1-Y)
    M = I[np.newaxis, :, :] - X[:, :, np.newaxis]
    Z2 = np.multiply(-Y,Y)
    J = np.multiply(X[:, :, np.newaxis], np.swapaxes(M, 1, 2))
    #print(np.shape(delta[:,np.newaxis,:]), "delta")
    #print(np.shape(np.swapaxes(jacob, 1, 2)))
    return (np.matmul(delta[:, np.newaxis, :], np.swapaxes(J, 1, 2))).squeeze(axis=1)
        # return np.multiply(Z,delta)
    #raise NotImplementedError
    # END TODO
