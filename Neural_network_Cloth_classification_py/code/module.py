import numpy as np

np.random.seed(1)

#function to implement relu
def relu(x):
	y=x
	y[x<0]=0
	return y

#function to implement softmax
def softmax(x):
	exp_scores = np.exp(x)
	return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

#function to implement derivative of relu
def diff_relu(x):
	y=np.zeros(x.shape)
	y[x>0]=1
	return y

#class of neural network
class NN(object):
    def __init__(self,input_shape,output_shape,lamda):
		self.layer_1_nodes = (input_shape+output_shape)/2
		self.lamda = lamda

		# Generate weights with value between -1 to 1 so that mean is overall 0
		self.w_1 = 0.4 * np.random.rand(input_shape, self.layer_1_nodes) - 0.2
		self.w_2 = 0.4 * np.random.rand(self.layer_1_nodes, output_shape) - 0.2

    def forward(self,Input):
		x1 = np.dot(Input, self.w_1)
		a1 = relu(x1)
		x2 = np.dot(a1, self.w_2)
		a2 = softmax(x2)
		return a1,a2
    	
    def backward(self,x,a1,a2,y,eta=0.5):
		delta_2 = a2 - y
		dW2 = np.dot(a1.T, delta_2)/x.shape[0]

		delta_1 = np.dot(delta_2, self.w_2.T)
		da = diff_relu(a1)
		delta_1 *= da
		dW1 = np.dot(x.T, delta_1)/x.shape[0]

		#weight update
		self.w_1 += -eta*(dW1 + 2*self.lamda*self.w_1)
		self.w_2 += -eta*(dW2 + 2*self.lamda*self.w_2)