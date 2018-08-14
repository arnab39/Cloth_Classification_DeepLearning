import numpy as np
import data_loader as dl
import module as md



def cross_entropy_error(a2, y):
	temp = -1*y*np.log(a2)
	return temp.sum()

'''Implement mini-batch SGD here'''


#hyperparameters of the model
learning_rate = 0.5
epoch = 40
batch_size = 60
output_classes=10
lamda= 0.0001

#Loading Data
data = dl.DataLoader(batch_size)
X_train,Y_train = data.load_data('train')

num_batches = X_train.shape[0]/batch_size

#declaring the instance of class neyral network
nn = md.NN(X_train.shape[1],output_classes,lamda)

for i in range(epoch):
	#To create batches
	X_t,Y_t = data.create_batches(X_train,Y_train)	
	for j in range(num_batches):
		x = X_t[j,:,:]
		Y = Y_t[j,:]
		y = np.zeros([Y.shape[0], output_classes])
		y[np.arange(Y.shape[0]), Y] = 1

		# Forward pass
		a1, a2 = nn.forward(x)

		#Printing the loss
		if(j%1000==0):
			print "\nEpoch:"+str(i)+" ---- Loss:" + str(cross_entropy_error(a2, y))

		# Backpropagation and weight update
		nn.backward(x,a1,a2,y,learning_rate)

X_train,Y_train = data.load_data('train')
a1, a2 = nn.forward(X_train)
Y_pred=np.argmax(a2, axis = 1)
accuracy = np.mean((Y_pred == Y_train)) * 100.0
print "\nTraining accuracy: %lf%%" % accuracy

#For finding the test accuracy
X_test,Y_test = data.load_data('test')
a1, a2 = nn.forward(X_test)
Y_pred=np.argmax(a2, axis = 1)
accuracy = np.mean((Y_pred == Y_test)) * 100.0
print "\nTest accuracy: %lf%%" % accuracy