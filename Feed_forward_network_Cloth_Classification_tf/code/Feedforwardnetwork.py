from __future__ import print_function

import numpy as np
import data_loader as dl
import tensorflow as tf
import os
from sklearn import linear_model
import argparse


#Parsing argument definition
parser = argparse.ArgumentParser(description=' Assignment 3 ')
parser.add_argument('--layers', type=int, help='layer number')
group = parser.add_mutually_exclusive_group()
group.add_argument('--train', action='store_true')
group.add_argument('--test', action='store_true')
args = parser.parse_args()

#Hyperparameters
learning_rate = 0.001
num_steps = 100
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 400 # 1st layer number of neurons
n_hidden_2 = 400 # 2nd layer number of neurons
n_hidden_3 = 400 # 2nd layer number of neurons

num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
validation_set=10000

#Loading Data
data = dl.DataLoader(batch_size)

#Training data
X_train,Y_train = data.load_data('train')

Y_tr = np.zeros([Y_train.shape[0], num_classes])
Y_tr[np.arange(Y_train.shape[0]), Y_train] = 1

#Testing data
X_test,Y_t = data.load_data('test')

Y_test = np.zeros([Y_t.shape[0], num_classes])
Y_test[np.arange(Y_t.shape[0]), Y_t] = 1


#Hyperparameter
N=X_train.shape[0]
num_batches = (N-validation_set)/batch_size

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

print("\n\nASSIGNMENT 3\n")

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def lrelu(x, leak=0.0, name="lrelu"):
    return tf.maximum(x, leak * x)

def softmax(x):
	m=tf.reduce_max(x,axis=-1, keep_dims=True)
	return tf.exp(x-m) / tf.reduce_sum(tf.exp(x-m), axis=-1, keep_dims=True)

def cross_entropy(out,Y):
    cost=-tf.reduce_sum(tf.multiply(Y,tf.where(tf.greater(out,0),tf.log(out),tf.zeros_like(out))))
    print(cost.shape)
    return cost

saver = tf.train.Saver()

def save(sess,checkpoint_dir):
    model_name = "model.ckpt"
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    print("\nReading checkpoints...")
    saver.save(sess, os.path.join(checkpoint_dir, model_name))

def load(sess,checkpoint_dir):
    print("\nReading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
      return True
    else:
      return False


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = lrelu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = lrelu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Hidden fully connected layer with 256 neurons
    layer_3 = lrelu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer,layer_1,layer_2,layer_3


# Construct model
logits,l_1,l_2,l_3 = neural_net(X)
prediction = softmax(logits)


# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

#loss_op=tf.reduce_mean(cross_entropy(prediction,Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def train():
	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()
	print("\n")
	# Start training
	with tf.Session() as sess:

	  # Run the initializer
	  sess.run(init)
	  best_val_acc=0

	  for step in range(1, num_steps+1):
	    X_t,Y_t = data.create_batches(X_train[:-validation_set],Y_train[:-validation_set])
	    X_tr,Y_trt = X_train[:-validation_set],Y_train[:-validation_set]
	    Y_tr = np.zeros([Y_trt.shape[0], num_classes])
	    Y_tr[np.arange(Y_trt.shape[0]), Y_trt] = 1
	    X_v,Y_vt = X_train[N-validation_set:],Y_train[N-validation_set:]
	    Y_v = np.zeros([Y_vt.shape[0], num_classes])
	    Y_v[np.arange(Y_vt.shape[0]), Y_vt] = 1	
	    for j in range(num_batches):
	      batch_x=X_t[j,:,:]
	      Yt = Y_t[j,:]
	      batch_y = np.zeros([Yt.shape[0], num_classes])
	      batch_y[np.arange(Yt.shape[0]), Yt] = 1
	      
	      # Run optimization op (backprop)
	      sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

	    val_acc=sess.run(accuracy, feed_dict={X: X_v,Y: Y_v})
	    if best_val_acc<val_acc:
	    	best_val_acc=val_acc
	    	save(sess,"./weights")


	    if step % display_step == 0 or step == 1:
	      # Print last batch loss and validation accuracy
	      loss = sess.run(loss_op, feed_dict={X: X_tr,Y: Y_tr})
	      print("Step " + str(step) + ", Minibatch Loss= " + \
	              "{:.4f}".format(loss) + ", Current Validation Accuracy= " + \
	              "{:.3f}".format(val_acc)+ ", Best Validation Accuracy= " + \
	              "{:.3f}".format(best_val_acc))

	  print("\nOptimization Finished!")


def test():
	#For finding the test accuracy
	init = tf.global_variables_initializer()

	# Start training
	with tf.Session() as sess:

	  # Run the initializer
	  sess.run(init)
	  load(sess,"./weights")
	  loss,logi,pred,acc=sess.run([loss_op,logits,prediction,accuracy], feed_dict={X: X_test,Y: Y_test})
	  # Calculate accuracy for MNIST test images
	  print("\nTesting Accuracy:", acc)
	        

def reg_layer1():
	init = tf.global_variables_initializer()

	# Start training
	with tf.Session() as sess:

	  # Run the initializer
	  sess.run(init)
	  load(sess,"./weights")
	  # Calculate accuracy for MNIST test images
	  l1, l2, l3 = sess.run([l_1, l_2, l_3], feed_dict={X: X_train,Y: Y_tr})

	  l1_t, l2_t, l3_t = sess.run([l_1, l_2, l_3], feed_dict={X: X_test,Y: Y_test})

	  logreg = linear_model.LogisticRegression(C=1e5)
	  
	  logreg.fit(l1, Y_train)
	  acc=logreg.score(l1_t, Y_t)
	  print("\nRegression based classifier accuracy with features from Layer3:",acc)

def reg_layer2():
	init = tf.global_variables_initializer()

	# Start training
	with tf.Session() as sess:

	  # Run the initializer
	  sess.run(init)
	  load(sess,"./weights")
	  # Calculate accuracy for MNIST test images
	  l1, l2, l3 = sess.run([l_1, l_2, l_3], feed_dict={X: X_train,Y: Y_tr})

	  l1_t, l2_t, l3_t = sess.run([l_1, l_2, l_3], feed_dict={X: X_test,Y: Y_test})

	  logreg = linear_model.LogisticRegression(C=1e5)

	  logreg.fit(l2, Y_train)
	  acc=logreg.score(l2_t, Y_t)
	  print("\nRegression based classifier accuracy with features from Layer2:",acc)

def reg_layer3():
	init = tf.global_variables_initializer()

	# Start training
	with tf.Session() as sess:

	  # Run the initializer
	  sess.run(init)
	  load(sess,"./weights")
	  # Calculate accuracy for MNIST test images
	  l1, l2, l3 = sess.run([l_1, l_2, l_3], feed_dict={X: X_train,Y: Y_tr})

	  l1_t, l2_t, l3_t = sess.run([l_1, l_2, l_3], feed_dict={X: X_test,Y: Y_test})

	  logreg = linear_model.LogisticRegression(C=1e5)

	  logreg.fit(l3, Y_train)
	  acc=logreg.score(l3_t, Y_t)
	  print("\nRegression based classifier accuracy with features from Layer3:",acc)


if  __name__=='__main__':
	if args.train:
		train()
	elif args.test:
		test()
	else:
		#print("Parse --train or --test")
		if args.layers == 1:
			reg_layer1()
		elif args.layers == 2:
			reg_layer2()
		elif args.layers == 3:
			reg_layer3()
		else:
			print("\nThere are total of three layers\
			 so enter a valid argument")


