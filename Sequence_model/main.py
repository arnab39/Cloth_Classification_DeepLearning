import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.python.ops import rnn_cell_impl
from sklearn.model_selection import train_test_split
import data_loader
import argparse
import os

tf.set_random_seed(123)

parser = argparse.ArgumentParser()

parser.add_argument('--model', action="store", dest='type')
parser.add_argument('--train', action="store_true",dest='train',default=False)
parser.add_argument('--test' , action="store_true",dest='test' ,default=False)
parser.add_argument('--hidden_unit', action="store", dest="hidden_unit_size", type=int)

DL = data_loader.DataLoader()
train_data,labels = DL.load_data()
train_labels = np.eye(10)[np.asarray(labels, dtype=np.int32)]

test_data,labels = DL.load_data('test')
test_labels = np.eye(10)[np.asarray(labels, dtype=np.int32)]

train_data = train_data.astype(np.float64)
test_data = test_data.astype(np.float64)
train_data = train_data/255
test_data = test_data/255

train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.1, random_state=123)

time_steps=28
in_dim=28
lr_param=0.001
out_dim=10
batch_size=128
max_iter = 5000

os.system("git clone https://github.com/theSarge911/weights.git")

class GRU_Cell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, reuse=None):
        super(GRU_Cell, self).__init__(_reuse=reuse)
        self._num_units = num_units

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        input_depth = inputs_shape[1].value
        self._gate_kernel = self.add_variable("gates/kernel", shape=[input_depth + self._num_units, 2 * self._num_units])
        self._gate_bias = self.add_variable("gates/bias", shape=[2 * self._num_units],initializer= tf.constant_initializer(1.0))
        self._candidate_kernel = self.add_variable("candidate/kernel", shape=[input_depth + self._num_units, self._num_units])
        self._candidate_bias = self.add_variable( "candidate/bias", shape=[self._num_units], initializer= tf.initializers.zeros(dtype="float32"))

    def call(self, inputs, state):
        gate_inputs = tf.matmul(tf.concat([inputs, state],1), self._gate_kernel) #evaluates the inner part where weight is multiplied
        gate_inputs = tf.nn.bias_add(gate_inputs, self._gate_bias)
        value = tf.sigmoid(gate_inputs)

        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
        r_state = tf.multiply(r, state)

        candidate = tf.matmul(tf.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = tf.nn.bias_add(candidate, self._candidate_bias)

        c = tf.tanh(candidate)
        new_h = u * state + (1 - u) * c
        return new_h, new_h

class LSTM_Cell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, reuse=None):
        super(LSTM_Cell, self).__init__(_reuse=reuse)
        self._num_units = num_units

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable("kernel", shape=[input_depth + h_depth, 4*self._num_units])
        self._bias   = self.add_variable("bias"  , shape=[4*self._num_units], initializer=tf.initializers.zeros(dtype="float32"))

    def call(self, inputs, state):
        c, h = state
        gate_inputs = tf.matmul(tf.concat([inputs, h],1), self._kernel) #evaluates the inner part where weight is multiplied
        gate_inputs = tf.nn.bias_add(gate_inputs,self._bias)

        f, i, _c, o = tf.split(value=gate_inputs, num_or_size_splits=4, axis=1)

        new_c = tf.add(tf.multiply(c,tf.sigmoid(f)),tf.multiply(tf.sigmoid(i),tf.tanh(_c)))
        new_h = tf.multiply(tf.tanh(new_c),tf.sigmoid(o))
        new_state = rnn.LSTMStateTuple(new_c, new_h)

        return new_c, new_state

if(parser.parse_args().train and (parser.parse_args().type == 'LSTM')):
    num_units = parser.parse_args().hidden_unit_size
    path = "./weights/LSTM" + str(num_units) + ".ckpt"
    out_weights = tf.Variable(tf.random_normal([num_units,out_dim]))
    out_bias = tf.Variable(tf.random_normal([out_dim]))

    x = tf.placeholder("float",[None,time_steps,in_dim])
    y = tf.placeholder("float",[None,out_dim])
    input = tf.unstack(x ,time_steps,1)

    lstm_layer = LSTM_Cell(num_units)
    outputs,_ = rnn.static_rnn(lstm_layer,input, dtype="float32")
    prediction = tf.matmul(outputs[-1],out_weights)+out_bias
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    opt = tf.train.AdamOptimizer(learning_rate=lr_param).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    val_batch_x = val_data[:batch_size]
    val_batch_x = val_batch_x.reshape((batch_size,time_steps,in_dim))
    val_batch_y = val_labels[:batch_size]

    init=tf.global_variables_initializer()
    saver = tf.train.Saver()

    best_validation_accuracy = 0.0
    last_improvement = 0
    patience = 100

    #with tf.Session() as sess:
    sv = tf.train.Supervisor()
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sv.saver.restore(sess, path)
        sess.run(init)
        count = 1
        for epoch in range(max_iter+1):
            start = count*batch_size
            end = (count+1)*batch_size
            if (end > train_data.shape[0]):
                count=0
                start = count*batch_size
                end = (count+1)*batch_size

            batch_x = train_data[start:end]
            batch_y = train_labels[start:end]

            batch_x = batch_x.reshape((batch_size,time_steps,in_dim))

            sess.run(opt, feed_dict={x: batch_x, y: batch_y})

            if (epoch%50 == 0):
                los = sess.run(cost,feed_dict={x:batch_x, y:batch_y})
                acc = sess.run(accuracy,feed_dict={x:batch_x, y:batch_y})
                print 'Epoch : ' + str(epoch) + ' Training cost : ' + str(los) + ' Accuracy: ' + str(acc)
                acc=sess.run(accuracy,feed_dict={x:val_batch_x, y:val_batch_y})
                print 'Validation accuracy: ' + str(acc)

                if acc > best_validation_accuracy:
                    last_improvement = epoch
                    best_validation_accuracy = acc
                    save_path = saver.save(sess, path)

            #if epoch - last_improvement > patience:
            #    print("Early stopping ...")
            #    break

            count = count+1

if(parser.parse_args().train and (parser.parse_args().type == 'GRU')):
    num_units = parser.parse_args().hidden_unit_size
    path = "./weights/GRU" + str(num_units) + ".ckpt"
    out_weights = tf.Variable(tf.random_normal([num_units,out_dim]))
    out_bias = tf.Variable(tf.random_normal([out_dim]))

    x = tf.placeholder("float",[None,time_steps,in_dim])
    y = tf.placeholder("float",[None,out_dim])
    input = tf.unstack(x ,time_steps,1)

    lstm_layer=GRU_Cell(num_units)
    outputs,_=rnn.static_rnn(lstm_layer,input, dtype="float32")
    prediction=tf.matmul(outputs[-1],out_weights)+out_bias
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    opt=tf.train.AdamOptimizer(learning_rate=lr_param).minimize(cost)

    correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    val_batch_x = val_data[:batch_size]
    val_batch_x = val_batch_x.reshape((batch_size,time_steps,in_dim))
    val_batch_y = val_labels[:batch_size]

    init=tf.global_variables_initializer()
    saver = tf.train.Saver()

    best_validation_accuracy = 0.0
    last_improvement = 0
    patience = 10
    max_iter

    #with tf.Session() as sess:
    sv = tf.train.Supervisor()
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sv.saver.restore(sess, path)
        sess.run(init)
        count = 1
        for epoch in range(max_iter+1):
            start = count*batch_size
            end = (count+1)*batch_size
            if (end > train_data.shape[0]):
                count=0
                start = count*batch_size
                end = (count+1)*batch_size

            batch_x = train_data[start:end]
            batch_y = train_labels[start:end]

            batch_x = batch_x.reshape((batch_size,time_steps,in_dim))

            sess.run(opt, feed_dict={x: batch_x, y: batch_y})

            if (epoch%50 == 0):
                los = sess.run(cost,feed_dict={x:batch_x, y:batch_y})
                acc = sess.run(accuracy,feed_dict={x:batch_x, y:batch_y})
                print 'Epoch : ' + str(epoch) + ' Training cost : ' + str(los) + ' Accuracy: ' + str(acc)
                acc=sess.run(accuracy,feed_dict={x:val_batch_x, y:val_batch_y})
                print 'Validation accuracy: ' + str(acc)

                if acc > best_validation_accuracy:
                    last_improvement = epoch
                    best_validation_accuracy = acc
                    save_path = saver.save(sess, path)

            #if epoch - last_improvement > patience:
            #    print("Early stopping ...")
            #    break

            count = count+1

if(parser.parse_args().test and (parser.parse_args().type == 'LSTM')):
    num_units = parser.parse_args().hidden_unit_size
    out_weights = tf.Variable(tf.random_normal([num_units,out_dim]))
    out_bias = tf.Variable(tf.random_normal([out_dim]))
    x = tf.placeholder("float",[None,time_steps,in_dim])
    y = tf.placeholder("float",[None,out_dim])
    input = tf.unstack(x ,time_steps,1)

    lstm_layer=LSTM_Cell(num_units)
    outputs,_=rnn.static_rnn(lstm_layer,input, dtype="float32")
    prediction=tf.matmul(outputs[-1],out_weights)+out_bias
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    opt=tf.train.AdamOptimizer(learning_rate=lr_param).minimize(cost)

    correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #with graph.as_default():
    sv = tf.train.Supervisor()
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        path = "./weights/LSTM" + str(num_units) + ".ckpt"
        sv.saver.restore(sess, path)
        print("Restored!")
        #saver.restore(sess, path)
        acc = 0
        start = 0
        end = start + batch_size
        while(end <= test_data.shape[0]):
            test_data = test_data[start:end].reshape((-1, time_steps, in_dim))
            test_label = test_labels[start:end]
            acc += sess.run(accuracy, feed_dict={x: test_data, y: test_label})
            start += batch_size
            end += batch_size
        acc = acc/(start/batch_size)
        print("Testing Accuracy:", acc)

if(parser.parse_args().test and (parser.parse_args().type == 'GRU')):
    num_units = parser.parse_args().hidden_unit_size
    out_weights = tf.Variable(tf.random_normal([num_units,out_dim]))
    out_bias = tf.Variable(tf.random_normal([out_dim]))
    x = tf.placeholder("float",[None,time_steps,in_dim])
    y = tf.placeholder("float",[None,out_dim])
    input = tf.unstack(x ,time_steps,1)

    lstm_layer=GRU_Cell(num_units)
    outputs,_=rnn.static_rnn(lstm_layer,input, dtype="float32")
    prediction=tf.matmul(outputs[-1],out_weights)+out_bias
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    opt=tf.train.AdamOptimizer(learning_rate=lr_param).minimize(cost)

    correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #with graph.as_default():
    sv = tf.train.Supervisor()
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        path = "./weights/GRU" + str(num_units) + ".ckpt"
        sv.saver.restore(sess, path)
        print("Restored!")
        #saver.restore(sess, path)
        acc = 0
        start = 0
        end = start + batch_size
        while(end <= test_data.shape[0]):
            test_data = test_data[start:end].reshape((-1, time_steps, in_dim))
            test_label = test_labels[start:end]
            acc += sess.run(accuracy, feed_dict={x: test_data, y: test_label})
            start += batch_size
            end += batch_size
        acc = acc/(start/batch_size)
        print("Testing Accuracy:", acc)
