import tensorflow as tf 
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np 

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)#call mnist function

learningRate = 0.001
trainingIters = 200000
batchSize = 100
displayStep = 10

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 64 #number of neurons for the RNN
nClasses = 10 #this is MNIST so you know
method = 'lstm'
train_dir = './train_rnn_%s_%d/' % (method, nHidden) # directory where the results from the training are saved
test_dir='./test_rnn_%s_%d/' % (method, nHidden)

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, nInput])
	x = tf.split(x, nSteps, 0) #configuring so you can get it as needed for the 28 pixels
	if method == 'basic':
		lstmCell = rnn_cell.BasicRNNCell(nHidden)#find which lstm to use in the documentation
	elif method == 'lstm':
		lstmCell = rnn_cell.BasicLSTMCell(nHidden, forget_bias=0.9)
	elif method == 'gru':
		lstmCell = rnn_cell.GRUCell(nHidden)
	
	outputs, states = rnn.static_rnn(lstmCell, x, dtype=tf.float32)#for the rnn where to get the output and hidden state 

	return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.global_variables_initializer()


testData = mnist.test.images.reshape((-1, nSteps, nInput))
testLabel = mnist.test.labels

with tf.Session() as sess:
	sess.run(init)
	step = 1
	# Add a scalar summary for the snapshot loss.

	tf.summary.scalar('loss', cost)
	tf.summary.scalar('accuracy', accuracy)

	summary_op = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
	test_writer = tf.summary.FileWriter(test_dir,sess.graph)

	while step* batchSize < trainingIters:
		batchX, batchY = mnist.train.next_batch(batchSize)#mnist has a way to get the next batch
		batchX = batchX.reshape((batchSize, nSteps, nInput))

		sess.run(optimizer, feed_dict={x: batchX, y: batchY})

		if step % displayStep == 0:
			acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
			loss = sess.run(cost, feed_dict={x: batchX, y: batchY})
			print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
                  "{:.6f}".format(float(loss)) + ", Training Accuracy= " + \
                  "{:.5f}".format(float(acc)))
			summary_str = sess.run(summary_op, feed_dict={x:batchX, y:batchY})
			summary_writer.add_summary(summary_str, step)
			summary_writer.flush()

			test_summary=sess.run(summary_op,feed_dict={x: testData, y: testLabel})
			test_writer.add_summary(test_summary, step)
			test_writer.flush()
		step +=1
	print('Optimization finished')
	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
