import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Elapsed Time
start_time = time.time()

# MNIST data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot = True)

# Fixed values - should not change
input_size, num_classes = 784, 10

# Parameters
Learning_Rate = 1E-4
hidden_size = 360
batchSz = 50
num_batches = 2000
num_test_iters = int(10000/batchSz)

# get 'Batch size' number of images and answers
img = tf.placeholder(tf.float32, [batchSz, input_size])
ans = tf.placeholder(tf.float32, [batchSz, num_classes])

# turn img into 4d Tensor
image = tf.reshape(img, [batchSz, 28, 28, 1])

# Filters and Convolution
flts = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev = 0.1))
convOut = tf.nn.conv2d(image, flts, [1,  1,  1,  1], "SAME")
convOut= tf.nn.relu(convOut)
convOut = tf.nn.max_pool(convOut, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

flts2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev = 0.1))
convOut2 = tf.nn.conv2d(convOut, flts2, [1,  1,  1,  1], "SAME")
convOut2 = tf.nn.relu(convOut2)
convOut2 = tf.nn.max_pool(convOut2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
convOut2 = tf.reshape(convOut2, [batchSz, 3136])

# Weights and Biases
W1 = tf.Variable(tf.random_normal([3136, hidden_size], stddev = 0.1))
bW1 = tf.Variable(tf.random_normal([batchSz, hidden_size], stddev = 0.1))
W2 = tf.Variable(tf.random_normal([hidden_size, num_classes], stddev = 0.1))
bW2 = tf.Variable(tf.random_normal([batchSz, num_classes], stddev = 0.1))

# Input and Hidden Layers
L1Input = tf.matmul(convOut2, W1) + bW1
L1Output = tf.nn.relu(L1Input)

# probabilities
prbs = tf.nn.softmax(tf.matmul(L1Output, W2) + bW2)

# Cross Entropy Loss Function
xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))

# Train with Gradient Descent
train = tf.train.AdamOptimizer(Learning_Rate).minimize(xEnt)
numCorrect= tf.equal(tf.argmax(prbs,1), tf.argmax(ans,1))
accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(num_batches):
	imgs, anss = mnist.train.next_batch(batchSz)
	sess.run(train, feed_dict={img: imgs, ans: anss})
# Test
sumAcc=0
for i in range(num_test_iters):
        imgs, anss= mnist.test.next_batch(batchSz)
        sumAcc+=sess.run(accuracy, feed_dict={img: imgs, ans: anss})
print("Test Accuracy: " + str(sumAcc/num_test_iters*100.0) + "% \n")
print("Time taken: " + str(time.time() - start_time))
