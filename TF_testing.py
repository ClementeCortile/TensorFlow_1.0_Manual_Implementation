import tensorflow as tf
import numpy as np
import pandas as pd
rng = np.random


#Initialize the session
sess = tf.Session()


#Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

#Multiply
result = tf.multiply(x1,x2)

#Print the result
print(sess.run(result))

#sess.close()

#initializing session
with tf.Session() as sess:
    output = sess.run(result)
    print(output)

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]


#tf Graph Input
X = tf.placeholder('float')
Y = tf.placeholder('float')

#Set Model Weights -
W = tf.Variable(rng.randn(), name = "weight")
b = tf.Variable(rng.randn(), name = 'bias')
