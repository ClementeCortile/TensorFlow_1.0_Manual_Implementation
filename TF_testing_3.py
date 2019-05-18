import tensorflow as tf


a = tf.add(3,5)

sess = tf.Session()

with tf.Session() as sess:
    result = sess.run(a)

result
