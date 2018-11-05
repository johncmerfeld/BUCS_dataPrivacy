import tensorflow as tf

z = tf.constant(5.2, name="x", dtype=tf.float32)

k = tf.Variable(tf.zeros([1]), name="k")