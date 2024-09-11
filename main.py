import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist #this is fine

#preparing data
num_features = 784
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.array(x_train, np.float32)
x_test = np.array(x_test, np.float32)

x_train = x_train.reshape([-1, 784])
x_test = x_test.reshape([-1, 784])

x_train = x_train/255.
x_test = x_test/255.

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(256).prefetch(1)

#making weights and bias

num_classes = 10

weight = tf.Variable(tf.ones([num_features, num_classes]), name= "weight")
bias = tf.Variable(tf.zeros([num_classes], name="bias"))

def log_reg(x):
    tf.nn.softmax(tf.matmul((x, weight))+bias) # this just multiplies out one vector by the equivilent index of the other in x to weight, then adds bias.






