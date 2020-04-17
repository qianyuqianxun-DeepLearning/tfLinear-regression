import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
## 新添加

from IPython import display

## parameters
learning_rate = 0.1
training_epochs = 1000
display_step = 5

## create data
train_X = np.random.rand(100).astype(np.float32)
train_X.sort()
train_Y = (train_X - 5)*100 + 0.3
n_samples = train_X.shape[0]

X = tf.compat.v1.placeholder("float")
Y = tf.compat.v1.placeholder("float")

w = tf.compat.v1.Variable(np.random.randn(), name='weight')
b = tf.compat.v1.Variable(np.random.randn(), name='bias')

pred = tf.add(tf.multiply(X, w), b)
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    plt.figure(num=1)

    for epoch in range(training_epochs):

        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if (epoch + 1) % display_step == 0:
            #             plt.figure(num=3)
            plt.ion()
            plt.cla()
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(w), 'b=', sess.run(b))
            ### draw the picture
            plt.plot(train_X, train_Y, 'ro', label='Orginal data')
            plt.plot(train_X, sess.run(w)*train_X + sess.run(b) , label='Fitted line')
            plt.legend()

            plt.pause(0.001)
            #### 新添加
            display.clear_output(wait=True)

    plt.show()