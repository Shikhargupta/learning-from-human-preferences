"""
Core network which predicts rewards from frames,
for gym-moving-dot and Atari games.
"""

import tensorflow as tf

from nn_layers import dense_layer, conv_layer
import tensorflow_probability as tfp
import numpy as np
from pref_db import PrefDB, PrefBuffer
from tensorflow.keras import optimizers
from utils import RunningStat, batch_iter

tf.enable_eager_execution()

def get_dot_position(s):
    """
    Estimate the position of the dot in the gym-moving-dot environment.
    """
    # s is (?, 84, 84, 4)
    s = s[..., -1]  # select last frame; now (?, 84, 84)

    x = tf.reduce_sum(s, axis=1)  # now (?, 84)
    x = tf.argmax(x, axis=1)

    y = tf.reduce_sum(s, axis=2)
    y = tf.argmax(y, axis=1)

    return x, y

class gp_rp:
    def __init__(self):
        tfd = tfp.distributions
        psd_kernels = tfp.math.psd_kernels

        self.kernel = psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.Variable(1., dtype=np.float64, name='amplitude'),
            length_scale=tf.Variable(1., dtype=np.float64, name='length_scale'),
            feature_ndims=1)

        self.N=1000
        self.feat_dims = 3
        self.init_ = np.random.normal(size=(self.feat_dims,self.N))
        self.latent_index_points = tf.Variable(self.init_, name='latent_index_points')

        self.gp = tfd.GaussianProcess(self.kernel,self.latent_index_points)
        self.optimizer = optimizers.Adam()
        self.epochs = 1

    def reward(self,frames):
        s = np.array(frames[0])

        a = s[:,0,0,-1]
        a = tf.cast(a, tf.float64) / 4.0

        xc, yc = self.get_dot_position(s)
        xc = tf.cast(xc, tf.float64) / 83.0
        yc = tf.cast(yc, tf.float64) / 83.0

        features = [a, xc, yc]
        x1 = tf.stack(features, axis=2)

        r = self.gp(x1)
        return r

    def get_dot_position(self,s):
        """
        Estimate the position of the dot in the gym-moving-dot environment.
        """
        # s is (?, 84, 84, 4)
        s = s[..., -1]  # select last frame; now (?, 84, 84)

        x = tf.reduce_sum(s, axis=2)  # now (?, 84)
        x = tf.argmax(x, axis=2)

        y = tf.reduce_sum(s, axis=3)
        y = tf.argmax(y, axis=2)

        return x, y

    def get_features(self,clip_batch):
        s = np.array(clip_batch)
        a = s[:, :,0,0, -1]
        a = tf.cast(a, tf.float64) / 4.0

        xc, yc = self.get_dot_position(s)
        xc = tf.cast(xc, tf.float64) / 83.0
        yc = tf.cast(yc, tf.float64) / 83.0

        features = [a, xc, yc]
        x1 = tf.stack(features, axis=2)
        return x1

    def train(self, pref_db_train):
        for ep in range(self.epochs):
            print("Epoch: ",ep)
            for _, batch in enumerate(batch_iter(pref_db_train.prefs,
                                                 batch_size=32,
                                                 shuffle=True)):
                s1s = [pref_db_train.segments[k1] for k1, k2, pref, in batch]
                s2s = [pref_db_train.segments[k2] for k1, k2, pref, in batch]
                prefs = [pref for k1, k2, pref, in batch]
                x1 = self.get_features(s1s)
                x2 = self.get_features(s2s)
                self.train_step(x1,x2,prefs)


    def train_step(self,x1,x2,prefs):
        with tf.GradientTape() as tape:
            r1 = self.gp.prob(x1)
            r2 = self.gp.prob(x2)
            rs1 = tf.reduce_sum(r1, axis=1)
            rs2 = tf.reduce_sum(r2, axis=1)
            rs = tf.stack([rs1, rs2], axis=1)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=prefs,
                                                           logits=rs)
        grads = tape.gradient(loss, self.gp.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.gp.trainable_variables))


def net_moving_dot_features(s, batchnorm, dropout, training, reuse):
    # Action taken at each time step is encoded in the observations by a2c.py.
    a = s[:, 0, 0, -1]
    a = tf.cast(a, tf.float32) / 4.0

    xc, yc = get_dot_position(s)

    xc = tf.cast(xc, tf.float32) / 83.0
    yc = tf.cast(yc, tf.float32) / 83.0

    features = [a, xc, yc]
    x = tf.stack(features, axis=1)

    x = dense_layer(x, 64, "d1", reuse, activation='relu')
    x = dense_layer(x, 64, "d2", reuse, activation='relu')
    x = dense_layer(x, 64, "d3", reuse, activation='relu')
    x = dense_layer(x, 1, "d4", reuse, activation=None)
    x = x[:, 0]

    return x


def net_cnn(s, batchnorm, dropout, training, reuse):
    x = s / 255.0
    # Page 15: (Atari)
    # "[The] input is fed through 4 convolutional layers of size 7x7, 5x5, 3x3,
    # and 3x3 with strides 3, 2, 1, 1, each having 16 filters, with leaky ReLU
    # nonlinearities (α = 0.01). This is followed by a fully connected layer of
    # size 64 and then a scalar output. All convolutional layers use batch norm
    # and dropout with α = 0.5 to prevent predictor overfitting"
    x = conv_layer(x, 16, 7, 3, batchnorm, training, "c1", reuse, 'relu')
    x = tf.layers.dropout(x, dropout, training=training)
    x = conv_layer(x, 16, 5, 2, batchnorm, training, "c2", reuse, 'relu')
    x = tf.layers.dropout(x, dropout, training=training)
    x = conv_layer(x, 16, 3, 1, batchnorm, training, "c3", reuse, 'relu')
    x = tf.layers.dropout(x, dropout, training=training)
    x = conv_layer(x, 16, 3, 1, batchnorm, training, "c4", reuse, 'relu')

    w, h, c = x.get_shape()[1:]
    x = tf.reshape(x, [-1, int(w * h * c)])

    x = dense_layer(x, 64, "d1", reuse, activation='relu')
    x = dense_layer(x, 1, "d2", reuse, activation=None)
    x = x[:, 0]

    return x
