import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from datetime import datetime

####################################################

def norm_init(size):
    in_dim = size[0]
    norm_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random.normal(shape=size, stddev=norm_stddev)

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.matmul(G_h1, G_W2) + G_b2
    #G_h2_mean, G_h2_var = tf.nn.moments(G_h2,[0])
    #scale2 = tf.Variable(tf.ones([img_size]))
    #beta2 = tf.Variable(tf.zeros([img_size]))
    #epsilon = 1e-3
    #BN2 = tf.nn.batch_normalization(G_h2,G_h2_mean,G_h2_var,beta2,scale2,epsilon)
    #G_prob = tf.nn.tanh(BN2)
    G_prob = tf.nn.sigmoid(G_h2)
    return G_prob

def sample_Z(m, n):
    return np.random.normal(0.0, 0.5, size=[m, n])

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.tanh(D_h2)
    return D_prob, D_h2

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)






####################################################

seed = 42
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

Z_dim = 100
img_size = 784 
batch_size = 128

####################################################

with tf.device("/job:localhost/replica:0/task:0/device:XLA_GPU:0"):
    #weights and biases for each discriminator layer
    D_W1 = tf.Variable(norm_init([img_size, batch_size]))
    D_b1 = tf.Variable(tf.zeros(shape=[batch_size]))
    D_W2 = tf.Variable(norm_init([batch_size, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))
    theta_D = [D_W1, D_W2, D_b1, D_b2]
    #input to your discrim -- i.e. real data or output from generator
    X = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name='X')
    
    #weights and biases for each generator layer
    G_W1 = tf.Variable(norm_init([Z_dim, batch_size]))
    G_b1 = tf.Variable(tf.zeros(shape=[batch_size]))
    G_W2 = tf.Variable(norm_init([batch_size, img_size]))
    G_b2 = tf.Variable(tf.zeros(shape=[img_size]))
    theta_G = [G_W1, G_W2, G_b1, G_b2]
    #input to your generator, sampled from some prior
    Z = tf.compat.v1.placeholder(tf.float32, shape=[None, Z_dim])
    
    G_sample = generator(Z)
    D_real, D_logit_real = discriminator(X)
    D_fake, D_logit_fake = discriminator(G_sample)
    #D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    #D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    #D_loss = D_loss_real + D_loss_fake
    #G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    D_loss = tf.contrib.gan.losses.wargs.modified_discriminator_loss(D_real, D_fake)
    G_loss = tf.contrib.gan.losses.wargs.modified_generator_loss(D_fake)
    D_solver = tf.compat.v1.train.GradientDescentOptimizer(0.05).minimize(D_loss, var_list=theta_D)
    G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

####################################################

#mnist = tf.keras.datasets.mnist
x_path = 'X_kannada_MNIST_train.npz' 
x = np.load(x_path, mmap_mode='r')
y_path = 'y_kannada_MNIST_train.npz'
y = np.load(y_path, mmap_mode='r')
#(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x['arr_0']
y_train = y['arr_0']
x_train = x_train.reshape(x_train.shape[0], -1)
train_filter = np.where((y_train == 5))
x_train, y_train = x_train[train_filter], y_train[train_filter]
x_train = x_train.astype('float32')
x_train /= 255

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0
D_loss_list = []
G_loss_list = []
startTime = datetime.now()

sess = tf.compat.v1.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess.run(tf.global_variables_initializer())

for it in range(100000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
    X_mb, _ = next_batch(batch_size, x_train, y_train)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})
    D_loss_list.append(D_loss_curr)
    G_loss_list.append(G_loss_curr)
    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print("Time taken:", datetime.now() - startTime)

discrim, = plt.plot(D_loss_list, label='Discrim Loss', c='r')
gen, = plt.plot(G_loss_list, label='Gen Loss', c='b')
plt.legend(handles=[discrim, gen])
plt.savefig("out/losses.png")
