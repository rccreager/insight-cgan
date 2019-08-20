from datetime import datetime
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Conv2DTranspose, Conv2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib
from keras import backend as K

####################################################

seed = 42
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

####################################################

class DCGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        optimizer = Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.discriminator.trainable = False #just for combined model
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        valid = self.discriminator(img)
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()
        
        model.add(Conv2D(256, kernel_size=3, strides=2, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='tanh'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):
        x_path = 'X_kannada_MNIST_train.npz'
        x = np.load(x_path, mmap_mode='r')
        y_path = 'y_kannada_MNIST_train.npz'
        y = np.load(y_path, mmap_mode='r')
        x_train = x['arr_0']
        y_train = y['arr_0']
        train_filter = np.where((y_train == 5))
        x_train, y_train = x_train[train_filter], y_train[train_filter]
        x_train = x_train.astype('float32')
        x_train = x_train / 127.5 - 1.
        x_train = np.expand_dims(x_train, axis=3)
        if not os.path.exists('out/'):
            os.makedirs('out/')
        
        i = 0
        D_loss_list = []
        G_loss_list = []
        startTime = datetime.now()
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Select a random batch of images 
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = np.add(d_loss_real, d_loss_fake)
            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                print("Time taken:", datetime.now() - startTime)
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("out/%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    if (tf.test.is_gpu_available): 
        print("good job, your GPU is available! name: " + tf.test.gpu_device_name())
    if (not tf.test.is_gpu_available): print("no GPU found by tf :(")
    dcgan = DCGAN()
    dcgan.train(epochs=2501, batch_size=100, save_interval=50)


        
