from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers,losses
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

img_rows = 28
img_cols = 28
channels = 1
z_dim = 100
img_shape = (img_rows,img_cols,channels)

def build_generator(random_noise_dim):
    gen_mode = models.Sequential(
        [
            layers.Dense(7*7*254, input_dim=random_noise_dim),
            layers.Reshape((7,7,254)),
            layers.Conv2DTranspose(128,kernel_size=3, strides=2,padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.01),
            layers.Conv2DTranspose(64,kernel_size=3,strides=1,padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.01),
            layers.Conv2DTranspose(1,kernel_size=3,strides=2,padding='same'),
            layers.Activation('tanh')
        ]
    )

    return gen_mode


def build_discriminator(img_shape):
    dis_mode = models.Sequential(
        [
            layers.Input(img_shape),
            layers.Conv2D(32,kernel_size=3,strides=2,padding='same'),
            layers.LeakyReLU(alpha=0.01),
            layers.Conv2D(64,kernel_size=3,strides=2,padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.01),
            layers.Conv2D(128,kernel_size=3,strides=2,padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.01),
            layers.Flatten(),
            layers.Dense(1,activation='sigmoid')
        ]
    )

    return dis_mode


def build_gan(generator,discriminator):
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)

    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)


generator = build_generator(z_dim)
discriminator.trainable = False

gan = build_gan(generator,discriminator)
#print(gan.summary())
gan.compile(
    loss='binary_crossentropy',
    optimizer=Adam()
)

losses_ = []
accuracies = []
iteration_checkpoints = []



def sample_images(generator, image_grid_rows=4,image_grid_columns=4):
    z = np.random.normal(0,1,(image_grid_rows*image_grid_columns,z_dim))
    gen_images = generator.predict(z)
    gen_images = 0.5*gen_images+0.5
    fig, axs = plt.subplots(image_grid_rows,image_grid_columns,sharey=True,sharex=True,figsize=(4,4),)
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i:j].imshow(gen_images[cnt,:,:,0],cmap='gray')
            axs[i:j].axis('off')
            cnt+=1



def train(iterations, batch_sizes, sample_interval):
    (image, _), (_, _) = mnist.load_data()
    image = image/127.5 - 1
    image = np.expand_dims(image, axis=3)

    real = np.ones((batch_sizes, 1))
    fake = np.zeros((batch_sizes, 1))
    for iteration in range(iterations):
        idx = np.random.randint(0, image.shape[0],batch_sizes)  # 0-60000
        imgs = image[idx]

        z = np.random.normal(0,1,(batch_sizes, 100))
        gen_imgs = generator.predict(z)

        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs,fake)
        d_loss, accuracy = 0.5*np.add(d_loss_real,d_loss_fake)

        g_loss = gan.train_on_batch(z,real)

        if (iteration+1)%sample_interval == 0:
            losses_.append((d_loss,g_loss))
            accuracies.append(100*accuracy)
            iteration_checkpoints.append(iteration + 1)

            print("%d [D_loss: %f, acc: %.2f%%] [G_loss: %f]" % (iteration + 1, d_loss,100 * accuracy,g_loss))

            sample_images(generator)


iterations = 20000
batches = 128
sample = 1000

train(iterations, batches, sample)











