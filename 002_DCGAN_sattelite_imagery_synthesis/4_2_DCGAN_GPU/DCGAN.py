# IMPORT PACKAGES
from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
#from keras.utils.vis_utils import plot_model
from util import *



def make_generator_model(latent_dim, GENERATE_RES = 2):
    model = Sequential()
    model.add(Dense(4*4*256, use_bias=False, input_shape=(latent_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((4, 4, 256)))
    assert model.output_shape == (None, 4, 4, 256)  # Note: None is the batch size

    # ADDITIONAL UPSAMPLING WILL LEAD TO HIGHER RESOLUTION
    for i in range(GENERATE_RES):
        model.add(UpSampling2D())  # 8*8 ==> 16*16
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.2))
        model.add(LeakyReLU(alpha=0.2))

    # FINAL LAYER
    model.add(Conv2D(3, kernel_size=3, strides=1, padding="same"))
    model.add(Activation("tanh"))

    # SUMMARIZE THE MODEL
    model.summary()
    # PLOT THE MODEL
    #plot_model(model, to_file='model_figure/generatorNet.png', show_shapes=True, show_layer_names=True)
    return model


def make_discriminator_model(image_shape, GENERATE_RES = 2):
    model = Sequential()

    # DOWNSAMPLE 64*64 ==> 32*32
    model.add(Conv2D(128, kernel_size=3, strides=2, input_shape=image_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    for i in range(GENERATE_RES-2):
        # DOWNSAMPLE 32*32 ==> 16*16
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

    # DOWNSAMPLE 8*8 ==> 4*4
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    # CNN LAYER
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    # CNN LAYER
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    # FINAL DENSE LAYER
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # COMPILE MODEL
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # SUMMARIZE THE MODEL
    model.summary()
    # PLOT THE MODEL
    #plot_model(model, to_file='model_figure/discriminatorNet.png', show_shapes=True, show_layer_names=True)
    return model

def make_dcgan(g_model, d_model):
    discriminator = d_model
    generator = g_model
    model = Sequential()
    discriminator.trainable = False
    model.add(generator)
    model.add(discriminator)
    opt = Adam(1.5e-4, 0.5)  # learning rate and momentum adjusted from paper
    model.compile(loss="binary_crossentropy", optimizer=opt)
    # SUMMARIZE THE MODEL
    model.summary()
    # PLOT THE MODEL
    #plot_model(model, to_file='model_figure/dcganNet.png', show_shapes=True, show_layer_names=True)
    return model

