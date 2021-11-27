from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
import os


class MNIST_Autoencoder(Model):
    def __init__(self, encoder_dim=32, input_shape=(784,), optimizer=Nadam(), *args, **kwargs):
        super(MNIST_Autoencoder, self).__init__(*args, **kwargs)
        self.encoder_dim = encoder_dim
        self.optimizer = optimizer

        input_img = Input(shape=input_shape)

        encoded = Dense(encoder_dim, activation="relu")(input_img)
        self.encoder = Model(input_img, encoded)

        decoded = Dense(784, activation="sigmoid")(encoded)
        self.autoencoder = Model(input_img, decoded)
        encoded_input = Input(shape=(encoder_dim,))

        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

    def call(self, inputs, training=None, mask=None):
        return self.autoencoder(inputs)


def display_reconstruction_images(X_test, decoded_imgs, num_imgs=10, save_img=False, img_output=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 4))
    for i in range(num_imgs):
        ax = plt.subplot(2, num_imgs, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, num_imgs, i + 1 + num_imgs)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if save_img:
        if not os.path.exists("images"):
            os.mkdir("images")
        if img_output is None:
            plt.savefig("images/Autoencoder_32.png")
        else:
            plt.savefig(f"images/{img_output}.png")
    plt.show()

