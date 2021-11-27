from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import TensorBoard
import os


class MNISTAutoencoder(Model):
    def __init__(self, encoder_dim=32, input_shape=(784,), optimizer=Nadam(), *args, **kwargs):
        super(MNISTAutoencoder, self).__init__(*args, **kwargs)
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


class ConvAutoEncoder:
    def __init__(self, input_shape=(28, 28, 1), optimizer=Nadam()):
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.__build_model()

    def __call__(self, inputs):
        return self.autoencoder(inputs)

    def __build_model(self):
        input_img = Input(shape=self.input_shape)
        x = Conv2D(16, (3, 3), activation="relu", padding="same")(input_img)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
        encoded = MaxPooling2D((2, 2), padding="same")(x)

        x = Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

        self.autoencoder = Model(input_img, decoded, name="autoencoder")
        self.autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
        print(self.autoencoder.summary())

    def train_model(self, x_train, x_test, epochs=10, batch_size=256):
        self.autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test),
                             callbacks=[TensorBoard(log_dir="/tmp/autoencoder")])

    def save_model(self, name_to_save="models/conv_autoencoder.h5"):
        self.autoencoder.save(name_to_save)


def display_reconstruction_images(x_test, decoded_imgs, num_imgs=10, save_img=False, img_output=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 4))
    for i in range(num_imgs):
        ax = plt.subplot(2, num_imgs, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
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
