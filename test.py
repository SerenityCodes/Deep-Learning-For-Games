from mnist_autoencoder import MNIST_Autoencoder, display_reconstruction_images
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np
import os

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalization of the input and flattening the 28x28 pictures
X_train = X_train.astype("float32") / 255.0
X_train = X_train.reshape((len(X_train)), np.prod(X_train.shape[1:]))
X_test = X_test.astype("float32") / 255.0
X_test = X_test.reshape((len(X_test)), np.prod(X_test.shape[1:]))

if not os.path.exists("models"):
    os.mkdir("models")

encoding_dim = 64
autoencoder = MNIST_Autoencoder(encoder_dim=encoding_dim)
model_save_path = os.path.join("models", "mnist_autoencoder.tf")
if os.path.exists(model_save_path):
    autoencoder = load_model(model_save_path)
else:
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.fit(X_train, X_train, epochs=100, batch_size=256, shuffle=True, validation_data=(X_test, X_test))
    autoencoder.save(model_save_path)

encoded_imgs = autoencoder.encoder(X_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

display_reconstruction_images(X_test, decoded_imgs, save_img=True, img_output=f"Autoencoder_{encoding_dim}")
