from mnist_autoencoder import ConvAutoEncoder, display_reconstruction_images
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import os

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalization of the input and flattening the 28x28 pictures
X_train = X_train.astype("float32") / 255.0
X_train = X_train.reshape(len(X_train), 28, 28, 1)
X_test = X_test.astype("float32") / 255.0
X_test = X_test.reshape(len(X_test), 28, 28, 1)

if not os.path.exists("models"):
    os.mkdir("models")

encoding_dim = 64
autoencoder = ConvAutoEncoder()
model_save_path = os.path.join("models", "conv_autoencoder.h5")
if os.path.exists(model_save_path):
    autoencoder = load_model(model_save_path)
else:
    autoencoder.train_model(x_train=X_train, x_test=X_test, epochs=50, batch_size=128)
    autoencoder.save_model(model_save_path)

decoded_imgs = autoencoder(X_test).numpy()
display_reconstruction_images(X_test, decoded_imgs, save_img=True, img_output=f"Conv_Autoencoder_Comparison")
