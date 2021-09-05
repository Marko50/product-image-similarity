from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D

weights_path = 'weights/autoencoder'

class Autoencoder(Model):
    KERNEL_CONV = (3, 3)
    KERNEL_POOL = (2, 2)

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.zero_pad_layer = ZeroPadding2D((1, 1))

        self.enc_first_conv_2d_layer = Conv2D(16, self.KERNEL_CONV, activation="relu", padding="same")
        self.enc_sec_conv_2d_layer = Conv2D(8, self.KERNEL_CONV, activation="relu", padding="same")
        self.enc_third_conv_2d_layer = Conv2D(8, self.KERNEL_CONV, activation="relu", padding="same")

        self.enc_first_max_pool = MaxPooling2D(self.KERNEL_POOL, padding="same")
        self.enc_sec_max_pool = MaxPooling2D(self.KERNEL_POOL, padding="same")
        self.enc_third_max_pool = MaxPooling2D(self.KERNEL_POOL, padding="same")

        self.dec_first_conv_2d_layer = Conv2D(8, self.KERNEL_CONV, activation="relu", padding="same")
        self.dec_second_conv_2d_layer = Conv2D(8, self.KERNEL_CONV, activation="relu", padding="same")
        self.dec_third_conv_2d_layer = Conv2D(16, self.KERNEL_CONV, activation="relu", padding="same")
        self.dec_fourth_conv_2d_layer = Conv2D(3, self.KERNEL_CONV, activation="sigmoid", padding="same")

        self.dec_first_up_samp_layer = UpSampling2D(self.KERNEL_POOL)
        self.dec_sec_up_samp_layer = UpSampling2D(self.KERNEL_POOL)
        self.dec_third_up_samp_layer = UpSampling2D(self.KERNEL_POOL)

        self.cropping_layer = Cropping2D((1, 1))

    def encode(self, x):
        encoded = self.zero_pad_layer(x)
        encoded = self.enc_first_conv_2d_layer(encoded)
        encoded = self.enc_first_max_pool(encoded)
        encoded = self.enc_sec_conv_2d_layer(encoded)
        encoded = self.enc_sec_max_pool(encoded)
        encoded = self.enc_third_conv_2d_layer(encoded)
        encoded = self.enc_third_max_pool(encoded)
        return encoded

    def decode(self, encoded):
        decoded = self.dec_first_conv_2d_layer(encoded)
        decoded = self.dec_first_up_samp_layer(decoded)
        decoded = self.dec_second_conv_2d_layer(decoded)
        decoded = self.dec_sec_up_samp_layer(decoded)
        decoded = self.dec_third_conv_2d_layer(decoded)
        decoded = self.dec_third_up_samp_layer(decoded)
        decoded = self.dec_fourth_conv_2d_layer(decoded)
        decoded = self.cropping_layer(decoded) 
        return decoded

    def call(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

def load_autoencoder():
    autoencoder = Autoencoder()
    autoencoder.load_weights(weights_path)
    return autoencoder

def train_autoencoder(X_train, y_train, X_test, y_test, epochs=50):
    autoencoder = Autoencoder()
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    autoencoder.fit(X_train, y_train, epochs=epochs)
    results = autoencoder.evaluate(X_test, y_test)
    print("test loss, test acc:", results)
    autoencoder.save_weights(weights_path)
    return autoencoder