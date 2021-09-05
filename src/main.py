import sys

from numpy import expand_dims
from lib import read_original_images, flattened_images, train_test_split_images, train_autoencoder, train_knn, write_decoded_images, load_autoencoder, write_neighborhood_images


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'train':
        X_train, X_test, y_train, y_test = train_test_split_images()
        autoencoder = train_autoencoder(X_train, y_train, X_test, y_test, epochs=40)
        write_decoded_images(autoencoder)
    elif action == 'load':
        imgs = read_original_images()
        autoencoder = load_autoencoder()
        encoded_ims = [autoencoder.encode(expand_dims(im, axis=0)).numpy() for im in imgs]
        imgs_flat = flattened_images(encoded_ims)
        knn = train_knn(imgs_flat)
        write_neighborhood_images(knn, imgs, imgs_flat)
        

        
