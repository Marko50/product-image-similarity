from os import read, makedirs
from cv2 import imread, imwrite
from numpy import asarray, expand_dims, squeeze
from pandas import read_csv
from sklearn.model_selection import train_test_split

## All the images are 150x150 3 RGB channels

name_column = "file"
prefix = "data/style/"
dec_ims_path = "data/decoded/"
neib_ims_path = "data/neighbors/"
file_dataset_name = "style.csv"
file_ds_path = f"{prefix}{file_dataset_name}"
MAX_PIXEL_RGB_VALUE = 255


def reg_img(img):
    return img / MAX_PIXEL_RGB_VALUE

def de_reg_img(img):
    return img * MAX_PIXEL_RGB_VALUE


def train_test_split_images():
    ds_array = read_original_images()
    # we divide the data in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        ds_array, ds_array, test_size=0.3, shuffle=True
    )

    return X_train, X_test, y_train, y_test

def read_original_images():
    # read dataframe
    dataframe = read_csv(file_ds_path)
    ds_array = []
    for im_name in dataframe[name_column]:
        np_arr = imread(f"{prefix}{im_name}")
        # we regularize the data comprehending the values from 0 to 1
        ds_array.append((reg_img(np_arr)).astype("float32"))

    return asarray(ds_array)

def flattened_images(ds_array):
    return asarray([im.flatten() for im in ds_array])

def write_decoded_images(autoencoder):
    dataframe = read_csv(file_ds_path)
    for im_name in dataframe[name_column]:
        np_arr = imread(f"{prefix}{im_name}")
        print("Current image: ", im_name)
        np_arr_corr = expand_dims((reg_img(np_arr)).astype("float32"), axis=0)
        decoded = autoencoder.predict(np_arr_corr)
        corr_img = (de_reg_img(squeeze(decoded, axis=0))).astype('int32')
        imwrite(f"{dec_ims_path}{im_name}", corr_img)

def write_neighborhood_images(knn, ds_array, enc_ds_array):
    dataframe = read_csv(file_ds_path)
    for index, im_name in enumerate(dataframe[name_column]):
        indices = knn.kneighbors([enc_ds_array[index]], return_distance = False)
        neighbors = [ds_array[i] for i in squeeze(indices)]
        path_name = f"{neib_ims_path}{im_name}"
        makedirs(path_name, exist_ok=True)
        for i, neib in enumerate(neighbors):
            print("Writing: ", f"{path_name}/{i}_neighbor.png")
            imwrite(f"{path_name}/{i}_neighbor.png", de_reg_img(neib))
            


        