# Description

This repository will host Python code I've written in an attempt to create a content-based recommender system which uses information present in product images and recommends products with similar images.

In order to do this, using **Tensorflow** and its **sub-classing API** I built a Autoencoder which produces image embeddings in a lower dimensional space.
The encoder does this by minimizing a **categorical cross entropy** error by encoding and decoding the image and comparing it to the original one.
The images were regularized, on a scale of 0-1 (initially, 0-255).
All the used images are 150x150 pixels with 3 RGB channels obtained from [here](https://www.kaggle.com/jonathanoheix/product-recommendation-based-on-visual-similarity).

## Usage

You can install the dependencies by installing `pipenv` and then typing `pipenv install`.
After that you can train the autoencoder using `pipenv run python train` which will run a training session and then save the training results locally which can later be loaded from disk.
In this case, a series of encoded/decoded images will be also written so you can compare them 
with the original ones.
If you run `pipenv run python load`, this will load the autoencoder from disk and then it will 
encode the images, flatten them into a 1-dimensional space and them run a Nearest Neighbor algorithm with K=5. It will write in disk for each elegible image, the more similar ones.
