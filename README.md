# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. 

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

## Specifications

The project submission must include at least two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images. Make sure to include all files necessary to run train.py and predict.py in your submission.

### 1. Train

Train a new network on a data set with train.py

- Basic usage: python train.py data_directory
- Prints out training loss, validation loss, and validation accuracy as the network trains
- Options: * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory * Choose architecture: python train.py data_dir --arch "vgg13" * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 * Use GPU for training: python train.py data_dir --gpu

### 2. Predict

Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

- Basic usage: python predict.py /path/to/image checkpoint
- Options: * Return top KK most likely classes: python predict.py input checkpoint --top_k 3 * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json * Use GPU for inference: python predict.py input checkpoint --gpu
