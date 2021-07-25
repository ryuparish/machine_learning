import tensorflow as tf
import numpy as np
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Getting the path to the data
from pathlib import Path
import os

path = Path(os.getcwd())

# Function to get all the image paths and glob them into one large list and then convert the PosixPath's into
# strings
def image_paths(dirpath):
    # PosixPath.glob will get all the path names of a specific pattern and return all of them in a list
    return [str(path) for path in dirpath.glob("*.jpg")]

# Getting the paths to all the images
clothing_dataset = image_paths(path/"data")

# Making an encoder to store all the encoded versions of the clothing classes
encoder = preprocessing.LabelEncoder()

# All classes
clothing_table = [
    "black_dress",
    "black_pants",
    "black_shirt",
    "black_shoes",
    "black_shorts",
    "black_suit",
    "blue_dress",
    "blue_pants",
    "blue_shirt",
    "blue_shoes",
    "blue_shorts",
    "brown_hoodie",
    "brown_pants",
    "brown_shoes",
    "green_pants",
    "green_shirt",
    "green_shoes",
    "green_shorts",
    "green_suit",
    "pink_hoodie",
    "pink_pants",
    "pink_skirt",
    "red_dress",
    "red_hoodie",
    "red_pants",
    "red_shirt",
    "red_shoes",
    "silver_shoes",
    "silver_skirt",
    "white_dress",
    "white_pants",
    "white_shoes",
    "white_shorts",
    "white_suit",
    "yellow_dress",
    "yellow_shorts",
    "yellow_skirt"
]

encoder.fit(clothing_table)

# For labelling each image
def label_image(filename):
    # Getting the last file instead of the entire file path
    depth_of_file = filename.count("/")
    split_filepath = filename.split("/", depth_of_file)
    real_filename = split_filepath[-1]
    
    label = real_filename[0:real_filename.find("_", 7)]
    # Returning the image and the encoded label
    return filename, encoder.transform([label])[0]

# Image parsing and then attaching a label based on the name of the file
def parse_image(filepath):
    # Getting the image from the filepath
    image = Image.open(filepath)
    image = image.resize((224, 224))
    image = np.asarray(image)
    # Returning the image and the encoded label
    return image

# Labelling the images
clothing_dataset = map(label_image, clothing_dataset)
clothing_dataset = list(clothing_dataset)

# Shuffling the dataset and splitting it up into train, validation, and test sets
np.random.shuffle(clothing_dataset)
clothing_dataset = np.asarray(clothing_dataset)
train = clothing_dataset[:14170]
valid = clothing_dataset[14170:15170]
test = clothing_dataset[15170:16170]

# Splitting into instances and labels
x_train = train[:,0]
y_train = train[:,1].astype(np.uint8)
x_valid = valid[:,0]
y_valid = valid[:,1].astype(np.uint8)
x_test = test[:,0]
y_test = test[:,1].astype(np.uint8)

################### Saving the dataset as a h5 file  ###########################
# Converting the filepaths into images and then into numpy arrays
#x_train = map(parse_image, x_train)
#x_train = np.asarray(list(x_train))
#x_train = tf.keras.applications.mobilenet_v2.preprocess_input(x_train)

#data_file = h5py.File('x_train_data.h5', 'w')
#data_file.create_dataset('x_train_data', data=x_train)
#data_file.close()
#data_file = h5py.File('y_train_data.h5', 'w')
#data_file.create_dataset('y_train_data', data=y_train)
#data_file.close()
#
#x_valid = map(parse_image, x_valid)
#x_valid = np.asarray(list(x_valid))
#x_valid = tf.keras.applications.mobilenet_v2.preprocess_input(x_valid)
#
#data_file = h5py.File('x_valid_data.h5', 'w')
#data_file.create_dataset('x_valid_data', data=x_valid)
#data_file.close()
#data_file = h5py.File('y_valid_data.h5', 'w')
#data_file.create_dataset('y_valid_data', data=y_valid)
#data_file.close()
#
#x_test = map(parse_image, x_test)
#x_test = np.asarray(list(x_test))
#x_test = tf.keras.applications.mobilenet_v2.preprocess_input(x_test)
#
#data_file = h5py.File('x_test_data.h5', 'w')
#data_file.create_dataset('x_test_data', data=x_test)
#data_file.close()
#data_file = h5py.File('y_test_data.h5', 'w')
#data_file.create_dataset('y_test_data', data=y_test)
#data_file.close()

