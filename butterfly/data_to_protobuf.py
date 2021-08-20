# This file transforms a .h5 format dataset into 10 distinct files that can then be used to efficiently load a tensorflow 2 dataset 
# by utilizing Tensorflow Records, a serialized object that can allow training WITHOUT having to load an entire dataset into RAM.

# Loading the Protobuf format and making the Example object function
import tensorflow as tf
import numpy as np
from sys import argv
import h5py
import matplotlib.pyplot as plt
from typing import List
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example
from contextlib import ExitStack

# Restoring the dataset as a numpy array
# WARNING! Load these in one pair at a time of there will not be enough RAM and the process will be killed over and over (at least on my 8GB RAM laptop)
def load_data(filenames : List[str], data_name : List[str], limit = False, bound = None) -> (List[List[int]], List[List[int]]):
    data_file = h5py.File(filename[0], 'r')
    if(limit):
        samples = data_file[data_name[0]][:bound]
    else:
        samples = data_file[data_name[0]][...]
    data_file.close()
    data_file = h5py.File(filename[1], 'r')
    if(limit):
        labels = data_file[data_name[1]][:bound]
    else:
        labels = data_file[data_name[1]][...]
    return samples, labels

    # ie :
    # You may want to limit the number of samples you read in for the sake of crashing
    #data_file = h5py.File('x_train_data.h5', 'r')
    #x_train = data_file["x_train_data"][8000:]
    #data_file.close()
    #data_file = h5py.File('y_train_data.h5', 'r')
    #y_train = data_file["y_train_data"][8000:]
    #data_file.close()
    #data_file = h5py.File('x_valid_data.h5', 'r')
    #x_valid = data_file["x_valid_data"][...]
    #data_file.close()
    #data_file = h5py.File('y_valid_data.h5', 'r')
    #y_valid = data_file["y_valid_data"][...]
    #data_file.close()

def create_example(image, label) -> Example:
    image_data = tf.io.serialize_tensor(image)
    return Example(
        features=Features(
            feature={
                # I tried to use just image as a float32 numpy array but it does not work with FloatList
                # so we will use bytes instead.
                "image": Feature(bytes_list=BytesList(value=[image_data.numpy()])),
                "label": Feature(int64_list=Int64List(value=[label])),
            }))

# Creating function that splits each dataset into ten files and write to them in one at a time
def write_tfrecords(name, dataset, n_shards=10):
    
    # Getting a list of uniquely named paths for the data set
    paths = ["{}.tfrecord-{:05d}-of-{:05d}".format(name, index, n_shards)
             for index in range(n_shards)]
    
    # For the second portion of the training dataset because it is too much for my RAM
    #paths = ["{}.tfrecord-{:05d}-of-{:05d}".format(name, index + 10, n_shards + 10)
    #         for index in range(n_shards)]
    # The exit stack will ensure that the writers will auto-exit on the case that they are not closed manually
    with ExitStack() as stack:
        
        # Getting a list of active writers for each of the ten files
        writers = [stack.enter_context(tf.io.TFRecordWriter(path))
                   for path in paths]
        
        # Using enumerate to cleverly get each of the names of the file paths defined earlier
        for index, (image, label) in dataset.enumerate():
            shard = (index) % (n_shards)
            example = create_example(image, label)
            writers[shard].write(example.SerializeToString())
    return paths

# Loading each of the datasets to their respective ten file paths 
def write_filepaths_to_txt_file(filepaths, file_name, mode = "w"):
    new_filepaths = [filepath + "\n" for filepath in filepaths]
    filepath_destination = open(file_name, mode)
    filepath_destination.writelines(new_filepaths)
    filepath_destination.close()

def main():
    # Checking for arguments
    if(len(argv) < 5):
        print("\nUsage: python3 data_to_protobuf.py <training h5 file> <testing h5 file> <train data name> <test data name> [False/True] [limit]\n")
        exit(0)

    # First loading in my dataset
    if(len(argv) > 5):
        x_data, y_data = load_data([argv[1], argv[2]], [argv[3], argv[4]], argv[5], argv[6])
    else:
        x_data, y_data = load_data([argv[1], argv[2]], [argv[3], argv[4]])
    
    # Creating the dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    # dataset.shuffle(1000)
    # dataset.batch(32)

    # Writing the data into tfrecords
    # ie: 
    #   train_filepaths = write_tfrecords("butterfly.train", training_dataset)
    #   write_filepaths_to_txt_file(train_filepaths, "train_filepaths.txt")
    #   write_filepaths_to_txt_file(train_filepaths, "train_filepaths.txt", "a")
    filepaths = write_tfrecords(argv[1], dataset)
    write_filepaths_to_txt_file(filepaths, "{}filepaths.txt".format(argv[1]))

main()
