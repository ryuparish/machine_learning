# Loading the Protobuf format and making the Example object function
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example
from contextlib import ExitStack

# Restoring the dataset as a numpy array
# Load these in one pair at a time of there will not be enough RAM and the process will be killed over and over
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
data_file = h5py.File('x_test_data.h5', 'r')
x_test = data_file["x_test_data"][...]
data_file.close()
data_file = h5py.File('y_test_data.h5', 'r')
y_test = data_file["y_test_data"][...]
data_file.close()

# Creating the tensorflow datasets
#training_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
testing_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#training_dataset = training_dataset.shuffle(1000)
#valid_dataset = valid_dataset.shuffle(2000)
###testing_dataset = testing_dataset.batch(32)

def create_example(image, label):
    image_data = tf.io.serialize_tensor(image)
    return Example(
        features=Features(
            feature={
                # I tried to use just image as a float32 numpy array but it does not work with FloatList
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

#train_filepaths = write_tfrecords("butterfly.train", training_dataset)
#write_filepaths_to_txt_file(train_filepaths, "train_filepaths.txt")
#write_filepaths_to_txt_file(train_filepaths, "train_filepaths.txt", "a")
#valid_filepaths = write_tfrecords("butterfly.valid", valid_dataset)
#write_filepaths_to_txt_file(valid_filepaths, "valid_filepaths.txt")
test_filepaths = write_tfrecords("butterfly.test", testing_dataset)
write_filepaths_to_txt_file(test_filepaths, "test_filepaths.txt")
