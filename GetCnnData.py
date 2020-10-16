import os
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
 
 
#load image directory
train_data_dir = r'C:\important\python\dataset\train_data'
classes = []
image_list = []
label_list = []
 
 
 
def get_files(file_path,ratdio):
    for str_classes in os.listdir(train_data_dir):
        classes.append(str_classes)
    for index, name in enumerate(classes):
        path = file_path + '\\'+name
        for file in os.listdir(path):
            image_list.append(path + '\\' + file)
            label_list.append(index)
        print(name,'ok')
   
   #disorganize the image
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
 
   #get training data and testing data
    image_data = list(temp[:,0])
    image_label = list(temp[:,1])
 
    n_sample = len(image_label)
    n_val = int(math.ceil(n_sample * ratdio))
    n_train = n_sample - n_val
 
    train_images = image_data[0:n_train]
    train_labels = image_label[0:n_train]
    train_labels = [int(float(i)) for i in train_labels]
    val_images = image_data[n_train:-1]
    val_labels = image_label[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
 
    return train_images,train_labels,val_images,val_labels
 
def get_batch(image,label,image_W,image_H,batch_size,capacity):
    #unify the data type
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
    #make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  
    #decode the image(jpeg)
    image = tf.image.decode_jpeg(image_contents, channels=3)
 
  	#resize, cut, standardize the image
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
 
    #image_batch: 4D tensor[batch_size, width, height, 3], dtype = tf.float32
    #label_batch: 1D tensor [batch_size]
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
   	
   	#reorder label, number of rows = [batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch
