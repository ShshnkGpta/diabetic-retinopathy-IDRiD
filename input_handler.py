import os
import cv2
import sys
import csv
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.dirname('D:/Projects/diabetic_retinopathy_IDRD/Disease Grading/')
train_images = os.path.join(PATH, 'Images/Training Set')
test_images = os.path.join(PATH, 'Images/Testing Set')
label_path = os.path.join(PATH, 'Labels')


data_list = []

f = open(os.path.join(label_path, "Training_Labels.csv"), 'r')
reader = csv.reader(f)
header = next(reader)
for row in reader:
  tmp = [None, None]
  tmp2 = row[0] + ".jpg"
  tmp[0] = os.path.join(train_images, tmp2)
  tmp[1] = row[1]
  data_list.append(tmp)
f.close()


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

train_filename = 'train.tfrecords' 

writer = tf.io.TFRecordWriter(train_filename)
for i in tqdm(range(len(data_list))):
    # Load the image
    img = cv2.imread(data_list[i][0])
    #img = cv2.resize(img, (50, 50))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.io.serialize_tensor(img_array)
    image_shape = img_array.shape
    label = data_list[i][1]
    # Create a feature
    feature = {
              'train/label': _int64_feature(int(label)),
              'train/image': _bytes_feature(img),
              'height'     : _int64_feature(image_shape[0]),
              'width'      : _int64_feature(image_shape[1]),
              'depth'      : _int64_feature(image_shape[2]),
              }
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()
