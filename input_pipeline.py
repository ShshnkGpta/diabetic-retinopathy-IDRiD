import tensorflow as tf
import os
import matplotlib.pyplot as plt

def parse(serialized):
    features = {
        'train/label': tf.io.FixedLenFeature([], tf.int64),
        'train/image': tf.io.FixedLenFeature([], tf.string),
        'height'     : tf.io.FixedLenFeature((), tf.int64),
        'width'      : tf.io.FixedLenFeature((), tf.int64),
        'depth'      : tf.io.FixedLenFeature((), tf.int64)
    }

    example        = tf.io.parse_single_example(serialized=serialized,features=features)
    image          = tf.io.parse_tensor(example['train/image'], out_type = float)
    image_shape    = [example['height'], example['width'], example['depth']]
    image          = tf.reshape(image, image_shape)
    label          = example['train/label']
    return image, label



def getInput(batch_size):
    filename  = os.path.join(os.path.dirname(os.path.realpath(__file__)), "train.tfrecords")
    filenames = [filename]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(20)
    dataset = dataset.prefetch(8)
    return dataset

"""
dataset = getInput(20)
for i, data in enumerate(dataset.take(9)):
    img = tf.keras.preprocessing.image.array_to_img(data[0][0])
    print(data[1][0])
    plt.subplot(3,3,i+1)
    plt.imshow(img)
"""