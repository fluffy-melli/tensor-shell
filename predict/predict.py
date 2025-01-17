import tensorflow as tf
import numpy as np

def ReadImage(path,size):
    img = tf.keras.preprocessing.image.load_img(path, target_size=tuple(map(int, size.strip('()').split('x'))))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array