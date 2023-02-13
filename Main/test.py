import tensorflow as tf
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=True))
print(tf.config.list_physical_devices('GPU'))

import tensorflow as tf

print(tf.reduce_sum(tf.random.normal([1000, 1000])))

import keras
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())