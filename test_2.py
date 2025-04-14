# import tensorflow as tf
# device_name = tf.test.gpu_device_name()
# if device_name == '':
#     device_name = "None"
# print('Using TensorFlow version:', tf.__version__, ', GPU:', device_name)

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import os
from glob import glob

data_path = "/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/250402/250_05_78_0_0_0_6_0_0.99_c1/*/*.txt"

files = glob(data_path)

total_training_sample = 0
total_eval_sample = 0
for file in files:
    fp = open(file)
    lines = fp.readlines()
    if "/train/" in file:
        total_training_sample += len(lines)
    else:
        total_eval_sample += len(lines)
    for line in lines:
        if "Error" in line:
            print(file)
            break

print(f"Total of Train-sample: {total_training_sample}")
print(f"Total of Eval-sample: {total_eval_sample}")
print(total_training_sample//16)

