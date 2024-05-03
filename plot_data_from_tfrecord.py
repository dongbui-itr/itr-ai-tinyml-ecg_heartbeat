import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def main(filename='/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/231003/250_10_100_0_0_0_3_0_0.99_c1_cS(p+)/train/train_00001-of-00024.tfrecord'):
    def _preprocess_proto(example_proto, feature_len, label_len, class_num) :
        """Read sample from protocol buffer."""
        encoding_scheme = {
            'sample' : tf.io.FixedLenFeature(shape=[feature_len, ], dtype=tf.float32),
            'label' : tf.io.FixedLenFeature(shape=[label_len], dtype=tf.int64),
        }
        proto = tf.io.parse_single_example(example_proto, encoding_scheme)
        sample = proto["sample"]
        label = proto["label"]
        label = tf.one_hot(label, class_num)
        return sample, label

    beat_class = {
        'NOTABEAT': [],
        'N':[],
        'S': [],
        'V': [],
        'R': []

    }

    train_dataset = tf.data.TFRecordDataset(filename)
    from functools import partial
    train_dataset = train_dataset.map(partial(_preprocess_proto,
                                              feature_len=2500,
                                              label_len=100,
                                              class_num=len(beat_class.keys())),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for data in train_dataset.take(5):
        sample = (data[0]).numpy()

        plt.plot(sample)

    plt.show()



main()

