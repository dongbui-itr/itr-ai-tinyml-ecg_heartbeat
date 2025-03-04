import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from model_2D import beat_concat_seq3_250Hz
import os

def main(filename='/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/250228/250_05_78_0_0_0_6_0_0.99_c1/train/train_00010-of-00024.tfrecord'):
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
        return tf.expand_dims(tf.expand_dims(sample, axis=0), axis=-1), tf.expand_dims(label, axis=0)

    def _get_tfrecord_filenames(dir_path, is_training):
        if not os.path.exists(dir_path):
            raise FileNotFoundError("{}; No such file or directory.".format(dir_path))

        filenames = sorted(glob(os.path.join(dir_path, "*.tfrecord")))
        if not filenames:
            raise FileNotFoundError("No TFRecords found in {}".format(dir_path))

        # if is_training:
        #     shuffle(filenames)

        return filenames

    # beat_class = {
    #     'NOTABEAT': [],
    #     'N':[],
    #     'S': [],
    #     'V': [],
    #     'R': []
    #
    # }

    beat_class = {
        'NOTABEAT': [],
        'N': [],
        'V': [],
        'Q': []
    }



    a=10
    train_dataset = tf.data.TFRecordDataset(filename)
    from functools import partial
    train_dataset = train_dataset.map(partial(_preprocess_proto,
                                              feature_len=1250,
                                              label_len=78,
                                              class_num=len(beat_class.keys())),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    while True:
        try:
            element = next(train_dataset)
            # print(element)
        except Exception as eStopIteration:
        # Handle the end of the iterator
            print(filename)
            print("End of iterator: ".format(eStopIteration))
    # for data in train_dataset.take(5):
    #     sample = (data[0]).numpy()
    #     label = (data[1]).numpy()
    #
    #     plt.plot(sample)

    # plt.show()

    train_filenames = _get_tfrecord_filenames(train_directory, True)
    train_dataset = tf.data.TFRecordDataset(train_filenames)

    # train_dataset = tf.data.TFRecordDataset(filename)
    from functools import partial
    train_dataset = train_dataset.map(partial(_preprocess_proto,
                                              feature_len=1250,
                                              label_len=78,
                                              class_num=len(beat_class.keys())),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_filenames = _get_tfrecord_filenames(eval_directory, False)
    val_dataset = tf.data.TFRecordDataset(val_filenames)

    val_dataset = val_dataset.map(partial(_preprocess_proto,
                                          feature_len=1250,
                                          label_len=78,
                                          class_num=len(beat_class.keys())),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.batch(32)
    val_dataset = val_dataset.prefetch(32 * 5)

    train_dataset = train_dataset.batch(32)
    train_dataset = train_dataset.prefetch(32 * 5)

    CLASS_WEIGHTS = {
        0: 1,
        1: 4,
        # 2: 2,
        2: 3,
        3: 6,
        # 4: 1
    }

    model = beat_concat_seq3_250Hz(feature_len=1250,
                                   # model = beat_depthwise2_128Hz(feature_len=640,
                                   # model = beat_concat_sequeeze_add_more2_128Hz(feature_len=640,
                                   num_of_class=4,
                                   from_logits=False,
                                   filters_rhythm_net=[8, 16, 8],  # [8, 16, 32],
                                   num_loop=2,
                                   rate=0.5,
                                   name='beat_concat_seq3_250Hz')
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    model.fit(train_dataset,
              epochs=1,
              # class_weight=CLASS_WEIGHTS,
              validation_data=val_dataset,
              verbose=1)


train_directory = "/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/250228/250_05_78_0_0_0_6_0_0.99_c1/train/"
eval_directory = "/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/250228/250_05_78_0_0_0_6_0_0.99_c1/eval/"

from glob import glob
data_path = ("/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/250228/250_05_78_0_0_0_6_0_0.99_c1/train/")

files = glob(data_path + "*.tfrecord")
# for file in files:
#     main(file)
main()

