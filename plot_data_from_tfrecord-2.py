import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from model_2D import beat_concat_seq3_250Hz
import os

from glob import glob

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
    # from functools import partial
    # train_dataset = train_dataset.map(partial(_preprocess_proto,
    #                                           feature_len=1250,
    #                                           label_len=78,
    #                                           class_num=len(beat_class.keys())),
    #                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # i = 100
    # print(filename)
    # cnt = 0
    # try:
    #     # # element = train_dataset.take()
    #     count_test = train_dataset.repeat(1)
    #     count_test = count_test.batch(1)
    #     test_counter = count_test.make_one_shot_iterator()
    #     # # i += 100
    #     # print(element)
    #     cnt = np.sum(1 for _ in test_counter)
    #     a=10
    # except Exception as eStopIteration:
    #     # Handle the end of the iterator
    #     #     print(filename)
    #         print("End of iterator: ".format(eStopIteration))

    # print(f'Total of samples in {filename}: {cnt}')
    try:
        cnt = 0
        for data in train_dataset.take(10):
            print(cnt)
            cnt += 1
            # sample = (data[0]).numpy().flatten()
            # label = (data[1]).numpy()
            #
            # plt.plot(sample)
            # plt.show()
            example = tf.train.Example()
            example.ParseFromString(data.numpy())
            result = {}
            for key, feature in example.features.feature.items():
                # The values are the Feature objects which contain a `kind` which contains:
                # one of three fields: bytes_list, float_list, int64_list

                kind = feature.WhichOneof('kind')
                result[key] = np.array(getattr(feature, kind).value)

            if len(result['sample']) != 1250 or len(result['label']) != 78:
                print(result)
            # a=10
    # except Exception as err:
    except tf.errors.OutOfRangeError:
        print("tf.errors.OutOfRangeError")
    a=10
    # train_filenames = _get_tfrecord_filenames(train_directory, True)
    # train_dataset = tf.data.TFRecordDataset(train_filenames)
    #
    # # train_dataset = tf.data.TFRecordDataset(filename)
    # from functools import partial
    # train_dataset = train_dataset.map(partial(_preprocess_proto,
    #                                           feature_len=1250,
    #                                           label_len=78,
    #                                           class_num=len(beat_class.keys())),
    #                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #
    # val_filenames = _get_tfrecord_filenames(eval_directory, False)
    # val_dataset = tf.data.TFRecordDataset(val_filenames)
    #
    # val_dataset = val_dataset.map(partial(_preprocess_proto,
    #                                       feature_len=1250,
    #                                       label_len=78,
    #                                       class_num=len(beat_class.keys())),
    #                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #
    # val_dataset = val_dataset.batch(32)
    # val_dataset = val_dataset.prefetch(32 * 5)
    #
    # train_dataset = train_dataset.batch(32)
    # train_dataset = train_dataset.prefetch(32 * 5)
    #
    # CLASS_WEIGHTS = {
    #     0: 1,
    #     1: 4,
    #     # 2: 2,
    #     2: 3,
    #     3: 6,
    #     # 4: 1
    # }
    #
    # model = beat_concat_seq3_250Hz(feature_len=1250,
    #                                # model = beat_depthwise2_128Hz(feature_len=640,
    #                                # model = beat_concat_sequeeze_add_more2_128Hz(feature_len=640,
    #                                num_of_class=4,
    #                                from_logits=False,
    #                                filters_rhythm_net=[8, 16, 8],  # [8, 16, 32],
    #                                num_loop=2,
    #                                rate=0.5,
    #                                name='beat_concat_seq3_250Hz')
    # model.summary()
    # model.compile(optimizer='adam', loss='binary_crossentropy')
    # print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    # model.fit(train_dataset,
    #           epochs=1,
    #           # class_weight=CLASS_WEIGHTS,
    #           validation_data=val_dataset,
    #           verbose=1)


train_directory = "/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/250402/250_05_78_0_0_0_6_0_0.99_c1/train/"
eval_directory = "/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/250402//250_05_78_0_0_0_6_0_0.99_c1/eval/"


# data_path = ("/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/250228/250_05_78_0_0_0_6_0_0.99_c1/train/")
# files = glob(data_path + "*.tfrecord")

# files = glob(eval_directory + "*.tfrecord")
files = glob(eval_directory + "eval_00008-of-00024.tfrecord")

for file in files:
    print(file)
    main(file)
# main()

