import os
import json
import numpy as np

import keras
import tensorflow as tf
import model as beat_model
import matplotlib.pyplot as plt

import all_config as cf
from functools import partial
from tensorflow.keras import backend as K

from train_loop_utils import train_beat_classification, retrain_freeze_beat_classification
from preprocess_data import _get_tfrecord_filenames, _preprocess_proto
print(tf.test.is_gpu_available())
tf.config.list_physical_devices('GPU')


def get_output_from_pretrained_model(pretrain_checkpoint_dir, input_data):
    with tf.device('/gpu:0'):
        train_directory = PATH + '/128_05_40_0_0_0_5_0_0.99_c1/train'
        eval_directory = PATH + '/128_05_40_0_0_0_5_0_0.99_c1/eval'
        datastore_file = PATH + '/128_05_40_0_0_0_5_0_0.99_c1/datastore.txt'
        model_dir = f'{output_dir}/model/{model_name}_1/'
        log_dir = f'{output_dir}/log/{model_name}_/'

        with open(datastore_file, 'r') as json_file:
            datastore_dict = json.load(json_file)

        feature_len = datastore_dict["feature_len"]
        beat_class = datastore_dict["beat_class"]
        num_block = datastore_dict["num_block"]
        _qrs_model_path = model_name.split('_')

        num_loop = 3 #int(_qrs_model_path[-4])
        num_filters = np.asarray([int(i) for i in _qrs_model_path[-3].split('.')], dtype=int)

        train_filenames = _get_tfrecord_filenames(train_directory, True)
        train_dataset = tf.data.TFRecordDataset(train_filenames)

        train_dataset = train_dataset.map(partial(_preprocess_proto,
                                                  feature_len=feature_len,
                                                  label_len=num_block,
                                                  class_num=len(beat_class.keys())),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train_dataset = train_dataset.shuffle(buffer_size=8192)

        train_dataset = train_dataset.batch(50)
        train_dataset = train_dataset.prefetch(50 * 5)

        val_filenames = _get_tfrecord_filenames(eval_directory, False)
        val_dataset = tf.data.TFRecordDataset(val_filenames)

        val_dataset = val_dataset.map(partial(_preprocess_proto,
                                              feature_len=feature_len,
                                              label_len=num_block,
                                              class_num=len(beat_class.keys())),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

        val_dataset = val_dataset.batch(50)
        val_dataset = val_dataset.prefetch(50 * 5)


        retrain_freeze_beat_classification(
            0,
            model_name,
            log_dir,
            model_dir,
            datastore_dict,
            True,
            train_directory,
            eval_directory,
            128,
            4,
            2,
            100,
            "/media/xuandung-ai/Data_4T1/AI-Database/Research/QRS_classification/TinyML-HeartBeat/240529_NSV_bk/128_05_40_0_0_0_6_0_0.99_c1/output/model/beat_concat_seq_add_more2_128Hz_3_8.16.32_0_0.5_1/best_squared_error_metric"
        )

        # train_model = getattr(beat_model,
        #                       'beat_concat_seq_add_more2_128Hz')(
        #     feature_len=feature_len,
        #     # num_of_class=len(beat_class),
        #     num_of_class=5,
        #     from_logits=False,
        #     filters_rhythm_net=num_filters,
        #     num_loop=num_loop,
        #     rate=float(_qrs_model_path[-1]),
        #     # name='beat_concat_seq_add_more2_other'
        # )
        #
        # train_model.summary()
        # train_model.load_weights(pretrain_checkpoint_dir)
        # test_model = beat_model.freeze_model(train_model, num_class=len(beat_class.keys()))
        # optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        # loss = keras.losses.CategoricalCrossentropy(from_logits=False)
        # # if from_logits:
        # #     metrics = [
        # #         ConfusionMatrix(classes=len(beat_class), name='confusion_matrix'),
        # #         CustomRecall(name='recall'),
        # #         CustomPrecision(name='precision'),
        # #         # CustomCategoricalAccuracy(name='accuracy')
        # #     ]
        # #     beat = [c for _, c in enumerate(beat_class.keys())]
        # #     for i in range(len(beat_class)):
        # #         metrics.append(CustomRecall(class_id=i, name='{}_Se'.format(beat[i])))
        # #         metrics.append(CustomPrecision(class_id=i, name='{}_P'.format(beat[i])))
        # # else:
        # metrics = [
        #     keras.metrics.CategoricalAccuracy(name='accuracy'),
        #     # ConfusionMatrix(classes=len(beat_class), name='confusion_matrix'),
        #     keras.metrics.Recall(name='recall'),
        #     keras.metrics.Precision(name='precision')
        # ]
        # beat = [c for _, c in enumerate(beat_class.keys())]
        # for i in range(len(beat_class)):
        #     metrics.append(keras.metrics.Recall(class_id=i, name='{}_Se'.format(beat[i])))
        #     metrics.append(keras.metrics.Precision(class_id=i, name='{}_P'.format(beat[i])))
        #
        # test_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        #
        # test_model.fit(x=train_dataset,
        #                 # # epochs=begin_at_epoch + i_epoch_num + 1,
        #                 epochs=20,
        #                 # verbose=0,
        #                 # callbacks=[log_callback],  # tf.compat.v1.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)],
        #                 # validation_data=val_dataset,
        #                 # validation_freq=[valid_freq * (x + 1) for x in
        #                 #                  range((begin_at_epoch + epoch_num) // valid_freq)],
        #                 # class_weight=cf.CLASS_WEIGHTS,
        #                 # initial_epoch=begin_at_epoch
        #                 )
    #
    # return layer_output


def main():
    pretrain_checkpoint_dir = f'{output_dir}/model/{model_name}_1/best_squared_error_metric/{model_name}-epoch-36.weights.h5'
    raw_data = ''
    get_output_from_pretrained_model(pretrain_checkpoint_dir, raw_data)


if __name__ == '__main__':
    PATH = '/media/xuandung-ai/Data_4T1/AI-Database/Research/QRS_classification/TinyML-HeartBeat/240529_NSV_bk/'
    output_dir = PATH + '/128_05_40_0_0_0_5_0_0.99_c1/output/'
    model_name = 'beat_concat_seq_add_more2_128Hz_3_8.16.32_0_0.5'

    main()


