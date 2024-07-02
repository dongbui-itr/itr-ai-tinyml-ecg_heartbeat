import os
import re
import json
import numpy as np

import keras
import tensorflow as tf
import model as beat_model
import matplotlib.pyplot as plt

import all_config as cf
from functools import partial
from run_ec57_utils import run_ec57_tune
from train_loop_utils import train_beat_classification, retrain_freeze_beat_classification
from preprocess_data import _get_tfrecord_filenames, _preprocess_proto

print(tf.test.is_gpu_available())
tf.config.list_physical_devices('GPU')


def retrain_model(
        retrain=False,
        run_ec57=False
):
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

        num_loop = 3  # int(_qrs_model_path[-4])
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

        if retrain:
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
                f"{PATH}/128_05_40_0_0_0_6_0_0.99_c1/output/model/{model_name}_1/best_squared_error_metric"
            )

        if run_ec57:
            # region ec57
            checkpoint_dir = "{}/best_squared_error_metric".format(model_dir)
            output_ec57_directory = '{}/ec57/{}/'.format(output_dir, model_name)
            if not os.path.isdir(output_ec57_directory):
                os.makedirs(output_ec57_directory)
            try:
                run_ec57_tune(use_gpu_index=0,
                              model_name=model_name,
                              datastore_file=datastore_file,
                              checkpoint_dir=checkpoint_dir,
                              test_ec57_dir=cf.DB_TESTING,
                              output_ec57_directory=output_ec57_directory,
                              physionet_directory=cf.PATH_DATA_EC57,
                              overlap=5,
                              num_of_process=2)

                # endregion
                # region check
                num_class = len(beat_class.keys())
                squared_error_gross = 0
                squared_error_average = 0
                for f in cf.DB_TESTING:
                    f = output_ec57_directory + "/{}".format(f[0])
                    path_out = '{}/{}_QRS_report_line.out'.format(f, os.path.basename(f))
                    file_out = open(path_out)
                    content = file_out.readlines()
                    gross_value = re.findall(r'[-+]?\d*\.\d+|\d+|[-]', content[-5])
                    average_value = re.findall(r'[-+]?\d*\.\d+|\d+|[-]', content[-4])

                    _keys = ["N(se)", "N(p+)", "V(se)", "V(p+)", "S(se)", "S(p+)"]

                    for n, k in enumerate(_keys):
                        if n < len(gross_value) and \
                                "-" not in gross_value[n] and \
                                (num_class < 0 or n < num_class * 2):
                            squared_error_gross += np.square(1 - float(gross_value[n]) / 100)
                            squared_error_average += np.square(1 - float(average_value[n]) / 100)
                        elif n < len(gross_value) and \
                                "-" in gross_value[n] and \
                                (num_class < 0 or n < num_class * 2):
                            squared_error_gross += np.square(1 - float(0) / 100)
                            squared_error_average += np.square(1 - float(0) / 100)

                fstatus = open(output_ec57_directory + '/squared_error.txt', 'w')
                fstatus.writelines('squared_error_gross: {:.6f}\n'.format(squared_error_gross))
                fstatus.writelines('squared_error_average: {:.6f}\n'.format(squared_error_average))
                fstatus.close()
                # endregion

            except Exception as err:
                print('Error at run_ec57:', err)


if __name__ == '__main__':
    PATH = '/media/xuandung-ai/Data_4T1/AI-Database/Research/QRS_classification/TinyML-HeartBeat/240529_NSV_bk/'
    output_dir = PATH + '/128_05_40_0_0_0_5_0_0.99_c1/output/'
    model_name = 'beat_concat_seq_add_more2_128Hz_3_8.16.32_0_0.5'

    retrain_model(retrain=False, run_ec57=True)
