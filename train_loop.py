# Define paths to model files
import getpass
import os
import json
import re
import inputs as data_model
from train_loop_utils import train_beat_classification
from run_ec57_utils import run_ec57
import shutil
import numpy as np
import multiprocessing
# from distutils.dir_util import copy_tree
from datetime import datetime
from all_config import DB_TESTING, PATH_DATA_EC57

import copy

def train():
    MAX_EPOCH = 50
    dir_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    # MEDIA_PATH = '/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/{}/'.format(datetime.today().strftime("%y%m%d"))
    MEDIA_PATH = '/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/{}/'.format(datetime.today().strftime("250505"))
    if not os.path.exists(MEDIA_PATH):
        os.makedirs(MEDIA_PATH)

    DATA_SOURCE = '/mnt/4T_DATA/DATA_4TINYML/tiny_hb3_data_strip_2/'

    PATH_DATA_TRAINING = PATH_DATA_EC57

    # sampling_rate
    # feature_len
    # num_block
    # bwr
    # norm
    # overlap
    # class_index
    # add_artifact
    # percent_train

    DATA = [
        # '128_05_40_0_0_0_6_0_0.99', #240529_NSV
        # '128_05_40_0_0_0_7_0_0.99', #240712_NSV
        '250_05_78_0_0_0_7_0_0.99'
    ]
    BATCH_SIZE_TRAINING = [
        512,
    ]

    MODEL = [
        # "beat_concat_seq_add_more_other_7_16.32.48.64_0_0.5",
        # "beat_concat_seq_add_more_other_11_16.32.48.64_0_0.5",
        # "beat_concat_seq_add_more2_128Hz_3_8.16.32_0_0.5", #240529_NSV
        # "beat_concat_seq_add_depthwise_250Hz_7_16.32.48_0_0.5",
        # "beat_depthwise2_128Hz_7_32.32.48_0_0.5",
        # "beat_seq_mobilenet_v2_1d_0_0.0.0_0_0.5",
        # "beat_seq_mobilenet_v2keras_1d_0_0.0.0_0_0.5",
        # "beat_concat_seq3_250Hz_2_8.16.32_0_0.5",
        # "beat_concat_seq3_250Hz_2_8.16.8_0_0.5",
        "beat_concat_seq3_250Hz_2_8.24.8_0_0.5",
    ]

    # MobileNetv2_1D(input_shape, num_of_class, k, alpha=1.0, rate=0.5):

    THR = 7
    k = 0
    for batch_size, pt in zip(BATCH_SIZE_TRAINING, DATA):
        k += 1
        # sl.split_data_2(DATA_SOURCE, MEDIA_PATH, k)
        num_try_on_with_dataset = 1
        try_on_with_dataset = 0
        squared_error_gross_ec57 = THR
        count = 0
        pt_bk = copy.copy(pt)
        flag_reset_data = False
        while try_on_with_dataset < num_try_on_with_dataset and count < 30:
        # while count < 1:
            count += 1
            # sl.split_data_2(DATA_SOURCE, MEDIA_PATH, count)
            # pt = pt + '_d{}_c{}'.for/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/240529_NSV//128_05_40_0_0_0_5_0_0.99_c1/datastore.txt'mat(k, count)
            pt = copy.copy(pt_bk)
            pt = pt + '_c{}'.format(count)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {} try on {} get {} "
                  "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(pt, count, try_on_with_dataset))
            # region random create data_input
            data_model_dir = MEDIA_PATH + '/{}/'.format(pt)
            datastore_file = MEDIA_PATH + '/' + pt + '/datastore.txt'
            ds_eval_file = MEDIA_PATH + '/' + pt + '/ds_eval.txt'
            ds_train_file = MEDIA_PATH + '/' + pt + '/ds_train.txt'

            ds_study_eval = MEDIA_PATH + '/' + pt + '/ds_study_eval.txt'
            ds_study_train = MEDIA_PATH + '/' + pt + '/ds_study_train.txt'
            ds_study_info = MEDIA_PATH + '/' + pt + '/ds_study_info.txt'
            ds_train_study_info = MEDIA_PATH + '/' + pt + '/ds_train_study_info.txt'
            ds_eval_study_info = MEDIA_PATH + '/' + pt + '/ds_eval_study_info.txt'

            all_beat_type = MEDIA_PATH + '/' + pt + '/all_beat_type.txt'
            all_train_beat_type = MEDIA_PATH + '/' + pt + '/all_train_beat_type.txt'
            all_eval_beat_type = MEDIA_PATH + '/' + pt + '/all_eval_beat_type.txt'

            train_beat_type = MEDIA_PATH + '/' + pt + '/train_beat_type.txt'

            finish_file = MEDIA_PATH + '/' + pt + '/finish.txt'
            start_file = MEDIA_PATH + '/' + pt + '/start.txt'

            output_dir = MEDIA_PATH + '/' + pt + '/output'
            train_directory = MEDIA_PATH + '/' + pt + '/train'
            eval_directory = MEDIA_PATH + '/' + pt + '/eval'
            # if os.path.exists(datastore_file):
            #     if os.path.exists(train_directory):
            #         shutil.rmtree(train_directory)
            #
            #     if os.path.exists(eval_directory):
            #         shutil.rmtree(eval_directory)
            #
            #     if os.path.exists(output_dir):
            #         shutil.rmtree(output_dir)
            #
            #     if os.path.exists(datastore_file):
            #         os.remove(datastore_file)
            #
            #     if os.path.exists(ds_eval_file):
            #         os.remove(ds_eval_file)
            #
            #     if os.path.exists(ds_train_file):
            #         os.remove(ds_train_file)
            #
            #     if os.path.exists(ds_study_eval):
            #         os.remove(ds_study_eval)
            #
            #     if os.path.exists(ds_study_train):
            #         os.remove(ds_study_train)
            #
            #     if os.path.exists(ds_study_info):
            #         os.remove(ds_study_info)
            #
            #     if os.path.exists(ds_train_study_info):
            #         os.remove(ds_train_study_info)
            #
            #     if os.path.exists(ds_eval_study_info):
            #         os.remove(ds_eval_study_info)
            #
            #     if os.path.exists(all_beat_type):
            #         os.remove(all_beat_type)
            #
            #     if os.path.exists(all_train_beat_type):
            #         os.remove(all_train_beat_type)
            #
            #     if os.path.exists(all_eval_beat_type):
            #         os.remove(all_eval_beat_type)
            #
            #     if os.path.exists(train_beat_type):
            #         os.remove(train_beat_type)
            #
            #     os.remove(finish_file)
            #     os.remove(start_file)
            #
            # # data_model.create_tfrecord_from_portal_event_2(data_model_dir=data_model_dir,
            # data_model.create_tfrecord_from_portal_event_3(data_model_dir=data_model_dir,
            #                                               data_dir=DATA_SOURCE,
            #                                               media_dir=MEDIA_PATH,
            #                                               save_image=False,
            #                                               org_num_processes=os.cpu_count(),
            #                                               org_num_shards=os.cpu_count())
            # # endregion random create data_input

            with open(datastore_file, 'r') as json_file:
                datastore_dict = json.load(json_file)

            for i, model_name in enumerate(MODEL):
                model_dir = '{}/model/{}_{}'.format(output_dir, model_name, count)
                log_dir = '{}/log/{}_'.format(output_dir, model_name, count)

                if not os.path.exists(output_dir):
                    # shutil.move(output_dir, MEDIA_PATH + f'm{i}')
                    # shutil.rmtree(output_dir)
                    os.makedirs(output_dir)

                for i in [model_dir, log_dir]:
                    if not os.path.exists(i):
                        os.makedirs(i)

                # region training
                process_train = multiprocessing.Process(target=train_beat_classification,
                                                        args=(0,
                                                              model_name,
                                                              log_dir,
                                                              model_dir,
                                                              datastore_dict,
                                                              None,
                                                              train_directory,
                                                              eval_directory,
                                                              batch_size,
                                                              4,
                                                              2,
                                                              MAX_EPOCH))
                process_train.start()
                process_train.join()
                # endregion training

                # region ec57
                # checkpoint_dir = "{}/best_squared_error_metric".format(model_dir)
                checkpoint_dir = "{}/best_avg".format(model_dir)
                output_ec57_directory = '{}/ec57/{}/'.format(output_dir, model_name)
                if not os.path.isdir(output_ec57_directory):
                    os.makedirs(output_ec57_directory)
                try:
                    run_ec57(use_gpu_index=0,
                             model_name=model_name,
                             datastore_file=datastore_file,
                             checkpoint_dir=checkpoint_dir,
                             test_ec57_dir=DB_TESTING,
                             output_ec57_directory=output_ec57_directory,
                             physionet_directory=PATH_DATA_TRAINING,
                             overlap=5,
                             num_of_process=3)

                except Exception as err:
                    print('Error at run_ec57:', err)

if __name__ == '__main__':
    train()
