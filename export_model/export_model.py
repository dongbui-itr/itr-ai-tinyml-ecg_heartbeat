import csv
import json
import numpy as np
import tensorflow as tf
# import model as model_new
import model_2D as model_new
import model_old as model_old
import os
import getpass
import datetime
import glob

from export_model_tf2 import export_model


def main():
    beat_model_path = "/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/241204/250_05_78_0_0_0_5_0_0.99_c2/"
                       # "beat_concat_seq_add_more2_128Hz_3_8.16.32_0_0.5_2/best_squared_error_metric/beat_concat_seq_add_more2_128Hz_3_8.16.32_0_0.5-epoch-8.weights.h5")

    model_name = os.listdir(beat_model_path + "/output/model")[0]
    beat_checkpoint = beat_model_path + "/output/model/{}/best_squared_error_metric".format(model_name)
    # beat_checkpoint = beat_model_path + "/model/{}/last".format(model_name)
    beat_datastore_file = beat_model_path + '/datastore.txt'

    _case = model_name.replace('/', '').replace('=', '').replace('-', '').replace('_', '').replace('.', '')
    ext_ai = ""
    for c in _case.strip():
        if c.isdigit():
            ext_ai += str.lower(chr(int(c) + 65))
        else:
            ext_ai += c

    with open(beat_datastore_file, 'r') as json_file:
        datastore_dict = json.load(json_file)

    feature_len = datastore_dict["feature_len"]
    beat_class = datastore_dict["beat_class"]

    _qrs_model_path = model_name.split('_')
    func = ""
    m = 0
    for m in range(len(_qrs_model_path)):
        if _qrs_model_path[m].isnumeric():
            break
        else:
            func += _qrs_model_path[m] + "_"

    func = func[:-1]

    day_export = beat_checkpoint.split('/')[5]
    day_export = day_export.split('_')[0]
    day_export = datetime.datetime.strptime(day_export, '%y%m%d')
    # day_export = datetime.datetime.strptime(day_export, '%y%m%d')
    num_loop = int(_qrs_model_path[m])
    num_filters = np.asarray([int(i) for i in _qrs_model_path[m + 1].split('.')], dtype=int)
    try:
        from_logits = bool(int(_qrs_model_path[m + 2]))
    except:
        from_logits = False
    if day_export < datetime.datetime(2021, 12, 1):
        beat_model = getattr(model_old, func)(feature_len,
                                              len(beat_class),
                                              from_logits,
                                              num_filters,
                                              num_loop,
                                              0.5,
                                              False)
    else:
        beat_model = getattr(model_new, func)(feature_len,
                                          len(beat_class),
                                          from_logits,
                                          num_filters,
                                          num_loop,
                                          0.5,
                                          False)

    beat_model.summary()

    # last_model = tf.train.latest_checkpoint(beat_checkpoint)
    check_point = glob.glob(beat_checkpoint + '/*.h5')[0]
    beat_model.load_weights(check_point)

    export_model(beat_model, output_path=beat_checkpoint + '_export', signatures='beats')


    # # datastore_file = DATAPATH + '/' + '/datastore.txt'
    # with open(datastore_file, 'r') as json_file:
    #     datastore_dict = json.load(json_file)
    #
    # feature_len = datastore_dict["feature_len"]
    # rhythm_class = datastore_dict["rhythm_class"]
    #
    # _qrs_model_path = MODEL_NAME.split('_')
    # func = ""
    # m = 0
    # for m in range(len(_qrs_model_path)):
    #     if _qrs_model_path[m].isnumeric():
    #         break
    #     else:
    #         func += _qrs_model_path[m] + "_"
    # func = func[:-1]
    #
    # if 'rhythm' in func:
    #     num_loop = int(_qrs_model_path[m])
    #     num_filters = np.asarray([int(i) for i in _qrs_model_path[m + 1].split('.')], dtype=int)
    #     try:
    #         from_logits = bool(int(_qrs_model_path[m + 2]))
    #     except:
    #         from_logits = False
    #
    #     model = getattr(beat_model, func)(feature_len,
    #                                       len(rhythm_class),
    #                                       from_logits,
    #                                       num_filters,
    #                                       num_loop)
    #     model.summary()
    #     model.load_weights(tf.train.latest_checkpoint(CKPT_DIR)).expect_partial()
    #     export_model(model, output_path=CKPT_DIR + '_export', signatures='rhythms')
    # exit()


main()
