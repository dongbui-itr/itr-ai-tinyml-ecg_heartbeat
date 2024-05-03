from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import getpass
import os
from os.path import basename, dirname
import json
import re
import numpy as np
from datetime import datetime
from run_ec57_utils import comb_run_ec57
import inputs as data_model


def ec57_run():
    test_ec57_dir = [
        ['mitdb', 'atr', 'atr'],
        ['nstdb', 'atr', 'atr'],
        ['ahadb', 'atr', 'atr'],
        ['escdb', 'atr', 'atr']
    ]
    report_dir = "/media/{}/Project/ECG/Report_TinyML".format(getpass.getuser())
    physionet_directory = '/media/{}/Dataset/ECG/PhysionetData/'.format(getpass.getuser())
    data_model_path = "/media/dev01/Project/ECG/Model/BeatNetTiny/240503/128_05.0_40_0_0_0_2_0_0.99"

    lst_output = os.listdir(data_model_path)
    lst_beat_model_path = ["{}/{}".format(data_model_path, d) for d in lst_output if "output-" in d if "best" not in d]
    DAY_TIME = datetime.today().strftime("%y%m%d%H%M")
    for beat_model_path in lst_beat_model_path:
        model_folder = os.listdir(beat_model_path + "/model")[0]
        beat_checkpoint = beat_model_path + "/model/{}/best_squared_error_metric".format(model_folder)
        beat_datastore_file = beat_model_path + '/datastore.txt'

        rhythm_checkpoint = \
            '/media/{}/Project/ECG/Model/RhythmNet/211030/256_10.0_10_0_0_0_0.7/' \
            'output-4.198500/model/rhythm_net_9_32.48.64.80.96.112.128_0_0.5/best_squared_error_metric'.format(
                getpass.getuser())
        rhythm_datastore_file = "/media/{}/Project/ECG/Model/RhythmNet/211030/256_10.0_10_0_0_0_0.7/" \
                                "output-4.198500/datastore.txt".format(getpass.getuser())
        model_name = "[{}][{}][{}][{}]".format(beat_checkpoint.split("/")[-4],
                                               beat_model_path.split("/")[-3],
                                               beat_checkpoint.split("/")[-5],
                                               beat_checkpoint.split("/")[-2])
        curr_dir = os.path.dirname(os.path.abspath(__file__))

        output_ec57_directory = '{}/{}/ec57-{}-{}/{}/'.format(
            report_dir,
            basename(curr_dir),
            data_model.MODE,
            DAY_TIME,
            model_name)
        if not os.path.isdir(output_ec57_directory):
            os.makedirs(output_ec57_directory)

        with open(beat_datastore_file, 'r') as json_file:
            beat_datastore = json.load(json_file)

        with open(rhythm_datastore_file, 'r') as json_file:
            rhythm_datastore = json.load(json_file)

        comb_run_ec57(use_gpu_index=0,
                      beat_datastore=beat_datastore,
                      beat_checkpoint=beat_checkpoint,
                      rhythm_datastore=rhythm_datastore,
                      rhythm_checkpoint=rhythm_checkpoint,
                      test_ec57_dir=test_ec57_dir,
                      output_ec57_directory=output_ec57_directory,
                      physionet_directory=physionet_directory,
                      num_of_process=8)

        # region check
        num_class = 3
        squared_error_gross = 0
        squared_error_average = 0
        for f in test_ec57_dir:
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


if __name__ == '__main__':
    ec57_run()
