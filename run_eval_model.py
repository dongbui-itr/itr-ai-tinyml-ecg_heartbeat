from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import getpass
import os
import re
import numpy as np
from os.path import basename, dirname
from datetime import datetime
import inputs as data_model

from run_ec57_utils import run_ec57


def portal_run():
    DB_PATH = '/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/Collection_20240510/export_noise/'
    curr_dir = os.getcwd()
    physionet_directory = '/media/{}/MegaProject/PhysionetData/'.format(getpass.getuser())
    beat_model_path = "/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/240503/128_05_20_0_0_0_4_0_0.99_c2/output/"

    model_name = os.listdir(beat_model_path + "/model")[0]
    beat_checkpoint = beat_model_path + "/model/{}/best_squared_error_metric".format(model_name)
    beat_datastore_file = beat_model_path + '/datastore.txt'

    output_ec57_directory = beat_model_path + '/ec57-{}-{}/{}/'.format(data_model.MODE,
                                                                       datetime.today().strftime("%y%m%d%H%M"),
                                                                       model_name)
    if not os.path.isdir(output_ec57_directory):
        os.makedirs(output_ec57_directory)

    dir_image = curr_dir + "/img"
    if not os.path.isdir(dir_image):
        os.makedirs(dir_image)

    run_ec57(use_gpu_index=0,
             model_name=model_name,
             datastore_file=beat_datastore_file,
             checkpoint_dir=beat_checkpoint,
             test_ec57_dir=DB_TESTING,
             output_ec57_directory=output_ec57_directory,
             physionet_directory=physionet_directory,
             overlap=0,
             num_of_process=8,
             dir_image=None)

    # region check
    num_class = 3
    squared_error_gross = 0
    squared_error_average = 0
    for f in DB_TESTING:
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
    portal_run()
