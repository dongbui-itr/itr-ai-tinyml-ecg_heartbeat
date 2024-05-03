from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import time
import os
from os.path import basename
import json
import datetime
from utils.ec57_test import ec57_eval, del_result, bxb_eval, del_result2
from glob import glob
from multiprocessing import Pool
from run_ec57_multiprocess_utils import process_beat_rhythm_classification, process_beat_classification
from all_config import EXT_BEAT_EVAL


def run_ec57(use_gpu_index,
             model_name,
             datastore_file,
             checkpoint_dir,
             test_ec57_dir,
             output_ec57_directory,
             physionet_directory,
             overlap,
             num_of_process,
             dir_image=None):
    """

    """

    if num_of_process is None or num_of_process == 0:
        num_of_process = os.cpu_count()

    with open(datastore_file, 'r') as json_file:
        datastore_dict = json.load(json_file)

    if os.path.exists(output_ec57_directory + '/finish.txt'):
        return

    fstatus = open(output_ec57_directory + '/start.txt', 'w')
    fstatus.writelines(str(datetime.datetime.now()))
    fstatus.close()

    _case = model_name.replace('/', '').replace('=', '').replace('-', '').replace('_', '').replace('.', '')
    ext_ai = ""
    for c in _case.strip():
        if c.isdigit():
            ext_ai += str.lower(chr(int(c) + 65))
        else:
            ext_ai += c

    with open(output_ec57_directory + '/eval_timing.txt', 'w') as log_eval:
        for db in test_ec57_dir:
            del_result(db[0], physionet_directory, output_ec57_directory)
            path2db = physionet_directory + db[0]

            file_names = glob(path2db + '/*.dat')
            # Get rid of the extension
            file_names = [p[:-4] for p in file_names
                          if basename(p)[:-4] not in ['104', '102', '107', '217', 'bw', 'em', 'ma']
                          if '_200hz' not in basename(p)[:-4]]

            file_names = sorted(file_names)

            num_file_each_process = int(len(file_names) / num_of_process)
            while num_file_each_process == 0:
                num_of_process -= 1
                num_file_each_process = int(len(file_names) / num_of_process)

            file_process_split = [file_names[x:x + num_file_each_process] for x in range(0, len(file_names),
                                                                                         num_file_each_process)]
            arg_list = list()
            for i, file_list in enumerate(file_process_split):
                arg = (i,
                       use_gpu_index,
                       file_list,
                       model_name,
                       checkpoint_dir,
                       datastore_dict,
                       ext_ai,
                       True,
                       False,
                       0,
                       overlap,
                       1024,
                       dir_image)

                # process_beat_classification(i,
                #        use_gpu_index,
                #        file_list,
                #        model_name,
                #        checkpoint_dir,
                #        datastore_dict,
                #        ext_ai,
                #        True,
                #        False,
                #        0,
                #        overlap,
                #        1024,
                #        dir_image)

                arg_list.append(arg)

            process_start_time = time.time()
            with Pool(processes=num_of_process) as pool:
                # print same numbers in arbitrary order
                for log_lines in pool.starmap(process_beat_classification, arg_list):
                    log_eval.writelines(log_lines)

            log_eval.writelines(log_lines)

            process_end_time = time.time()
            str_log = 'multiprocess {} take {} seconds\n'.format(basename(os.path.dirname(file_names[0])),
                                                                 process_end_time - process_start_time)
            print(str_log)

            # EC57 Eval-Full db
            ec57_eval(db[0],
                      output_ec57_directory,
                      physionet_directory,
                      db[1],
                      db[2],
                      ext_ai,
                      None)

            del_result(db[0], physionet_directory, output_ec57_directory)

    fstatus = open(output_ec57_directory + '/finish.txt', 'w')
    fstatus.writelines(str(datetime.datetime.now()))
    fstatus.close()
    return ext_ai


def comb_run_ec57(use_gpu_index,
                  beat_datastore,
                  beat_checkpoint,
                  rhythm_datastore,
                  rhythm_checkpoint,
                  test_ec57_dir,
                  output_ec57_directory,
                  physionet_directory,
                  num_of_process):
    """

    """

    if num_of_process is None or num_of_process == 0:
        num_of_process = os.cpu_count()

    if os.path.exists(output_ec57_directory + '/finish.txt'):
        return

    fstatus = open(output_ec57_directory + '/start.txt', 'w')
    fstatus.writelines(str(datetime.datetime.now()))
    fstatus.close()
    beat_ext_tech = None
    beat_ext_ai = "aib"
    rhythm_ext_ai = "air"

    with open(output_ec57_directory + '/eval_timing.txt', 'w') as log_eval:
        for db in test_ec57_dir:
            del_result(db[0], physionet_directory, output_ec57_directory)
            path2db = physionet_directory + db[0]

            file_names = glob(path2db + '/*.dat')
            # Get rid of the extension
            file_names = [p[:-4] for p in file_names
                          if basename(p)[:-4] not in ['104', '102', '107', '217', 'bw', 'em', 'ma']
                          if '_200hz' not in basename(p)[:-4]]

            file_names = sorted(file_names)

            num_file_each_process = int(len(file_names) / num_of_process)
            while num_file_each_process == 0:
                num_of_process -= 1
                num_file_each_process = int(len(file_names) / num_of_process)

            file_process_split = [file_names[x:x + num_file_each_process] for x in range(0, len(file_names),
                                                                                         num_file_each_process)]
            arg_list = list()
            for i, file_list in enumerate(file_process_split):
                arg = (i,
                       use_gpu_index,
                       file_list,
                       beat_datastore,
                       beat_checkpoint,
                       rhythm_datastore,
                       rhythm_checkpoint,
                       beat_ext_tech,
                       beat_ext_ai,
                       rhythm_ext_ai,
                       None,
                       True,
                       False,
                       1024)
                arg_list.append(arg)

            process_start_time = time.time()
            lst_true_all = []
            lst_pred_all = []
            lst_file_all = []
            with Pool(processes=num_of_process) as pool:
                # print same numbers in arbitrary order
                for log_lines, lst_file, lst_true, lst_pred in pool.starmap(process_beat_rhythm_classification, arg_list):
                    log_eval.writelines(log_lines)
                    lst_true_all += lst_true
                    lst_pred_all += lst_pred
                    lst_file_all += lst_file

            log_eval.writelines(log_lines)
            process_end_time = time.time()
            str_log = 'multiprocess {} take {} seconds\n'.format(basename(os.path.dirname(file_names[0])),
                                                                 process_end_time - process_start_time)
            print(str_log)

            # EC57 Eval-Full db
            ec57_eval(db[0],
                      output_ec57_directory,
                      physionet_directory,
                      db[1],
                      db[2],
                      beat_ext_ai,
                      rhythm_ext_ai)

            del_result(db[0], physionet_directory, output_ec57_directory)

    fstatus = open(output_ec57_directory + '/finish.txt', 'w')
    fstatus.writelines(str(datetime.datetime.now()))
    return lst_true_all, lst_pred_all, lst_file_all


def comb_run_eval(use_gpu_index,
                  beat_datastore,
                  beat_checkpoint,
                  rhythm_datastore,
                  rhythm_checkpoint,
                  eval_data_dir,
                  output_eval_directory,
                  num_of_process):
    """

    """

    if num_of_process is None or num_of_process == 0:
        num_of_process = os.cpu_count()

    tmp_directory = output_eval_directory + "/tmp"
    if not os.path.isdir(tmp_directory):
        os.makedirs(tmp_directory)

    # if os.path.exists(output_eval_directory + '/finish.txt'):
    #     return

    fstatus = open(output_eval_directory + '/start.txt', 'w')
    fstatus.writelines(str(datetime.datetime.now()))
    fstatus.close()

    beat_ext_tech = "atb"
    beat_ext_ai = "aib"
    rhythm_ext_ai = "air"

    with open(output_eval_directory + '/eval_timing.txt', 'w') as log_eval:
        file_names = glob(eval_data_dir + '/*/*/*.{}'.format(EXT_BEAT_EVAL))
        # Get rid of the extension
        file_names = [p[:-4] for p in file_names]
        file_names = sorted(file_names)
        num_file_each_process = int(len(file_names) / num_of_process)
        while num_file_each_process == 0:
            num_of_process -= 1
            num_file_each_process = int(len(file_names) / num_of_process)

        file_process_split = [file_names[x:x + num_file_each_process] for x in range(0, len(file_names),
                                                                                     num_file_each_process)]
        arg_list = list()
        for i, file_list in enumerate(file_process_split):
            arg = (i,
                   use_gpu_index,
                   file_list,
                   beat_datastore,
                   beat_checkpoint,
                   rhythm_datastore,
                   rhythm_checkpoint,
                   beat_ext_tech,
                   beat_ext_ai,
                   rhythm_ext_ai,
                   tmp_directory,
                   True,
                   False,
                   1024)
            arg_list.append(arg)

        process_start_time = time.time()
        lst_true_all = []
        lst_pred_all = []
        lst_file_all = []
        with Pool(processes=num_of_process) as pool:
            # print same numbers in arbitrary order
            for log_lines, lst_file, lst_true, lst_pred in pool.starmap(process_beat_rhythm_classification, arg_list):
                log_eval.writelines(log_lines)
                lst_true_all += lst_true
                lst_pred_all += lst_pred
                lst_file_all += lst_file

        log_eval.writelines(log_lines)

        process_end_time = time.time()
        str_log = 'multiprocess {} take {} seconds\n'.format(basename(os.path.dirname(file_names[0])),
                                                             process_end_time - process_start_time)
        print(str_log)

        # EC57 Eval-Full db
        bxb_eval(output_eval_directory,
                 tmp_directory,
                 beat_ext_tech,
                 None,
                 beat_ext_ai,
                 None,
                 )

        del_result2(tmp_directory, output_eval_directory)

    fstatus = open(output_eval_directory + '/finish.txt', 'w')
    fstatus.writelines(str(datetime.datetime.now()))
    if os.path.isdir(tmp_directory):
        shutil.rmtree(tmp_directory)

    return lst_true_all, lst_pred_all, lst_file_all