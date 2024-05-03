from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import getpass
import os
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import json
import re
import numpy as np
from run_ec57_utils import comb_run_eval
from datetime import datetime, timedelta
import inputs as data_model


def eval_run():
    DATA_PATH = '/media/{}/Dataset/ECG/PortalData/BeatNetEval/2021-09-30'.format(getpass.getuser())
    report_dir = "/media/{}/Project/ECG/Report".format(getpass.getuser())
    lst_beat_model_path = [
        "/media/dev01/Project/ECG/Model/BeatNet/211224/256_60.0_480_1_1_0_2_0_0.7/output-3.552120",
        "/media/dev01/Project/ECG/Model/BeatNet/220123/256_60.0_480_0_0_0_2_0_0.99/output-3.780749",
        "/media/dev01/Project/ECG/Model/BeatNet/220119/256_60.0_480_0_0_0_2_0_0.9/output-3.581261",
        "/media/dev01/Project/ECG/Model/BeatNet/220115/256_60.0_480_0_0_0_2_0_0.99/output-3.778222",
        "/media/dev01/Project/ECG/Model/BeatNet/220120/256_60.0_480_0_0_0_2_0_0.99/output-3.699216",
        "/media/dev01/Project/ECG/Model/BeatNet/220117/256_60.0_480_0_0_0_2_0_0.8/output-3.622496",
        "/media/dev01/Project/ECG/Model/BeatNet/211224/256_60.0_480_1_1_0_2_0_0.7/output-3.552120",
        "/media/dev01/Project/ECG/Model/BeatNet/220119/256_60.0_480_0_0_0_2_0_0.9/output-3.581261",
        "/media/dev01/Project/ECG/Model/BeatNet/220115/256_60.0_480_0_0_0_2_0_0.99/output-3.778222",
        "/media/dev01/Project/ECG/Model/BeatNet/220120/256_60.0_480_0_0_0_2_0_0.99/output-3.699216",
        "/media/dev01/Project/ECG/Model/BeatNet/220117/256_60.0_480_0_0_0_2_0_0.8/output-3.622496",

    ]

    model_get_best = []
    beat_checkpoint_best = []
    squared_error_gross_best = []
    squared_error_average_best = []
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
        output_eval_directory = '{}/{}/event-{}-{}/{}/'.format(
            report_dir,
            os.path.basename(curr_dir),
            data_model.MODE,
            DAY_TIME,
            model_name)
        if not os.path.isdir(output_eval_directory):
            os.makedirs(output_eval_directory)

        with open(beat_datastore_file, 'r') as json_file:
            beat_datastore = json.load(json_file)

        with open(rhythm_datastore_file, 'r') as json_file:
            rhythm_datastore = json.load(json_file)

        lst_true_all, lst_pred_all, lst_file_all = comb_run_eval(
            use_gpu_index=0,
            beat_datastore=beat_datastore,
            beat_checkpoint=beat_checkpoint,
            rhythm_datastore=rhythm_datastore,
            rhythm_checkpoint=rhythm_checkpoint,
            eval_data_dir=DATA_PATH,
            output_eval_directory=output_eval_directory,
            num_of_process=8
        )

        # region check
        num_class = 3
        squared_error_gross = 0
        squared_error_average = 0

        path_out = '{}/QRS_report_line.out'.format(output_eval_directory)
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

        fstatus = open(output_eval_directory + '/squared_error.txt', 'w')
        fstatus.writelines('squared_error_gross: {:.6f}\n'.format(squared_error_gross))
        fstatus.writelines('squared_error_average: {:.6f}\n'.format(squared_error_average))
        fstatus.close()

        squared_error_gross_best.append(squared_error_gross)
        squared_error_average_best.append(squared_error_average)
        model_get_best.append(model_name)
        beat_checkpoint_best.append(beat_checkpoint)

        y_true_pred_label = np.concatenate((np.reshape(np.asarray(lst_true_all), (-1, 1)),
                                            np.reshape(np.asarray(lst_pred_all), (-1, 1))), axis=-1)
        np.savetxt(output_eval_directory + "/all_symbol.csv",
                   y_true_pred_label,
                   fmt="%s",
                   header="id, true, pred",
                   delimiter=",")
        label_true = np.unique(lst_true_all)
        label_pred = np.unique(lst_pred_all)
        cm = confusion_matrix(lst_true_all, lst_pred_all, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=np.union1d(label_true, label_pred))

        # NOTE: Fill all variables here with default values of the plot_confusion_matrix
        fig, axx = plt.subplots(nrows=1, ncols=1, figsize=(19.20, 10.80))
        disp = disp.plot(include_values=True, ax=axx)
        if output_eval_directory is not None:
            fig.savefig(output_eval_directory + "/confusion_matrix_normalize.svg", format='svg', dpi=1200)
            plt.close(fig)
        else:
            plt.show()

        cm = confusion_matrix(lst_true_all, lst_pred_all)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=np.union1d(label_true, label_pred))

        # NOTE: Fill all variables here with default values of the plot_confusion_matrix
        fig, axx = plt.subplots(nrows=1, ncols=1, figsize=(19.20, 10.80))
        disp = disp.plot(include_values=True, ax=axx)
        if output_eval_directory is not None:
            fig.savefig(output_eval_directory + "/confusion_matrix.svg", format='svg', dpi=1200)
            plt.close(fig)
        else:
            plt.show()

        # endregion

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dir_model_selected = '{}/{}/event-{}-{}'.format(report_dir,
                                                    os.path.basename(curr_dir),
                                                    data_model.MODE,
                                                    DAY_TIME)
    index_squared_error_gross = np.asarray([x for _, x in sorted(zip(squared_error_gross_best,
                                                                     np.arange(len(squared_error_gross_best))),
                                                                 reverse=False)])
    top_select = 10
    if len(index_squared_error_gross) < top_select:
        top_select = len(index_squared_error_gross)

    model_2_run_fieldnames = ['#',
                              'Case',
                              'SumSquaresError',
                              'AverageSumSquaresError',
                              'Checkpoint']
    count = 0
    model_2_run = open(dir_model_selected + '/model_2_run.csv', mode='w')
    model_2_run_writer = csv.DictWriter(model_2_run, fieldnames=model_2_run_fieldnames)
    model_2_run_writer.writeheader()

    for n, g in enumerate(index_squared_error_gross):
        row = dict()
        row["#"] = n
        row["Case"] = model_get_best[g]
        row["Checkpoint"] = beat_checkpoint_best[g]
        row["SumSquaresError"] = squared_error_gross_best[g]
        row["AverageSumSquaresError"] = squared_error_average_best[g]
        count += 1
        if n < top_select:
            model_2_run_writer.writerow(row)

    model_2_run.close()

    print()
    print('{} with squared_error_gross {}'.
          format(squared_error_gross_best[index_squared_error_gross[0]],
                 model_get_best[index_squared_error_gross[0]]))


if __name__ == '__main__':
    eval_run()
