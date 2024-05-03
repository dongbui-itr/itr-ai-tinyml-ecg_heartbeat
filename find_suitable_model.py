import getpass
import os
from os.path import basename, dirname
import re
import numpy as np
from datetime import datetime
import json
import itertools
import csv
import inputs as data_mode
import shutil

REF_CASE = ["ec57_ref", "ec57-2020-beat0-segment0"]  # ec57-20210623-holter4


def best_ec57():
    DISK = "Project"
    dir_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    report_dir = "/media/{}/{}/ECG/Report/{}".format(getpass.getuser(), DISK, dir_name)
    MEDIA_PATH = '/media/{}/{}/ECG/Model/{}'.format(getpass.getuser(), DISK, dir_name)
    top_select = 10
    lst = [
        ['mitdb', 'nstdb', 'escdb', 'ahadb'],
        ['mitdb', 'nstdb', 'escdb'],
        ['mitdb', 'nstdb'],
        ['mitdb', 'nstdb', 'ahadb'],
        ['mitdb'],
        ['nstdb'],
    ]
    lst_model_folder = [l for l in os.listdir(report_dir) if "ec57-" in l]
    ec57_report = report_dir + "/output"
    if not os.path.exists(ec57_report):
        os.makedirs(ec57_report)

    best_model = open(ec57_report + '/best_model.csv', mode='w')
    count_model = 0
    best_model_fieldnames = ['#',
                             'Case',
                             'SumSquaresError',
                             'AverageSumSquaresError',
                             'Checkpoint']

    best_model_writer = csv.DictWriter(best_model, fieldnames=best_model_fieldnames)
    best_model_writer.writeheader()
    is_ref = True
    for _model_folder in lst_model_folder:
        count_model += 1
        directory = report_dir + "/" + _model_folder
        best_row = dict()
        for ec, db in enumerate(lst):
            directory_ref = '/media/{}/Dataset/ECG/Device/{}/'.format(getpass.getuser(), REF_CASE[0])
            data_model = [x[0] for x in os.walk(directory)
                          if basename(x[0]) in db
                          if "output-" in x[0]]

            data_ref = [x[0] for x in os.walk(directory_ref) if REF_CASE[1] in x[0] if os.path.basename(x[0]) in db]
            path_model = set(map(lambda x: x[:-(len(os.path.basename(x)) + 1)], data_model))
            group_model = [[y for y in data_model if x in y] for x in path_model]
            group_model.sort()

            group_model.insert(0, data_ref)
            squared_error_gross = np.zeros(len(group_model))
            squared_error_average = np.zeros(len(group_model))

            dir_model_selected = directory + '/{}_{}'.format(REF_CASE[1], '_'.join(db))
            if not os.path.exists(dir_model_selected):
                os.makedirs(dir_model_selected)

            detail_report_file = open(dir_model_selected + '/ec57_detail_report.csv', mode='w')
            detail_fieldnames = ['#',
                                 'Case',
                                 'Metric',
                                 'Database',
                                 'N(se)',
                                 'N(p+)',
                                 'N_ref(se)',
                                 'N_ref(p+)',
                                 'V(se)',
                                 'V(p+)',
                                 'V_ref(se)',
                                 'V_ref(p+)',
                                 'S(se)',
                                 'S(p+)',
                                 'S_ref(se)',
                                 'S_ref(p+)']
            detail_report_writer = csv.DictWriter(detail_report_file, fieldnames=detail_fieldnames)
            detail_report_writer.writeheader()

            group_report_file = open(dir_model_selected + '/ec57_group_report.csv', mode='w')
            group_fieldnames = ['#',
                                'Case',
                                'Squared error gross',
                                'Squared error average',
                                'Squared error gross (ref)',
                                'Squared error average (ref)']
            group_report_writer = csv.DictWriter(group_report_file, fieldnames=group_fieldnames)
            group_report_writer.writeheader()

            index = 1
            ref_squared_error = dict()
            ref_detail = dict()
            for i, g in enumerate(group_model):
                row_item = dict()
                row_group_item = dict()
                row_detail_item = dict()
                row_tmp = g[0].split("/")
                if REF_CASE[1] in row_tmp[-5]:
                    isref = True
                    row_item["Case"] = REF_CASE[1]
                    row_detail_item["Metric"] = row_tmp[-2]
                else:
                    row_item["Case"] = row_tmp[-2]
                    row_detail_item["Metric"] = "best_squared_error_metric"
                    isref = False

                row_item["#"] = i
                row_group_item["#"] = i
                num_class = 3
                if isref:
                    row_detail_item["Case"] = row_tmp[-5]
                    row_group_item["Case"] = row_tmp[-5]
                else:
                    row_detail_item["Case"] = row_tmp[-2]
                    row_group_item["Case"] = row_tmp[-2]

                for f in g:
                    row_detail_item["#"] = index
                    row_detail_item["Database"] = os.path.basename(f)
                    path_out = '{}/{}_QRS_report_line.out'.format(f, os.path.basename(f))
                    file_out = open(path_out)
                    content = file_out.readlines()
                    # https://blog.finxter.com/regex-special-characters-examples-in-python-re/
                    gross_value = re.findall(r'[-+]?\d*\.\d+|\d+|[-]', content[-5])
                    average_value = re.findall(r'[-+]?\d*\.\d+|\d+|[-]', content[-4])

                    _keys = ["N(se)", "N(p+)", "V(se)", "V(p+)", "S(se)", "S(p+)"]

                    for n, k in enumerate(_keys):
                        if n < len(gross_value) and \
                                "-" not in gross_value[n] and \
                                (num_class < 0 or n < num_class * 2):
                            squared_error_gross[i] += np.square(1 - float(gross_value[n]) / 100)
                            squared_error_average[i] += np.square(1 - float(average_value[n]) / 100)
                            row_detail_item[k] = float(gross_value[n]) / 100
                        elif n < len(gross_value) and \
                                "-" in gross_value[n] and \
                                (num_class < 0 or n < num_class * 2):
                            squared_error_gross[i] += np.square(1 - float(0) / 100)
                            squared_error_average[i] += np.square(1 - float(0) / 100)
                            row_detail_item[k] = float(0) / 100
                        else:
                            row_detail_item[k] = "-"

                    if isref:
                        ref_detail[os.path.basename(f)] = dict()
                        ref_detail[os.path.basename(f)]["N_ref(se)"] = row_detail_item["N(se)"]
                        ref_detail[os.path.basename(f)]["N_ref(p+)"] = row_detail_item["N(p+)"]
                        ref_detail[os.path.basename(f)]["V_ref(se)"] = row_detail_item["V(se)"]
                        ref_detail[os.path.basename(f)]["V_ref(p+)"] = row_detail_item["V(p+)"]
                        ref_detail[os.path.basename(f)]["S_ref(se)"] = row_detail_item["S(se)"]
                        ref_detail[os.path.basename(f)]["S_ref(p+)"] = row_detail_item["S(p+)"]
                    else:
                        row_detail_item["N_ref(se)"] = ref_detail[os.path.basename(f)]["N_ref(se)"]
                        row_detail_item["N_ref(p+)"] = ref_detail[os.path.basename(f)]["N_ref(p+)"]
                        row_detail_item["V_ref(se)"] = ref_detail[os.path.basename(f)]["V_ref(se)"]
                        row_detail_item["V_ref(p+)"] = ref_detail[os.path.basename(f)]["V_ref(p+)"]
                        row_detail_item["S_ref(se)"] = ref_detail[os.path.basename(f)]["S_ref(se)"]
                        row_detail_item["S_ref(p+)"] = ref_detail[os.path.basename(f)]["S_ref(p+)"]

                        detail_report_writer.writerow(row_detail_item)
                        index += 1

                if isref:
                    ref_squared_error["gross"] = squared_error_gross[i]
                    ref_squared_error["average"] = squared_error_average[i]
                else:
                    row_item["Squared error gross"] = squared_error_gross[i]
                    row_item["Squared error average"] = squared_error_average[i]
                    row_item["Squared error gross (ref)"] = ref_squared_error["gross"]
                    row_item["Squared error average (ref)"] = ref_squared_error["average"]

                    row_group_item["Squared error gross"] = squared_error_gross[i]
                    row_group_item["Squared error average"] = squared_error_average[i]
                    row_group_item["Squared error gross (ref)"] = ref_squared_error["gross"]
                    row_group_item["Squared error average (ref)"] = ref_squared_error["average"]
                    group_report_writer.writerow(row_group_item)

            group_report_file.close()
            detail_report_file.close()
            index_squared_error_gross = np.asarray([x for _, x in sorted(zip(squared_error_gross,
                                                                             np.arange(len(squared_error_gross))),
                                                                         reverse=False)])

            if len(index_squared_error_gross) < top_select:
                top_select = len(index_squared_error_gross)

            all_model_2_run_file = open(dir_model_selected + '/all_model_2_run_file.csv', mode='w')

            model_2_run_fieldnames = ['#',
                                      'Case',
                                      'SumSquaresError',
                                      'AverageSumSquaresError',
                                      'Checkpoint',
                                      'ain_ext']

            all_model_2_run_writer = csv.DictWriter(all_model_2_run_file, fieldnames=model_2_run_fieldnames)
            all_model_2_run_writer.writeheader()
            count = 0
            all_row = dict()

            model_2_run = open(dir_model_selected + '/model_2_run.csv', mode='w')
            model_2_run_writer = csv.DictWriter(model_2_run, fieldnames=model_2_run_fieldnames)
            model_2_run_writer.writeheader()

            for n, g in enumerate(index_squared_error_gross):
                row = dict()

                row["#"] = n
                all_row["#"] = count

                case = group_model[g][0].split("/")[-2]
                row_tmp = re.split('\[|\]', case)
                # checkpoint =
                if "best_entropy_loss" in case:
                    row["Case"] = REF_CASE[1]
                    all_row["Case"] = REF_CASE[1]
                    row["Checkpoint"] = "None"
                    all_row["Checkpoint"] = "None"
                else:
                    row["Case"] = case
                    all_row["Case"] = case
                    row["Checkpoint"] = MEDIA_PATH + "/{}/{}/{}/model/{}/best_squared_error_metric".format(row_tmp[3],
                                                                                                           row_tmp[5],
                                                                                                           row_tmp[1],
                                                                                                           row_tmp[7])
                    all_row["Checkpoint"] = MEDIA_PATH + "/{}/{}/{}/model/{}/best_squared_error_metric".format(
                        row_tmp[3],
                        row_tmp[5],
                        row_tmp[1],
                        row_tmp[7])

                row["SumSquaresError"] = squared_error_gross[g]
                row["AverageSumSquaresError"] = squared_error_average[g]
                all_row["SumSquaresError"] = squared_error_gross[g]
                all_row["AverageSumSquaresError"] = squared_error_average[g]

                all_model_2_run_writer.writerow(all_row)
                count += 1
                if n < top_select:
                    if REF_CASE[1] in row_tmp:
                        continue

                    model_2_run_writer.writerow(row)

            model_2_run.close()
            all_model_2_run_file.close()

            if ec == 0:
                best_row["#"] = count_model
                case = group_model[index_squared_error_gross[0]][0].split("/")[-2]
                row_tmp = re.split('\[|\]', case)

                if "best_entropy_loss" in case and is_ref:
                    best_row["Case"] = REF_CASE[1]
                    best_row["Checkpoint"] = "None"
                    is_ref = False
                    best_row["SumSquaresError"] = squared_error_gross[index_squared_error_gross[0]]
                    best_row["AverageSumSquaresError"] = squared_error_average[index_squared_error_gross[0]]
                    best_model_writer.writerow(best_row)
                elif "best_entropy_loss" not in case:
                    best_row["Case"] = case
                    best_row["Checkpoint"] = MEDIA_PATH + "/{}/{}/{}/model/{}/best_squared_error_metric".format(
                        row_tmp[3],
                        row_tmp[5],
                        row_tmp[1],
                        row_tmp[7])

                    best_row["SumSquaresError"] = squared_error_gross[index_squared_error_gross[0]]
                    best_row["AverageSumSquaresError"] = squared_error_average[index_squared_error_gross[0]]
                    print(best_row["Checkpoint"])
                    best_model_writer.writerow(best_row)

    best_model.close()


def main():
    best_ec57()


if __name__ == "__main__":
    main()
