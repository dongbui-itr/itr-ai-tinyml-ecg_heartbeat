import copy
import os
import shutil

import numpy as np
import json
import wfdb as wf
import pandas as pd
from random import shuffle

from glob import glob
from collections import Counter
from all_config import CLASS_TYPES

DATAPATH = '/mnt/4T_DATA/DATA_4TINYML/strip2/'
# BEAT_TYPES = ['N', 'S', 'V', 'Q', 'R']

def split_data_2(data_dir=DATAPATH,
                 output_path='/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/240520/', ratio_train = 8 / 10, k=0):
    print("Mixing Study")
    while True:
        statictis_type = {}
        statictis_type["train"] = dict()
        statictis_type["train"]["studyID"] = []
        statictis_type["eval"] = dict()
        statictis_type["eval"]["studyID"] = []

        for type in CLASS_TYPES:
            statictis_type['total_{}'.format(type)] = 0
            statictis_type["train"]['total_{}'.format(type)] = 0
            statictis_type["eval"]['total_{}'.format(type)] = 0

        events = os.listdir(data_dir)
        print(events)
        for event in events:
            if '.json' in event or '.csv' in event:
                continue
            k = 0
            while(True):
                k += 1
                event_statictis_type = {}
                event_statictis_type["train"] = dict()
                event_statictis_type["train"]["studyID"] = []
                event_statictis_type["eval"] = dict()
                event_statictis_type["eval"]["studyID"] = []
                print(event)
                studies_path = os.listdir(f'{data_dir}/{event}/')
                event_statictis_type[event] = {}
                print('Number of {} studies: {}'.format(event, len(studies_path)))
                total_files = 0
                for type in CLASS_TYPES:
                    event_statictis_type[event]['total_{}'.format(type)] = 0
                    event_statictis_type["train"]['total_{}'.format(type)] = 0
                    event_statictis_type["eval"]['total_{}'.format(type)] = 0
                    # statictis_type["train"]["studyID"]['total_{}'.format(type)] = 0
                    # statictis_type["eval"]["studyID"]['total_{}'.format(type)] = 0

                idx = np.arange(len(studies_path), dtype=int)
                np.random.shuffle(idx)
                np.random.shuffle(idx)
                cnt_study = 0
                for study in studies_path:
                    study_id = study.split('/')[-1]
                    event_statictis_type[event][study_id] = {}
                    if (not study_id in statictis_type["eval"]["studyID"] and cnt_study/len(studies_path) <= ratio_train) or study_id in statictis_type["train"]["studyID"]:
                        flag_train = True
                        cnt_study += 1
                    else:
                        flag_train = False

                    for type in CLASS_TYPES:
                        event_statictis_type[event][study_id]['total_{}'.format(type)] = 0
                        event_statictis_type[event][study_id]['path'] = study_id

                    files = [i[:-4] for i in glob(f'{data_dir}/{event}/{study}/*.atr')]
                    total_files += len(files)
                    for file in files:
                        ann = wf.rdann(file, 'atr')
                        symbol = ann.symbol
                        symbol_statictis = Counter(symbol)
                        if flag_train:
                            if not study_id in event_statictis_type["train"]["studyID"]:
                                event_statictis_type["train"]["studyID"].append(study_id)
                        else:
                            if not study_id in event_statictis_type["eval"]["studyID"]:
                                event_statictis_type["eval"]["studyID"].append(study_id)

                        for key in symbol_statictis.keys():

                            # if key == '|':
                            #     _key = 'Q'
                            # else:
                            #     _key = copy.deepcopy(key)
                            if key not in CLASS_TYPES:
                                # print(f'key={key}')
                                continue

                            try:
                                event_statictis_type[event][study_id]['total_{}'.format(key)] += symbol_statictis[key]
                                event_statictis_type[event]['total_{}'.format(key)] += symbol_statictis[key]
                                if flag_train:
                                    event_statictis_type["train"]['total_{}'.format(key)] += symbol_statictis[key]
                                else:
                                    event_statictis_type["eval"]['total_{}'.format(key)] += symbol_statictis[key]
                            except:
                                a=10
                print(f'Total of files: {total_files}')
                print(f'k={k}')
                print(f"N: {event_statictis_type['train']['total_N']} vs {event_statictis_type['eval']['total_N']}")
                print(f"S: {event_statictis_type['train']['total_S']} vs {event_statictis_type['eval']['total_S']}")
                print(f"V: {event_statictis_type['train']['total_V']} vs {event_statictis_type['eval']['total_V']}")
                if ((event_statictis_type['train']['total_S'] > event_statictis_type['eval']['total_S'] and
                        event_statictis_type['train']['total_V'] > event_statictis_type['eval']['total_V']) or
                        (event_statictis_type[event]['total_S'] < 20 and event_statictis_type[event]['total_V'] < 20) or k > 10):
                    break

            for type in CLASS_TYPES:
                statictis_type['total_{}'.format(type)] += event_statictis_type['train']['total_{}'.format(type)]
                statictis_type['total_{}'.format(type)] += event_statictis_type['eval']['total_{}'.format(type)]
                statictis_type["train"]['total_{}'.format(type)] += event_statictis_type['train']['total_{}'.format(type)]
                statictis_type["eval"]['total_{}'.format(type)] += event_statictis_type['eval']['total_{}'.format(type)]

            statictis_type[event] = event_statictis_type[event]
            statictis_type[event] = event_statictis_type[event]
            statictis_type["train"]["studyID"].extend(event_statictis_type['train']['studyID'])
            statictis_type["eval"]["studyID"].extend(event_statictis_type['eval']['studyID'])

        flag_stop = True
        for type in CLASS_TYPES:
            if not type in ['S', 'V']:
                continue
            else:
                if statictis_type["train"]['total_{}'.format(type)]/statictis_type["eval"]['total_{}'.format(type)] <= 2:
                    flag_stop = False
        if flag_stop:
            break

        # break
    print(f'{output_path}/log_random_data.json')
    fp = open(f'{output_path}/log_random_data.json', 'w')
    fp.write(json.dumps(statictis_type, indent=4))
    fp.close()


def add_study(
        excel='/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/Collection/[STUDY-ID] Information.xlsx',
        data_path='/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/Collection/',
        export_path='/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/Collection_20231018/'):
    # types = ['N', 'S', 'V', 'R', 'TACHY', 'BRADY']
    types = ['N', 'S', 'V', 'IVCD', 'IVCD-SVE', 'IVCD-VE', 'TACHY', 'BRADY']
    # types = ['TACHY', 'BRADY']
    # ful = {'N': 'Normal', 'S': 'Supraventricular', 'V': 'Ventricular', 'R': 'IVCD'}

    xls = pd.ExcelFile(excel)
    log = open(data_path + 'Log_studies.txt', 'w')
    for type in types:
        df = pd.read_excel(xls, type)
        df_events = df['eventId']
        df_studies = df['studyId']
        event_ids = glob(data_path + 'export_{}/*'.format(type))
        if not os.path.exists(export_path + 'export_{}'.format(type)):
            os.makedirs(export_path + 'export_{}'.format(type))

        cnt_type = 0
        for event in event_ids:
            try:
                indx = list(df_events).index(os.path.basename(event))
                study = np.asarray(df_studies)[indx]

                if not os.path.exists(export_path + 'export_{}/{}'.format(type, study)):
                    os.makedirs(export_path + 'export_{}/{}'.format(type, study))

                from distutils.dir_util import copy_tree
                copy_tree(event, export_path + 'export_{}/{}/{}'.format(type, study, os.path.basename(event)))
            except Exception as err:

                cnt_type += 1
                log.writelines('{}: {}\n'.format(type, event))

        print('{}: {}'.format(type, cnt_type))

    log.close()


def statistic_study(
        excel='/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/Collection/[STUDY-ID] Information.xlsx'):
    types = ['N', 'S', 'V', 'R', 'RSE', 'RVE']
    # types = ['TACHY', 'BRADY']
    xls = pd.ExcelFile(excel)
    statistic = dict()
    statistic['Total'] = dict()
    statistic['study'] = []
    for type in types:
        statistic[type] = []

    cnt_study = 0
    for type in types:
        statistic['Total'][type] = 0
        df = pd.read_excel(xls, type)
        df_studies = df['studyId']
        for id in df_studies:
            if id in list(statistic.keys()):
                statistic[list(statistic.keys()).index(id)]
                statistic[id][type] += 1
            else:
                cnt_study += 1
                statistic[id] = dict()
                for itype in types:
                    statistic[id][itype] = 0

                statistic[id][type] += 1
                statistic['Total'][type] += 1

    print('Total Study:', cnt_study)
    # df_statistic = pd.DataFrame(statistic).to_excel(os.path.dirname(excel) + 'statistic_study.xlsx')


if __name__ == '__main__':
    split_data_2(data_dir='/mnt/4T_DATA/DATA_4TINYML/tiny_hb3_data_strip_2/',
                 output_path='/mnt/4T_DATA/DATA_4TINYML/tiny_hb3_data_strip_2/')
    # add_study()
    # statistic_study()
