import os
import shutil

import numpy as np
import json
import wfdb as wf
import pandas as pd
from random import shuffle

from glob import glob
from collections import Counter


DATAPATH = '/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/Collection_20231002/'


def split_data_2(data_dir='/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/Collection_20240510/',
                 output_path='/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/240510/', k=0):
    print("Mixing Study")
    if os.path.exists(output_path + 'log_data_noise.json'):
        shutil.move(output_path + 'log_data_noise.json', output_path + f'log_data_noise{k}.json')

    # types = ['S', 'V', 'R', 'N']
    types = ['N', 'S', 'V', 'IVCD', 'IVCD-SE', 'IVCD-VE', 'TACHY', 'BRADY', 'noise']
    # types = ['BRADY', 'noise']
    # types = ['TACHY', 'BRADY']

    statictis_type = {}
    for _type in types:
        if 'IVCD' in _type:
            type = 'R'
        elif 'noise' in _type:
            type = 'Q'
        else:
            type = _type

        if not type in list(statictis_type.keys()):
            statictis_type[type] = {}

        data_path = os.path.join(data_dir, 'export_{}'.format(_type.replace('IVCD', 'R').replace('-', '')))
        if _type == 'noise':
            studies_path = glob(data_path + '/*/*')
        else:
            studies_path = glob(data_path + '/*')

        print('Number of {} files: {}'.format(type, len(studies_path)))
        a=10
        for i in types:
            if 'IVCD' in i:
                i = 'R'
                # continue
            elif 'noise' in i:
                i = 'Q'

            if '-' in i:
                continue

            statictis_type[type]['total_{}'.format(i)] = 0

        for study in studies_path:
            study_id = study.split('/')[-1]
            statictis_type[type][study_id] = {}
            for i in types:
                if 'IVCD' in i:
                    i = 'R'
                elif 'noise' in i:
                    i = 'Q'

                if '-' in i:
                    continue

                # statictis_type[type]['total_{}'.format(i)] = 0
                statictis_type[type][study_id]['total_{}'.format(i)] = 0

            # events_id = glob(study + '/*')/
            events_id = [study]
            for event in events_id:
                files = [i[:-4] for i in glob(event + '/*.atr')]
                for file in files:
                    ann = wf.rdann(file, 'atr')
                    symbol = ann.symbol
                    if 'noise' in file:
                        symbol = ['Q'] * len(symbol)

                    symbol_statictis = Counter(symbol)
                    for key in symbol_statictis.keys():
                        if 'noise' in file:
                            key = 'Q'
                        # if 'Q' in key:
                        #     statictis_type[type][study_id]['total_{}'.format('N')] += symbol_statictis[key]
                        #     statictis_type[type]['total_{}'.format('N')] += symbol_statictis[key]
                        # else:
                        statictis_type[type][study_id]['total_{}'.format(key)] += symbol_statictis[key]
                        statictis_type[type]['total_{}'.format(key)] += symbol_statictis[key]

    total_S_study = len(statictis_type['S'])
    total_V_study = len(statictis_type['V'])
    total_R_study = len(statictis_type['R'])
    total_N_study = len(statictis_type['N'])
    total_Q_study = len(statictis_type['Q'])
    total_TACHY_study = len(statictis_type['TACHY'])
    total_BRADY_study = len(statictis_type['BRADY'])

    ratio_train = 8 / 10
    limit_train_S = int((total_S_study - len(types)) * ratio_train)
    limit_train_V = int((total_V_study - len(types)) * ratio_train)
    limit_train_R = int((total_R_study - len(types)) * ratio_train)
    limit_train_N = int((total_N_study - len(types)) * ratio_train)
    limit_train_NOISE = int((total_Q_study - len(types)) * ratio_train)
    limit_train_TACHY = int((total_TACHY_study - len(types)) * ratio_train)
    limit_train_BRADY = int((total_BRADY_study - len(types)) * ratio_train)

    ratio_eval = 2 / 10
    limit_eval_S = int((total_S_study - len(types)) * ratio_eval)
    limit_eval_V = int((total_V_study - len(types)) * ratio_eval)
    limit_eval_R = int((total_R_study - len(types)) * ratio_eval)
    limit_eval_N = int((total_N_study - len(types)) * ratio_eval)
    limit_eval_Q = int((total_Q_study - len(types)) * ratio_eval)

    final_total_train_S = 0
    final_total_train_V = 0
    final_total_train_R = 0
    final_total_train_N = 0
    final_total_train_Q = 0

    final_total_eval_S = 0
    final_total_eval_V = 0
    final_total_eval_R = 0
    final_total_eval_N = 0
    final_total_eval_Q = 0

    statictis_type['Train'] = {}
    statictis_type['Eval'] = {}

    for type in types:
        if 'IVCD' in type:
            type = 'R'
        elif 'noise' in type:
            type = 'Q'

        if not type in list(statictis_type['Train'].keys()):
            statictis_type['Train'][type] = {}
            statictis_type['Eval'][type] = {}

    ### S Region ###
    times = 0
    list_S_studies = [i for i in statictis_type['S'].keys() if not 'total_' in i]
    while True:
        times += 1
        print('S_train: ', times)
        shuffle(list_S_studies)
        shuffle(list_S_studies)
        S_in_S_study = 0
        V_in_S_study = 0
        R_in_S_study = 0
        N_in_S_study = 0
        Q_in_S_study = 0

        for i in range(limit_train_S):
            S_in_S_study += statictis_type['S'][list_S_studies[i]]['total_S']
            V_in_S_study += statictis_type['S'][list_S_studies[i]]['total_V']
            R_in_S_study += statictis_type['S'][list_S_studies[i]]['total_R']
            N_in_S_study += statictis_type['S'][list_S_studies[i]]['total_N']
            Q_in_S_study += statictis_type['S'][list_S_studies[i]]['total_Q']

        if S_in_S_study >= statictis_type['S']['total_S'] * ratio_train:
            break

    final_total_train_S += S_in_S_study
    final_total_train_V += V_in_S_study
    final_total_train_R += R_in_S_study
    final_total_train_N += N_in_S_study
    final_total_train_Q += Q_in_S_study
    statictis_type['Train']['S']['Total_S_in_S_study'] = S_in_S_study
    statictis_type['Train']['S']['Total_V_in_S_study'] = V_in_S_study
    statictis_type['Train']['S']['Total_R_in_S_study'] = R_in_S_study
    statictis_type['Train']['S']['Total_N_in_S_study'] = N_in_S_study
    statictis_type['Train']['S']['Total_Q_in_S_study'] = N_in_S_study
    statictis_type['Train']['S_study'] = list_S_studies[:limit_train_S]

    # times = 0
    # list_S_studies = list_S_studies[limit_train_S:]
    #
    # print('S_eval: ', times)
    # shuffle(list_S_studies)
    # shuffle(list_S_studies)
    S_in_S_study = 0
    V_in_S_study = 0
    R_in_S_study = 0
    N_in_S_study = 0
    Q_in_S_study = 0

    for i in range(limit_eval_S):
        i += limit_train_S
        S_in_S_study += statictis_type['S'][list_S_studies[i]]['total_S']
        V_in_S_study += statictis_type['S'][list_S_studies[i]]['total_V']
        R_in_S_study += statictis_type['S'][list_S_studies[i]]['total_R']
        N_in_S_study += statictis_type['S'][list_S_studies[i]]['total_N']
        Q_in_S_study += statictis_type['S'][list_S_studies[i]]['total_Q']

    final_total_eval_S += S_in_S_study
    final_total_eval_V += V_in_S_study
    final_total_eval_R += R_in_S_study
    final_total_eval_N += N_in_S_study
    final_total_eval_N += Q_in_S_study
    statictis_type['Eval']['S']['Total_S_in_S_study'] = S_in_S_study
    statictis_type['Eval']['S']['Total_V_in_S_study'] = V_in_S_study
    statictis_type['Eval']['S']['Total_R_in_S_study'] = R_in_S_study
    statictis_type['Eval']['S']['Total_N_in_S_study'] = N_in_S_study
    statictis_type['Eval']['S']['Total_Q_in_S_study'] = N_in_S_study
    statictis_type['Eval']['S_study'] = list_S_studies[:limit_eval_S]

    ### V Region ###
    times = 0
    list_V_studies = [i for i in statictis_type['V'].keys() if not 'total_' in i]
    while True:
        times += 1
        print('V_train: ', times)
        shuffle(list_V_studies)
        shuffle(list_V_studies)
        S_in_V_study = 0
        V_in_V_study = 0
        R_in_V_study = 0
        N_in_V_study = 0
        Q_in_V_study = 0

        ## Check V study in Eval S study
        cnt = 0
        indx = 0
        list_V_studies_train = []
        list_V_studies_eval = []
        while cnt < limit_train_V and indx < len(list_V_studies):
            if list_V_studies[indx] in statictis_type['Eval']['S_study']:
                list_V_studies_eval.append(list_V_studies[indx])
            else:
                list_V_studies_train.append(list_V_studies[indx])
                cnt += 1

            indx += 1

        while indx < len(list_V_studies):
            list_V_studies_eval.append(list_V_studies[indx])
            indx += 1

        for i in list_V_studies_train:
            S_in_V_study += statictis_type['V'][i]['total_S']
            V_in_V_study += statictis_type['V'][i]['total_V']
            R_in_V_study += statictis_type['V'][i]['total_R']
            N_in_V_study += statictis_type['V'][i]['total_N']
            Q_in_V_study += statictis_type['V'][i]['total_Q']

        if V_in_V_study >= statictis_type['V']['total_V'] * ratio_train:
            break

    final_total_train_S += S_in_V_study
    final_total_train_V += V_in_V_study
    final_total_train_R += R_in_V_study
    final_total_train_N += N_in_V_study
    final_total_train_Q += Q_in_V_study
    statictis_type['Train']['V']['Total_S_in_V_study'] = S_in_V_study
    statictis_type['Train']['V']['Total_V_in_V_study'] = V_in_V_study
    statictis_type['Train']['V']['Total_R_in_V_study'] = R_in_V_study
    statictis_type['Train']['V']['Total_N_in_V_study'] = N_in_V_study
    statictis_type['Train']['V']['Total_Q_in_V_study'] = N_in_V_study
    statictis_type['Train']['V_study'] = list_V_studies_train

    # times = 0
    # list_V_studies = list_V_studies[limit_train_V:]
    #
    # print('V_eval: ', times)
    # shuffle(list_V_studies)
    # shuffle(list_V_studies)
    S_in_V_study = 0
    V_in_V_study = 0
    R_in_V_study = 0
    N_in_V_study = 0
    Q_in_V_study = 0

    for i in list_V_studies_eval:
        S_in_V_study += statictis_type['V'][i]['total_S']
        V_in_V_study += statictis_type['V'][i]['total_V']
        R_in_V_study += statictis_type['V'][i]['total_R']
        N_in_V_study += statictis_type['V'][i]['total_N']
        Q_in_V_study += statictis_type['V'][i]['total_Q']

    final_total_eval_S += S_in_V_study
    final_total_eval_V += V_in_V_study
    final_total_eval_R += R_in_V_study
    final_total_eval_N += N_in_V_study
    final_total_eval_Q += Q_in_V_study
    statictis_type['Eval']['V']['Total_S_in_V_study'] = S_in_V_study
    statictis_type['Eval']['V']['Total_V_in_V_study'] = V_in_V_study
    statictis_type['Eval']['V']['Total_R_in_V_study'] = R_in_V_study
    statictis_type['Eval']['V']['Total_N_in_V_study'] = N_in_V_study
    statictis_type['Eval']['V']['Total_Q_in_V_study'] = N_in_V_study
    statictis_type['Eval']['V_study'] = list_V_studies_eval

    ### R Region ###
    times = 0
    list_R_studies = [i for i in statictis_type['R'].keys() if not 'total_' in i]
    while True:
        times += 1
        print('R_train: ', times)
        shuffle(list_R_studies)
        shuffle(list_R_studies)
        S_in_R_study = 0
        V_in_R_study = 0
        R_in_R_study = 0
        N_in_R_study = 0
        Q_in_R_study = 0

        ## Check R study in Eval S or V study
        cnt = 0
        indx = 0
        list_R_studies_train = [ ]
        list_R_studies_eval = [ ]
        while cnt < limit_train_R and indx < len(list_R_studies):
            if list_R_studies[ indx ] in statictis_type[ 'Eval' ][ 'S_study' ] \
                    or list_R_studies[ indx ] in statictis_type[ 'Eval' ][ 'V_study' ]:
                list_R_studies_eval.append(list_R_studies[ indx ])
            else:
                list_R_studies_train.append(list_R_studies[ indx ])
                cnt += 1

            indx += 1

        while indx < len(list_R_studies):
            list_R_studies_eval.append(list_R_studies[ indx ])
            indx += 1

        for i in list_R_studies_train:
            S_in_R_study += statictis_type['R'][i]['total_S']
            V_in_R_study += statictis_type['R'][i]['total_V']
            R_in_R_study += statictis_type['R'][i]['total_R']
            N_in_R_study += statictis_type['R'][i]['total_N']
            Q_in_R_study += statictis_type['R'][i]['total_Q']

        if R_in_R_study >= statictis_type['R']['total_R'] * ratio_train:
            break

    final_total_train_S += S_in_R_study
    final_total_train_V += V_in_R_study
    final_total_train_R += R_in_R_study
    final_total_train_N += N_in_R_study
    final_total_train_Q += Q_in_R_study
    statictis_type['Train']['R']['Total_S_in_R_study'] = S_in_R_study
    statictis_type['Train']['R']['Total_V_in_R_study'] = V_in_R_study
    statictis_type['Train']['R']['Total_R_in_R_study'] = R_in_R_study
    statictis_type['Train']['R']['Total_N_in_R_study'] = N_in_R_study
    statictis_type['Train']['R']['Total_Q_in_R_study'] = N_in_R_study
    statictis_type['Train']['R_study'] = list_R_studies_train

    # times = 0
    # list_R_studies = list_R_studies[limit_train_R:]
    #
    # print('R_eval: ', times)
    # shuffle(list_R_studies)
    # shuffle(list_R_studies)
    S_in_R_study = 0
    V_in_R_study = 0
    R_in_R_study = 0
    N_in_R_study = 0
    Q_in_R_study = 0

    for i in list_R_studies_eval:
        S_in_R_study += statictis_type['R'][i]['total_S']
        V_in_R_study += statictis_type['R'][i]['total_V']
        R_in_R_study += statictis_type['R'][i]['total_R']
        N_in_R_study += statictis_type['R'][i]['total_N']
        N_in_R_study += statictis_type['R'][i]['total_Q']

    final_total_eval_S += S_in_R_study
    final_total_eval_V += V_in_R_study
    final_total_eval_R += R_in_R_study
    final_total_eval_N += N_in_R_study
    final_total_eval_Q += Q_in_R_study
    statictis_type['Eval']['R']['Total_S_in_R_study'] = S_in_R_study
    statictis_type['Eval']['R']['Total_V_in_R_study'] = V_in_R_study
    statictis_type['Eval']['R']['Total_R_in_R_study'] = R_in_R_study
    statictis_type['Eval']['R']['Total_N_in_R_study'] = N_in_R_study
    statictis_type['Eval']['R']['Total_Q_in_R_study'] = Q_in_R_study
    statictis_type['Eval']['R_study'] = list_R_studies_eval

    ### N Region ###
    times = 0
    list_N_studies = [i for i in statictis_type['N'].keys() if not 'total_' in i]
    # while True:
    #     times += 1
    print('N_train: ', times)
    shuffle(list_N_studies)
    shuffle(list_N_studies)
    S_in_N_study = 0
    V_in_N_study = 0
    R_in_N_study = 0
    N_in_N_study = 0
    Q_in_N_study = 0

    ## Check N study in Eval S or V or R study
    cnt = 0
    indx = 0
    list_N_studies_train = [ ]
    list_N_studies_eval = [ ]
    while cnt < limit_train_N and indx < len(list_N_studies):
        if list_N_studies[ indx ] in statictis_type[ 'Eval' ][ 'S_study' ] \
                or list_N_studies[ indx ] in statictis_type[ 'Eval' ][ 'V_study' ] \
                or list_N_studies[ indx ] in statictis_type[ 'Eval' ][ 'R_study' ]:
            list_N_studies_eval.append(list_N_studies[ indx ])
        else:
            list_N_studies_train.append(list_N_studies[ indx ])
            cnt += 1

        indx += 1

    while indx < len(list_N_studies):
        list_N_studies_eval.append(list_N_studies[ indx ])
        indx += 1

    for i in list_N_studies_train:
        S_in_N_study += statictis_type['N'][i]['total_S']
        V_in_N_study += statictis_type['N'][i]['total_V']
        R_in_N_study += statictis_type['N'][i]['total_R']
        N_in_N_study += statictis_type['N'][i]['total_N']
        Q_in_N_study += statictis_type['N'][i]['total_Q']

    # if N_in_N_study >= statictis_type['N']['total_N'] * ratio_train:
    #     break

    final_total_train_S += S_in_N_study
    final_total_train_V += V_in_N_study
    final_total_train_R += R_in_N_study
    final_total_train_N += N_in_N_study
    final_total_train_Q += Q_in_N_study
    statictis_type['Train']['N']['Total_S_in_N_study'] = S_in_N_study
    statictis_type['Train']['N']['Total_V_in_N_study'] = V_in_N_study
    statictis_type['Train']['N']['Total_R_in_N_study'] = R_in_N_study
    statictis_type['Train']['N']['Total_N_in_N_study'] = N_in_N_study
    statictis_type['Train']['N']['Total_Q_in_N_study'] = Q_in_N_study
    statictis_type['Train']['N_study'] = list_N_studies_train

    # times = 0
    # list_N_studies_eval = list_N_studies[limit_train_N:]
    # times += 1
    # print('N_eval: ', times)
    # shuffle(list_N_studies)
    # shuffle(list_N_studies)
    S_in_N_study = 0
    V_in_N_study = 0
    R_in_N_study = 0
    N_in_N_study = 0
    Q_in_N_study = 0

    for i in list_N_studies_eval:
        S_in_N_study += statictis_type['N'][i]['total_S']
        V_in_N_study += statictis_type['N'][i]['total_V']
        R_in_N_study += statictis_type['N'][i]['total_R']
        N_in_N_study += statictis_type['N'][i]['total_N']
        Q_in_N_study += statictis_type['N'][i]['total_Q']

    final_total_eval_S += S_in_N_study
    final_total_eval_V += V_in_N_study
    final_total_eval_R += R_in_N_study
    final_total_eval_N += N_in_N_study
    final_total_eval_Q += Q_in_N_study
    statictis_type['Eval']['N']['Total_S_in_N_study'] = S_in_N_study
    statictis_type['Eval']['N']['Total_V_in_N_study'] = V_in_N_study
    statictis_type['Eval']['N']['Total_R_in_N_study'] = R_in_N_study
    statictis_type['Eval']['N']['Total_N_in_N_study'] = N_in_N_study
    statictis_type['Eval']['N']['Total_Q_in_N_study'] = Q_in_N_study
    statictis_type['Eval']['N_study'] = list_N_studies_eval

    ### BRADY Region ###
    times = 0
    list_BRADY_studies = [i for i in statictis_type['BRADY'].keys() if not 'total_' in i]
    # while True:
    #     times += 1
    print('BRADY_train: ', times)
    shuffle(list_BRADY_studies)
    shuffle(list_BRADY_studies)
    S_in_N_study = 0
    V_in_N_study = 0
    R_in_N_study = 0
    N_in_N_study = 0
    Q_in_N_study = 0

    ## Check N study in Eval S or V or R study
    cnt = 0
    indx = 0
    list_BRADY_studies_train = []
    list_BRADY_studies_eval = []
    while cnt < limit_train_BRADY and indx < len(list_BRADY_studies):
        if list_BRADY_studies[indx] in statictis_type['Eval']['S_study'] \
                or list_BRADY_studies[indx] in statictis_type['Eval']['V_study'] \
                or list_BRADY_studies[indx] in statictis_type['Eval']['R_study']:
            list_BRADY_studies_eval.append(list_BRADY_studies[indx])
        else:
            list_BRADY_studies_train.append(list_BRADY_studies[indx])
            cnt += 1

        indx += 1

    while indx < len(list_BRADY_studies):
        list_BRADY_studies_eval.append(list_BRADY_studies[indx])
        indx += 1

    for i in list_BRADY_studies_train:
        S_in_N_study += statictis_type['BRADY'][i]['total_S']
        V_in_N_study += statictis_type['BRADY'][i]['total_V']
        R_in_N_study += statictis_type['BRADY'][i]['total_R']
        N_in_N_study += statictis_type['BRADY'][i]['total_N']
        Q_in_N_study += statictis_type['BRADY'][i]['total_Q']

    # if N_in_N_study >= statictis_type['N']['total_N'] * ratio_train:
    #     break

    final_total_train_S += S_in_N_study
    final_total_train_V += V_in_N_study
    final_total_train_R += R_in_N_study
    final_total_train_N += N_in_N_study
    statictis_type['Train']['BRADY']['Total_S_in_N_study'] = S_in_N_study
    statictis_type['Train']['BRADY']['Total_V_in_N_study'] = V_in_N_study
    statictis_type['Train']['BRADY']['Total_R_in_N_study'] = R_in_N_study
    statictis_type['Train']['BRADY']['Total_N_in_N_study'] = N_in_N_study
    statictis_type['Train']['BRADY']['Total_Q_in_N_study'] = Q_in_N_study
    statictis_type['Train']['BRADY_study'] = list_BRADY_studies_train

    # times = 0
    # list_BRADY_studies = list_BRADY_studies[limit_train_BRADY:]
    # times += 1
    # print('BRADY_train: ', times)
    # shuffle(list_BRADY_studies)
    # shuffle(list_BRADY_studies)
    S_in_BRADY_study = 0
    V_in_BRADY_study = 0
    R_in_BRADY_study = 0
    N_in_BRADY_study = 0
    Q_in_BRADY_study = 0

    for i in list_BRADY_studies_eval:
        S_in_BRADY_study += statictis_type['BRADY'][i]['total_S']
        V_in_BRADY_study += statictis_type['BRADY'][i]['total_V']
        R_in_BRADY_study += statictis_type['BRADY'][i]['total_R']
        N_in_BRADY_study += statictis_type['BRADY'][i]['total_N']
        Q_in_BRADY_study += statictis_type['BRADY'][i]['total_Q']

    final_total_eval_S += S_in_BRADY_study
    final_total_eval_V += V_in_BRADY_study
    final_total_eval_R += R_in_BRADY_study
    final_total_eval_N += N_in_BRADY_study
    statictis_type['Eval']['BRADY']['Total_S_in_BRADY_study'] = S_in_BRADY_study
    statictis_type['Eval']['BRADY']['Total_V_in_BRADY_study'] = V_in_BRADY_study
    statictis_type['Eval']['BRADY']['Total_R_in_BRADY_study'] = R_in_BRADY_study
    statictis_type['Eval']['BRADY']['Total_N_in_BRADY_study'] = N_in_BRADY_study
    statictis_type['Eval']['BRADY']['Total_Q_in_BRADY_study'] = Q_in_BRADY_study
    statictis_type['Eval']['BRADY_study'] = list_BRADY_studies_eval

    ### TACHY Region ###
    times = 0
    list_TACHY_studies = [i for i in statictis_type['TACHY'].keys() if not 'total_' in i]
    # while True:
    #     times += 1
    print('TACHY_train: ', times)
    shuffle(list_TACHY_studies)
    shuffle(list_TACHY_studies)
    S_in_N_study = 0
    V_in_N_study = 0
    R_in_N_study = 0
    N_in_N_study = 0
    Q_in_N_study = 0

    ## Check N study in Eval S or V or R study
    cnt = 0
    indx = 0
    list_TACHY_studies_train = []
    list_TACHY_studies_eval = []
    while cnt < limit_train_TACHY and indx < len(list_TACHY_studies):
        if list_TACHY_studies[indx] in statictis_type['Eval']['S_study'] \
                or list_TACHY_studies[indx] in statictis_type['Eval']['V_study'] \
                or list_TACHY_studies[indx] in statictis_type['Eval']['R_study']:
            list_TACHY_studies_eval.append(list_TACHY_studies[indx])
        else:
            list_TACHY_studies_train.append(list_TACHY_studies[indx])
            cnt += 1

        indx += 1

    while indx < len(list_TACHY_studies):
        list_TACHY_studies_eval.append(list_TACHY_studies[indx])
        indx += 1

    for i in list_TACHY_studies_train:
        S_in_N_study += statictis_type['TACHY'][i]['total_S']
        V_in_N_study += statictis_type['TACHY'][i]['total_V']
        R_in_N_study += statictis_type['TACHY'][i]['total_R']
        N_in_N_study += statictis_type['TACHY'][i]['total_N']
        Q_in_N_study += statictis_type['TACHY'][i]['total_Q']

    # if N_in_N_study >= statictis_type['N']['total_N'] * ratio_train:
    #     break

    final_total_train_S += S_in_N_study
    final_total_train_V += V_in_N_study
    final_total_train_R += R_in_N_study
    final_total_train_N += N_in_N_study
    final_total_train_Q += Q_in_N_study
    statictis_type['Train']['TACHY']['Total_S_in_N_study'] = S_in_N_study
    statictis_type['Train']['TACHY']['Total_V_in_N_study'] = V_in_N_study
    statictis_type['Train']['TACHY']['Total_R_in_N_study'] = R_in_N_study
    statictis_type['Train']['TACHY']['Total_N_in_N_study'] = N_in_N_study
    statictis_type['Train']['TACHY']['Total_Q_in_N_study'] = Q_in_N_study
    statictis_type['Train']['TACHY_study'] = list_TACHY_studies_train

    # times = 0
    # list_TACHY_studies = list_TACHY_studies[limit_train_TACHY:]
    # times += 1
    # print('TACHY_eval: ', times)
    # shuffle(list_N_studies)
    # shuffle(list_N_studies)
    S_in_TACHY_study = 0
    V_in_TACHY_study = 0
    R_in_TACHY_study = 0
    N_in_TACHY_study = 0
    Q_in_TACHY_study = 0

    for i in list_TACHY_studies_eval:
        S_in_TACHY_study += statictis_type['TACHY'][i]['total_S']
        V_in_TACHY_study += statictis_type['TACHY'][i]['total_V']
        R_in_TACHY_study += statictis_type['TACHY'][i]['total_R']
        N_in_TACHY_study += statictis_type['TACHY'][i]['total_N']
        Q_in_TACHY_study += statictis_type['TACHY'][i]['total_Q']

    final_total_eval_S += S_in_TACHY_study
    final_total_eval_V += V_in_TACHY_study
    final_total_eval_R += R_in_TACHY_study
    final_total_eval_N += N_in_TACHY_study
    final_total_eval_Q += Q_in_TACHY_study
    statictis_type['Eval']['TACHY']['Total_S_in_TACHY_study'] = S_in_TACHY_study
    statictis_type['Eval']['TACHY']['Total_V_in_TACHY_study'] = V_in_TACHY_study
    statictis_type['Eval']['TACHY']['Total_R_in_TACHY_study'] = R_in_TACHY_study
    statictis_type['Eval']['TACHY']['Total_N_in_TACHY_study'] = N_in_TACHY_study
    statictis_type['Eval']['TACHY']['Total_Q_in_TACHY_study'] = Q_in_TACHY_study
    statictis_type['Eval']['TACHY_study'] = list_TACHY_studies_eval

    ### NOISE Region ###
    times = 0
    list_NOISE_studies = [i for i in statictis_type['Q'].keys() if not 'total_' in i]
    # while True:
    #     times += 1
    print('NOISE_train: ', times)
    shuffle(list_NOISE_studies)
    shuffle(list_NOISE_studies)
    S_in_Q_study = 0
    V_in_Q_study = 0
    R_in_Q_study = 0
    N_in_Q_study = 0
    Q_in_Q_study = 0

    ## Check N study in Eval S or V or R study
    cnt = 0
    indx = 0
    list_NOISE_studies_train = []
    list_NOISE_studies_eval = []
    while cnt < limit_train_NOISE and indx < len(list_NOISE_studies):
        if list_NOISE_studies[indx] in statictis_type['Eval']['S_study'] \
                or list_NOISE_studies[indx] in statictis_type['Eval']['V_study'] \
                or list_NOISE_studies[indx] in statictis_type['Eval']['R_study']:
            list_NOISE_studies_eval.append(list_NOISE_studies[indx])
        else:
            list_NOISE_studies_train.append(list_NOISE_studies[indx])
            cnt += 1

        indx += 1

    while indx < len(list_NOISE_studies):
        list_NOISE_studies_eval.append(list_NOISE_studies[indx])
        indx += 1

    for i in list_NOISE_studies_train:
        S_in_Q_study += statictis_type['Q'][i]['total_S']
        V_in_Q_study += statictis_type['Q'][i]['total_V']
        R_in_Q_study += statictis_type['Q'][i]['total_R']
        N_in_Q_study += statictis_type['Q'][i]['total_N']
        Q_in_Q_study += statictis_type['Q'][i]['total_Q']

    # if N_in_Q_study >= statictis_type['N']['total_N'] * ratio_train:
    #     break

    final_total_train_S += S_in_Q_study
    final_total_train_V += V_in_Q_study
    final_total_train_R += R_in_Q_study
    final_total_train_N += N_in_Q_study
    final_total_train_Q += Q_in_Q_study
    statictis_type['Train']['Q']['Total_S_in_Q_study'] = S_in_Q_study
    statictis_type['Train']['Q']['Total_V_in_Q_study'] = V_in_Q_study
    statictis_type['Train']['Q']['Total_R_in_Q_study'] = R_in_Q_study
    statictis_type['Train']['Q']['Total_N_in_Q_study'] = N_in_Q_study
    statictis_type['Train']['Q']['Total_Q_in_Q_study'] = Q_in_Q_study
    statictis_type['Train']['NOISE_study'] = list_NOISE_studies_train

    # times = 0
    # list_NOISE_studies = list_NOISE_studies[limit_train_NOISE:]
    # times += 1
    # print('NOISE_eval: ', times)
    # shuffle(list_N_studies)
    # shuffle(list_N_studies)
    S_in_Q_study = 0
    V_in_Q_study = 0
    R_in_Q_study = 0
    N_in_Q_study = 0
    Q_in_Q_study = 0

    for i in list_NOISE_studies_eval:
        S_in_Q_study += statictis_type['Q'][i]['total_S']
        V_in_Q_study += statictis_type['Q'][i]['total_V']
        R_in_Q_study += statictis_type['Q'][i]['total_R']
        N_in_Q_study += statictis_type['Q'][i]['total_N']
        Q_in_Q_study += statictis_type['Q'][i]['total_Q']

    final_total_eval_S += S_in_Q_study
    final_total_eval_V += V_in_Q_study
    final_total_eval_R += R_in_Q_study
    final_total_eval_N += N_in_Q_study
    final_total_eval_Q += Q_in_Q_study
    statictis_type['Eval']['Q']['Total_S_in_Q_study'] = S_in_Q_study
    statictis_type['Eval']['Q']['Total_V_in_Q_study'] = V_in_Q_study
    statictis_type['Eval']['Q']['Total_R_in_Q_study'] = R_in_Q_study
    statictis_type['Eval']['Q']['Total_N_in_Q_study'] = N_in_Q_study
    statictis_type['Eval']['Q']['Total_Q_in_Q_study'] = Q_in_Q_study
    statictis_type['Eval']['NOISE_study'] = list_NOISE_studies_eval

    statictis_type['Total_train_N'] = final_total_train_N
    statictis_type['Total_train_V'] = final_total_train_V
    statictis_type['Total_train_R'] = final_total_train_R
    statictis_type['Total_train_S'] = final_total_train_S
    statictis_type['Total_train_Q'] = final_total_train_Q

    statictis_type['Total_eval_N'] = final_total_eval_N
    statictis_type['Total_eval_V'] = final_total_eval_V
    statictis_type['Total_eval_R'] = final_total_eval_R
    statictis_type['Total_eval_S'] = final_total_eval_S
    statictis_type['Total_eval_Q'] = final_total_eval_Q

    statictis_type['ratio_train_eval_test'] = [8, 2]

    print('Train: N {}, V {}, R {}, S {}, Q {}\nEval: N {}, V {}, R {}, S {}, Q {}'.format(statictis_type['Total_train_N'],
                                                                                 statictis_type['Total_train_V'],
                                                                                 statictis_type['Total_train_R'],
                                                                                 statictis_type['Total_train_S'],
                                                                                 statictis_type['Total_train_Q'],
                                                                                 statictis_type['Total_eval_N'],
                                                                                 statictis_type['Total_eval_V'],
                                                                                 statictis_type['Total_eval_R'],
                                                                                 statictis_type['Total_eval_S'],
                                                                                 statictis_type['Total_eval_Q'],
                                                                                                                        ))

    fp = open(output_path + 'log_data_noise.json', 'w')
    fp.write(json.dumps(statictis_type, indent=4))
    fp.close()


def add_study(excel='/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/Collection/[STUDY-ID] Information.xlsx',
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
                indx  = list(df_events).index(os.path.basename(event))
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


def statistic_study(excel='/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/Collection/[STUDY-ID] Information.xlsx'):
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


    a=10

if __name__ == '__main__':
    split_data_2()
    # add_study()
    # statistic_study()


