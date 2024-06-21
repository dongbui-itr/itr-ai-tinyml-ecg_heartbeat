import numpy as np
import os

from glob import glob

def main(data_path='/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/240527_NSV_2/', sampling_rate='128'):
    dbs = ['mitdb', 'nstdb', 'ahadb', 'escdb', 'afdb']
    # for db in dbs:
    list_ec57 = np.sort(np.asarray(glob(data_path + '/*_c*/*/*/*/*/{}_*_line.out'.format('*'))))

    excel_file_path = '/mnt/Project/ECG/Source_Dong/project_tinyml_4Dong/itr-ai-tinyml-ecg_heartbeat_3/Report/{}_240529_NSV.csv'.format('result')
    excel_file = open(excel_file_path, 'w')

    line = 'CKT_name'
    for db in dbs:
        line += ', {}_Se, {}_P+, {}_VSe, {}_VP+, {}_SSe, {}_SP+'.format(db, db, db, db, db, db)

    excel_file.writelines(line + '\n')

    tmp_dict = dict()
    for file in list_ec57:
        db = os.path.basename(file).split('_')[0]
        ckt = [i for i in file.split('/') if sampling_rate in i and '_c' in i][0]

        if not ckt in list(tmp_dict.keys()):
            tmp_dict[ckt] = dict()

        if not db in list(tmp_dict[ckt].keys()):
            tmp_dict[ckt][db] = dict()


        fp = open(file)
        for line in fp.readlines():
            if 'Gross' in line:
                import re
                line = re.sub(r"\s+", " ", line, flags=re.UNICODE)
                tmp = line.split(' ')
                tmp_dict[ckt][db]['Se'] = tmp[1]
                tmp_dict[ckt][db]['P+'] = tmp[2]
                tmp_dict[ckt][db]['VSe'] = tmp[3]
                tmp_dict[ckt][db]['VP+'] = tmp[4]
                tmp_dict[ckt][db]['SSe'] = tmp[5]
                tmp_dict[ckt][db]['SP+'] = tmp[6]

    list_key = []
    for key in list(tmp_dict.keys()):
        line = '{}'.format(key)
        flag = True
        for db in dbs:
            # if (db == 'mitdb' and (float(tmp_dict[key][db]['Se']) < 98.5 or float(tmp_dict[key][db]['P+']) <= 99)) or \
            #         (db == 'nstdb' and (float(tmp_dict[key][db]['Se']) < 80 or float(tmp_dict[key][db]['P+']) < 88)):
            if (db == 'mitdb' and (float(tmp_dict[key][db]['Se']) < 98 or float(tmp_dict[key][db]['P+']) <= 98.5)) or \
                    (db == 'nstdb' and (float(tmp_dict[key][db]['Se']) < 80 or float(tmp_dict[key][db]['P+']) < 75)):
                flag = False
                break
            try:
                line += ', {}, {}'.format(tmp_dict[key][db]['Se'], tmp_dict[key][db]['P+'])
                line += ', {}, {}'.format(tmp_dict[key][db]['VSe'], tmp_dict[key][db]['VP+'])
                line += ', {}, {}'.format(tmp_dict[key][db]['SSe'], tmp_dict[key][db]['SP+'])
                tmp = key.split('_')[-1]
                tmp = tmp.replace('c', '')
                if not tmp in list_key:
                    list_key.append(tmp)
            except Exception as err:
                line += ', {}, {}'.format('-', '-')

        line += '\n'
        if flag:
            excel_file.writelines(line)

    print(list_key)

    excel_file.close()
    a=10


if __name__ == '__main__':
    main()


