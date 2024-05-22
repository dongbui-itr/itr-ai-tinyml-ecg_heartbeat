import numpy as np
import os

from glob import glob


def main(data_path='/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/240520/', sampling_rate='128'):
    dbs = ['mitdb', 'nstdb']
    # for db in dbs:
    list_ec57 = np.sort(np.asarray(glob(data_path + '/*_c*/*/*/*/*/{}_*_line.out'.format('*'))))

    excel_file_path = '/mnt/Project/ECG/Source_Dong/project_tinyml_4Dong/itr-ai-tinyml-ecg_heartbeat/Report/{}.csv'.format('result')
    excel_file = open(excel_file_path, 'w')

    line = 'CKT_name'
    for db in dbs:
        line += ', {}_Se, {}_P+'.format(db, db)

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

    for key in list(tmp_dict.keys()):
        line = '{}'.format(key)
        for db in dbs:
            line += ', {}, {}'.format(tmp_dict[key][db]['Se'], tmp_dict[key][db]['P+'])

        line += '\n'
        excel_file.writelines(line)

    excel_file.close()
    a=10


if __name__ == '__main__':
    main()


