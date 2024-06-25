import os.path

from matplotlib import pyplot as plt
from glob import glob
import wfdb as wf

# import matplotlib
# matplotlib.use('TkAgg')


# DATAPATH = '/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/eval_data/BeatComplexCalipers/64c8dc927c59d32cac72dd2b/'
DATAPATH = '/mnt/Dataset/ECG/PhysionetData/mitdb/'

# C - lower
# ['event-auto-09-26-23-10-32-12-28', 'event-auto-08-24-23-09-17-37-20', 'event-auto-08-24-23-09-16-04-20', 'event-auto-12-29-23-02-49-15-24', 'event-auto-12-29-23-04-12-27-20', 'event-auto-12-29-23-05-10-26-24', 'event-manual-10-27-23-19-33-29-16', 'event-manual-11-28-23-02-55-50-24']
# files = [i[:-4] for i in glob(DATAPATH + '/*/event-auto-09-26-23-10-32-12-28.dat')]
files = [i[:-4] for i in glob(DATAPATH + '/203.dat')]

for file in files:
    record = wf.rdsamp(file)
    signal = record[0]

    ann_tech = wf.rdann(file, 'atr')
    ann_ai = wf.rdann(file, 'beatconcatseqaddmorecbciHzdibgdcaaf')
    # file_pt = (DATAPATH + '/annotations/' + os.path.basename(file)).replace('BeatComplexCalipers', '')
    # ann_pt = wf.rdann(file_pt, 'pt')

    samp_tech = ann_tech.sample
    samp_ai = ann_ai.sample
    # samp_pt = ann_pt.sample

    sym_tech = ann_tech.symbol
    sym_ai = ann_ai.symbol

    if not ('V' in sym_tech or 'S' in sym_tech):
        continue

    # sym_pt = ann_pt.symbol

    ax = plt.subplot(211)
    plt.plot(signal[:, 0])
    plt.plot(samp_tech, signal[:, 0][samp_tech], 'ro')
    plt.plot(samp_ai, signal[:, 0][samp_ai], 'k*')
    # plt.plot(samp_pt, signal[:, 0][samp_pt], 'kp')
    [plt.annotate(sym_tech[i], (samp_tech[i], signal[:, 0][samp_tech[i]]), color='r') for i in range(len(samp_tech))]
    [plt.annotate(sym_ai[i], (samp_ai[i], signal[:, 0][samp_ai[i]] + 0.2), color='k') for i in range(len(samp_ai))]

    plt.subplot(212, sharex=ax)
    plt.plot(signal[:, 1])
    plt.plot(samp_tech, signal[:, 1][samp_tech], 'ro')
    plt.plot(samp_ai, signal[:, 1][samp_ai], 'm*')

    # plt.plot(samp_pt, signal[:, 1][samp_pt], 'kp')
    # plt.subplot(313)
    # plt.plot(signal[:, 2])
    # plt.plot(samp_tech, signal[:, 2][samp_tech], 'ro')
    # plt.plot(samp_ai, signal[:, 2][samp_ai], 'm*')
    # # plt.plot(samp_pt, signal[:, 2][samp_pt], 'kp')

    plt.show()

    a=10

