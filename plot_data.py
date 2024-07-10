import os.path

from matplotlib import pyplot as plt
from glob import glob
import wfdb as wf
from utils.reprocessing import beat_annotations


# import matplotlib
# matplotlib.use('TkAgg')


def review_signal():
    DATAPATH = '/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/eval_data/BeatComplexCalipers/64c8dc927c59d32cac72dd2b/'

    # C - lower
    # ['event-auto-09-26-23-10-32-12-28', 'event-auto-08-24-23-09-17-37-20', 'event-auto-08-24-23-09-16-04-20', 'event-auto-12-29-23-02-49-15-24', 'event-auto-12-29-23-04-12-27-20', 'event-auto-12-29-23-05-10-26-24', 'event-manual-10-27-23-19-33-29-16', 'event-manual-11-28-23-02-55-50-24']
    # files = [i[:-4] for i in glob(DATAPATH + '/*/event-auto-09-26-23-10-32-12-28.dat')]
    files = [i[:-4] for i in glob(DATAPATH + '/*.dat')]

    for file in files:
        record = wf.rdsamp(file)
        signal = record[0]

        ann_tech = wf.rdann(file, 'atrtech')
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

        plt.subplot(311)
        plt.plot(signal[:, 0])
        plt.plot(samp_tech, signal[:, 0][samp_tech], 'ro')
        plt.plot(samp_ai, signal[:, 0][samp_ai], 'm*')
        # plt.plot(samp_pt, signal[:, 0][samp_pt], 'kp')
        [plt.annotate(sym_tech[i], (samp_tech[i], signal[:, 0][samp_tech[i]])) for i in range(len(samp_tech))]

        plt.subplot(312)
        plt.plot(signal[:, 1])
        plt.plot(samp_tech, signal[:, 1][samp_tech], 'ro')
        plt.plot(samp_ai, signal[:, 1][samp_ai], 'm*')
        # plt.plot(samp_pt, signal[:, 1][samp_pt], 'kp')
        plt.subplot(313)
        plt.plot(signal[:, 2])
        plt.plot(samp_tech, signal[:, 2][samp_tech], 'ro')
        plt.plot(samp_ai, signal[:, 2][samp_ai], 'm*')
        # plt.plot(samp_pt, signal[:, 2][samp_pt], 'kp')
        plt.show()

        a = 10


def review_physionet():
    path = '/media/xuandung-ai/Data_4T1/AI-Database/PhysionetData/mitdb/'

    files = [p[:-4] for p in glob(path + '/*.dat')
             if os.path.basename(p)[:-4] not in ['104', '102', '107', '217', 'bw', 'em', 'ma']
             if '_200hz' not in os.path.basename(p)[:-4]]

    for file in sorted(files):
        record = wf.rdsamp(file)
        signal = record[0]

        file_name = os.path.basename(file)
        if str(file_name) in ['114', '8204']:
            ecg_raw = signal[:, 1]
        else:
            ecg_raw = signal[:, 0]

        ann_ai = wf.rdann(file, 'beatconcatseqaddmorecbciHzdibgdcaaf')
        ann_ref = wf.rdann(file, 'atr')

        sample_ai = ann_ai.sample
        sym_ai = ann_ai.symbol

        sample_ref, sym_ref = beat_annotations(ann_ref)

        fig, ax = plt.subplots(2, 1, sharex='all', sharey='all', figsize=(19.2, 10.8))
        fig.subplots_adjust(
            hspace=0.07,
            wspace=0,
            left=0.02,
            bottom=0.026,
            top=0.96,
            right=0.99
        )

        ax[0].grid(which='major', color='#CCCCCC', linestyle='--')
        ax[0].grid(which='minor', color='#CCCCCC', linestyle=':')
        ax[0].set_title(f'{file_name} - Initial R-Peaks', fontsize=9)

        ax[0].plot(ecg_raw)
        ax[0].plot(sample_ai, ecg_raw[sample_ai], 'r*', label="Predicted R-peaks")

        ax[1].grid(which='major', color='#CCCCCC', linestyle='--')
        ax[1].grid(which='minor', color='#CCCCCC', linestyle=':')
        ax[1].set_title(f'{file_name} - Reference R-Peaks', fontsize=9)

        ax[1].plot(ecg_raw)
        ax[1].plot(sample_ref, ecg_raw[sample_ref], 'kv', label="Ref R-peaks")
        ax[1].plot(sample_ai, ecg_raw[sample_ai], 'r*', label="Predicted R-peaks")

        plt.show()


if __name__ == '__main__':
    review_physionet()
