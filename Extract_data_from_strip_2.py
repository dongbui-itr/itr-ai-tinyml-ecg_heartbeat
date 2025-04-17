import os

import numpy as np
import wfdb as wf

from matplotlib import pyplot as plt
from glob import glob

DATA_PATH = "/mnt/4T_DATA/LLM/include-strip2/"
OUTPUT_PATH = "/mnt/4T_DATA/DATA_4TINYML/strip2"
DEBUG = False
LEN = 5 #sec

files = glob(f"{DATA_PATH}/*/*/*.hea")
# files = glob(f"{DATA_PATH}/377796/*/*.hea")
# files = glob(f"/mnt/4T_DATA/LLM/include-strip2/388215/6736f6b690e7350001219f47/event-strip-captured-2024-11-15-00-59-00-utc-06.hea")
fp_log = open(f"{OUTPUT_PATH}/log_err.csv", "w")
for no, file in enumerate(files):
    print(f"{no}/{len(files)}")
    try:
        file = file[:-4]
        header = wf.rdheader(file)
        startSample = 0
        stop_sampe = 0
        for line in header.comments:
            if "startSample" in line:
                startSample = int(line.split(":")[-1])
            elif "stopSample" in line:
                stopSample = int(line.split(":")[-1])
            elif "channel" in line:
                channel = int(line.split(":")[-1])
            elif "comment" in line:
                type = line.split(":")[-1]
            elif "eventType":
                eventType = line.split(":")[-1]

        signal, _ = wf.rdsamp(file)
        ann = wf.rdann(file, 'atr')
        symbols = np.asarray(ann.symbol)
        samples = np.asarray(ann.sample)
        name = file.replace(DATA_PATH, '')
        if DEBUG:
            plt.figure(1)
            plt.title(f"{name}\n{eventType}")
            for ch in range(len(header.sig_name)):
                if ch == 0:
                    ax = plt.subplot(len(header.sig_name) * 100 + 11 + ch)
                else:
                    plt.subplot(len(header.sig_name) * 100 + 11 + ch, sharex=ax, sharey=ax)

                plt.plot(signal[:, ch])
                plt.plot(samples, signal[:, ch][samples], '*r')
                [plt.annotate(symbols[i], (samples[i], 0.5)) for i, symbol in enumerate(symbols)]
                plt.axvspan(startSample, stopSample, alpha=0.5)

            # plt.show()

        if (stopSample - startSample) < (LEN * header.fs):
            # signal_segment = signal[startSample:startSample+LEN*header.fs, :]
        # else:
            startSample -= (LEN*header.fs - (stopSample - startSample))//2

        signal_segment = signal[startSample:startSample+LEN*header.fs, :]
        samples_segment = samples[np.flatnonzero((samples >= startSample) & (samples <= startSample + LEN * header.fs))] - startSample
        symbols_segment = symbols[np.flatnonzero((samples >= startSample) & (samples <= startSample + LEN * header.fs))]
        tmp = name.split("/")

        if DEBUG:
            plt.figure(2)
            plt.title(f"{name}")
            for ch in range(len(header.sig_name)):
                if ch == 0:
                    ax = plt.subplot(len(header.sig_name) * 100 + 11 + ch)
                else:
                    plt.subplot(len(header.sig_name) * 100 + 11 + ch, sharex=ax, sharey=ax)

                plt.plot(signal_segment[:, ch])
                plt.plot(samples_segment, signal_segment[:, ch][samples_segment], '*r')
                [plt.annotate(symbols_segment[i], (samples_segment[i], 0.5)) for i, symbol in enumerate(symbols_segment)]

        plt.show()
        plt.close(1)
        plt.close(2)

        OUTPUT_DIR = f"{OUTPUT_PATH}/{eventType}/{tmp[0]}/{tmp[1]}"
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # annotations = wf.Annotation(record_name=tmp[-1],
        #                             symbol=symbols_segment,
        #                             sample=samples_segment,
        #                             extension='atr',
        #                             fs=header.fs,
        #                             )
        # annotations.wrann(write_dir=OUTPUT_DIR, write_fs=True)
        # wf.wrsamp(record_name=tmp[-1],
        #           p_signal=signal_segment,
        #           fs=header.fs,
        #           units=header.units,
        #           sig_name=header.sig_name,
        #           adc_gain=header.adc_gain,
        #           fmt=header.fmt,
        #           baseline=header.baseline,
        #           comments=[f"from: {name}",
        #                     f"eventType: {eventType}",
        #                     f"channel: {channel}",
        #                     ],
        #           write_dir=OUTPUT_DIR
        #           )
    except Exception as err:
        print(f"{file}\n{err}")
        fp_log.writelines(f"{file}, {err}\n")
        pass