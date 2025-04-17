import wfdb as wf
from matplotlib import pyplot as plt


path = "/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/Collection_20240510_rm/export_S/174037/63fce72b082c123094dfa2b7/event-manual-02-27-23-11-18-47-24-0-2"

record = wf.rdsamp(path)
signal = record[0]
ann = wf.rdann(path, "atr")
samples = ann.sample
symbols = ann.symbol


plt.subplot(311)
plt.plot(signal[:, 0])
plt.subplot(312)
plt.plot(signal[:, 1])
plt.subplot(313)
plt.plot(signal[:, 2])
plt.plot(samples, signal[:, 2][samples], "r*")
[plt.annotate(symbols[i], (samples[i], 1)) for i in range(len(samples))]
plt.show()