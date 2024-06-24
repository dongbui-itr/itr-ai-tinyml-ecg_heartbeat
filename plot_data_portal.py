from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use("gtk3agg")
import wfdb as wf

event_path = '/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/eval_data/beat_maker/2024/month=03/297009/65ef8a11258bae7fe0b0dd85/event-auto-03-11-24-17-39-04-20'

atr_ai = wf.rdann(event_path, 'aiann')
atr_tech = wf.rdann(event_path, 'atrtech')
atr_ai_mark = wf.rdann(event_path, 'markaiann')
atr_tech_mark = wf.rdann(event_path, 'markatrtech')

record = wf.rdrecord(event_path)

samp_ai = atr_ai.sample
symbol_ai = atr_ai.symbol

samp_tech = atr_tech.sample
symbol_tech = atr_tech.symbol

samp_ai_mark = atr_ai_mark.sample
symbol_ai_mark = atr_ai_mark.symbol

samp_tech_mark = atr_tech_mark.sample
symbol_tech_mark = atr_tech_mark.symbol

signal = record.p_signal

plt.subplot(311)
plt.plot(signal[:, 0])
plt.plot(samp_ai_mark, signal[:, 0][samp_ai_mark], 'go')
plt.plot(samp_tech_mark, signal[:, 0][samp_tech_mark], 'r*')
plt.subplot(312)
plt.plot(signal[:, 1])
plt.plot(samp_ai_mark, signal[:, 1][samp_ai_mark], 'go')
plt.plot(samp_tech_mark, signal[:, 1][samp_tech_mark], 'r*')
plt.subplot(313)
plt.plot(signal[:, 2])
plt.plot(samp_ai_mark, signal[:, 2][samp_ai_mark], 'go')
plt.plot(samp_tech_mark, signal[:, 2][samp_tech_mark], 'r*')
plt.show()

a=10