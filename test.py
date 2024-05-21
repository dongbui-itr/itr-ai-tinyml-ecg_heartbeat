import numpy as np

beat_feature_len = 640
data_len = 23040
max = []
for k in range(40, 640, 1):
    max.append(np.max(np.arange(beat_feature_len)[None, :] + np.arange(0, data_len - beat_feature_len, beat_feature_len - k)[:, None]))

print(np.argmin(np.asarray(max) - data_len))

print("Stop")

