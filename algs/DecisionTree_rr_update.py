# using RR + Width to classify Normal, Abnormal
import numpy as np

### config ###
THR_RR_MIN = 0.7
THR_RR_MAX = 1.25
THR_W_LOCAL = 1.6
THR_AMP_MAX = 1.8
THR_AMP_MIN = 0.55
## S RUN
THR_RUN_RR_MIN = 0.85
THR_RUN_RR_MAX = 1.15
THR_RUN_RR1_MIN = 0.6
THR_RUN_RR1_MAX = 1.28


def RR_predict(RR2_RR3, RR2_RR, RR3_RR, RR2_rr, RR3_rr,
               w_20_cali, w_50_cali, w_80_cali):
    if (
            (RR3_RR < THR_RR_MIN and RR3_rr < THR_RR_MIN)
            or (RR3_RR <= THR_RR_MAX / 2 and RR2_rr <= THR_RR_MAX / 2)
            or (RR2_RR > THR_RR_MAX and RR2_rr > THR_RR_MAX)
            or (RR2_RR3 > THR_RR_MAX)
            or (w_20_cali > THR_W_LOCAL)
            or (w_50_cali > THR_W_LOCAL)
            or (w_80_cali > THR_W_LOCAL * 1.5)):

        label = 'A'
    else:
        label = 'N'

    return label


# Normal ==> using RR to classify Normal or SVes
def normal_RR_W_RR_predict(RR1_RR2, RR2_RR3, RR1_RR, RR2_RR, RR3_RR, RR1_rr, RR2_rr, RR3_rr, symbol_bef=None):
    a=10
    if (
            (RR3_RR < THR_RR_MIN and RR3_rr < THR_RR_MIN)
            or (RR2_RR > THR_RR_MAX and RR2_rr > THR_RR_MAX)
            or (RR2_RR3 > THR_RR_MAX)
            or (RR3_RR <= THR_RR_MAX / 2 and RR2_rr <= THR_RR_MAX / 2)):
        label = 'S'
    #cond 2
    elif (RR1_RR > THR_RR_MAX and RR1_rr > THR_RR_MAX and THR_RR_MIN < RR2_RR3 < THR_AMP_MAX and symbol_bef=='S'):
        label = 'S'
    else:
        label = 'N'

    return label


# Abnormal ==> using WIDTH + AMP to classify SVes, Ves
def abnormal_W_AMP_predict(w_20_cali, w_50_cali, w_80_cali,
                           w_20_local, w_50_local, w_80_local,
                           min_amp_cali, max_amp_cali, peak_peak_cali,
                           min_amp_local, max_amp_local, peak_peak_local,
                           min_amp, max_amp, label):
    # A ==> S, V or N
    # S, N
    ind_width = np.flatnonzero(
        np.asarray([w_20_cali, w_50_cali, w_80_cali, w_20_local, w_50_local, w_80_local]) < THR_W_LOCAL)

    ind_peak = np.flatnonzero((np.asarray([peak_peak_cali, peak_peak_local]) < THR_AMP_MAX)
                              & (np.asarray([peak_peak_cali, peak_peak_local]) > THR_AMP_MIN))

    ind_amp_max_s = np.flatnonzero((np.asarray([max_amp_cali, max_amp_local]) < THR_AMP_MAX)
                                   & (np.asarray([max_amp_cali, max_amp_local]) > THR_AMP_MIN))
    ind_amp_min_s = np.flatnonzero((np.asarray([min_amp_cali, min_amp_local]) < THR_AMP_MAX)
                                   & (np.asarray([min_amp_cali, min_amp_local]) > THR_AMP_MIN))

    ind_amp_max_v = np.flatnonzero((np.asarray([max_amp_cali, max_amp_local]) > THR_AMP_MAX)
                                   | (np.asarray([max_amp_cali, max_amp_local]) < THR_AMP_MIN))

    ind_amp_max_v_2 = np.flatnonzero((np.asarray([max_amp_cali, max_amp_local]) > 2 * THR_AMP_MAX)
                                     | (np.asarray([max_amp_cali, max_amp_local]) < 2 * THR_AMP_MIN))

    ind_amp_min_v = np.flatnonzero((np.asarray([min_amp_cali, min_amp_local]) > THR_AMP_MAX)
                                   | (np.asarray([min_amp_cali, min_amp_local]) < THR_AMP_MIN))

    ind_amp_min_v_2 = np.flatnonzero((np.asarray([min_amp_cali, min_amp_local]) > 2 * THR_AMP_MAX)
                                     | (np.asarray([min_amp_cali, min_amp_local]) < 2 * THR_AMP_MIN))

    if len(ind_width) > 2 and len(ind_amp_max_s) == 2 and len(ind_amp_min_s) == 2 and len(ind_peak) == 2:
        if label != 'N':
            label = 'S'
        else:
            label = 'N'
    else:
        if label != 'N':
            if not len(ind_amp_min_s) == 2:
                if ((np.abs(min_amp) <= 0.2 and len(ind_amp_min_v_2) >= 1)
                        or (0.2 < np.abs(min_amp) and len(ind_amp_min_v) >= 1)):
                    label = 'V'
                else:
                    label = 'S'

            elif not len(ind_amp_max_s) == 2:
                if ((np.abs(max_amp) <= 0.2 and len(ind_amp_max_v_2) >= 1)
                        or (0.2 < np.abs(max_amp) and len(ind_amp_max_v) >= 1)):
                    label = 'V'
                else:
                    label = 'S'
            else:
                label = 'V'
        else:
            if not len(ind_amp_min_s) == 2 or not len(ind_amp_max_s) == 2:
                label = 'N'
            else:
                label = 'V'

    return label


# Abnormal (Normal-SVes) ==> using RR to classify Normal, SVes
def abnormal_RR_NS_predict(RR1_RR2, RR2_RR3, RR1_RR, RR2_RR, RR3_RR, RR1_rr, RR2_rr, RR3_rr):
    if (RR3_RR < THR_RUN_RR1_MIN or RR3_rr < THR_RUN_RR1_MIN) and 0.9 < RR2_RR3 < 1.1:
        if (RR3_RR < THR_RR_MIN or RR3_rr < THR_RR_MIN) and (RR2_RR < THR_RR_MIN or RR2_rr < THR_RR_MIN):
            return 'N'
        else:
            return 'S'
    elif RR3_RR <= THR_RUN_RR_MIN or RR3_rr <= THR_RUN_RR_MIN:
        if RR2_RR3 >= THR_RUN_RR_MAX:
            return 'S'
        elif RR2_RR < THR_RUN_RR_MIN and RR2_rr < THR_RUN_RR_MIN and 0.9 <= RR2_RR3 <= 1.1:
            return 'S'
        else:
            return 'N'

    # elif RR3_RR < THR_RR_MIN or RR3_rr < THR_RR_MIN:
    #     return 'N'
    else:
        return 'N'
