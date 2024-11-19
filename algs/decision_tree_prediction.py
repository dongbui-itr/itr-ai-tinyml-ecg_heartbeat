# using RR + Width to classify Normal, Abnormal
def RR_2_predict(
        RR1_RR2, RR2_RR3, RR1_RR, RR2_RR, RR3_RR, RR1_rr, RR2_rr, RR3_rr,
        w_20_cali, w_50_cali, w_80_cali):
    if w_50_cali <= 1.2094:
        if RR1_RR <= 0.70495:
            if RR2_RR3 <= 1.62907:
                if RR3_RR <= 0.707:
                    if w_50_cali <= 1.03189:
                        if w_80_cali <= 1.06296:
                            if RR3_RR <= 0.59704 and RR2_rr <= 0.54398:
                                return 'A'
                            elif (RR3_RR <= 0.59704 and RR2_rr > 0.54398) or (RR3_RR > 0.59704 and RR1_RR <= 0.59644):
                                return 'N'
                            else:  # if RR1_RR > 0.59644
                                if w_80_cali <= 0.97679 and RR1_RR2 <= 1.06015:
                                    return 'A'
                                elif (w_80_cali <= 0.97679 and RR1_RR2 > 1.06015) or (w_80_cali > 0.97679):
                                    return 'N'
                        elif w_80_cali > 1.06296 and w_20_cali <= 0.97826:
                            return 'A'
                        else:  # if w_20_cali_w20 > 0.97826
                            if RR1_RR2 <= 0.80881:
                                return 'N'
                            else:  # if RR1_RR2 > 0.80881
                                if RR1_RR2 <= 1.07294 and w_80_cali <= 1.09722:
                                    return 'A'
                                elif (RR1_RR2 <= 1.07294 and w_80_cali > 1.09722) or (RR1_RR2 > 1.07294):
                                    return 'N'
                    elif w_50_cali > 1.03189 and RR2_RR <= 0.8609:
                        return 'N'
                    else:  # if w_50_cali_w50 > 1.03189 and RR2_RR > 0.8609:
                        return 'A'
                else:  # if RR3_RR > 0.707
                    if w_80_cali > 1.56517 and RR3_rr <= 0.98379:
                        return 'A'
                    else:  # if (w_80_cali_w80 <= 1.56517) or (w_80_cali_w80 > 1.56517 and RR3_rr > 0.98379):
                        return 'N'
            else:  # if RR2_RR3 > 1.62907
                if RR3_RR <= 0.80482 and RR1_rr <= 1.04737:
                    return 'A'
                else:  # if (RR3_RR <= 0.80482 and RR1_rr > 1.04737) or (RR3_RR > 0.80482):
                    return 'N'
        else:  # if RR1_RR > 0.70495
            if RR3_rr <= 0.81065:
                if RR2_rr <= 0.9372:
                    if w_80_cali <= 1.06296:
                        if RR2_RR3 <= 1.32616:
                            if RR1_rr <= 0.89704:
                                return 'N'
                            else:  # if RR1_rr > 0.89704
                                if RR2_RR <= 0.77036:
                                    if (RR2_RR3 <= 0.83751) or (
                                            RR2_RR3 > 0.83751 and (RR2_RR3 <= 0.97512 and w_80_cali <= 0.90648)):
                                        return 'N'
                                    else:  # if (RR2_RR3 <= 0.97512 and w_80_cali_w80 > 0.90648) or (RR2_RR3 > 0.97512):
                                        return 'A'
                                elif RR2_RR > 0.77036 and w_80_cali <= 0.68981:
                                    return 'A'
                                else:  # if RR2_RR > 0.77036 and w_80_cali_w80 > 0.68981:
                                    return 'N'
                        elif RR2_RR3 > 1.32616 and RR1_RR <= 1.18462:
                            return 'A'
                        else:  # if RR1_RR > 1.18462
                            if RR3_RR > 0.65521 and RR3_rr <= 0.65685:
                                return 'N'
                            else:  # if (RR3_RR > 0.65521 and RR3_rr > 0.65685) or RR3_RR <= 0.65521:
                                return 'A'

                    else:  # if w_80_cali_w80 > 1.06296
                        if w_50_cali <= 1.06252:
                            if RR1_RR <= 0.94802:
                                if w_20_cali <= 0.97826:
                                    return 'A'
                                else:  # if w_20_cali_w20 > 0.97826
                                    return 'N'
                            else:  # if RR1_RR > 0.94802
                                return 'A'
                        else:  # if w_50_cali_w50 > 1.06252
                            if w_20_cali <= 1.20571:
                                if RR3_RR > 0.572 and w_20_cali > 0.87137 and w_80_cali <= 1.24573:
                                    return 'N'
                                else:  # if (w_20_cali_w20 > 0.87137 and w_80_cali_w80 > 1.24573) or (w_20_cali_w20 <= 0.87137):
                                    return 'A'
                            else:  # if w_20_cali_w20 > 1.20571
                                return 'A'

                else:  # if RR2_rr > 0.9372
                    if w_20_cali <= 0.84616:
                        if w_20_cali <= 0.83557:
                            if (RR1_rr <= 1.09762 and RR3_RR <= 0.84394) or (RR1_rr > 1.09762 and RR3_rr <= 0.6447):
                                return 'A'
                            else:  # if RR3_rr > 0.6447
                                return 'N'
                        else:  # if w_20_cali_w20 > 0.83557
                            return 'N'

                    else:  # if w_20_cali_w20 > 0.84616
                        if RR3_RR <= 0.82761:
                            if RR2_RR <= 0.80214:
                                return 'N'
                            else:  # if RR2_RR > 0.80214
                                if RR2_RR3 <= 3.48928 and RR1_RR2 <= 0.92002:
                                    if (RR2_RR <= 0.95268) or (RR2_RR > 0.95268 and RR2_RR3 <= 1.42504):
                                        return 'N'
                                    else:  # if RR2_RR > 0.95268 and RR2_RR3 > 1.42504:
                                        return 'A'
                                elif RR2_RR3 <= 3.48928 and RR1_RR2 > 0.92002:
                                    return 'A'
                                else:  # if RR2_RR3 > 3.48928
                                    return 'N'
                        else:  # if RR3_RR > 0.82761
                            if w_20_cali <= 1.17971 and RR2_RR3 <= 1.94609:
                                return 'N'
                            else:  # if (w_20_cali_w20 <= 1.17971 and RR2_RR3 > 1.94609) or (w_20_cali_w20 > 1.17971):
                                return 'A'

            else:  # if RR3_rr > 0.81065
                if RR2_RR3 <= 1.15899:
                    if RR2_RR3 <= 1.1088:
                        if RR3_RR <= 0.68973 and RR1_rr > 1.70038:
                            return 'A'
                        else:  # if RR3_RR > 0.68973 or (RR3_RR <= 0.68973 and RR1_rr <= 1.70038):
                            return 'N'

                    else:  # if RR2_RR3 > 1.1088
                        if (RR3_rr <= 0.90686 and RR3_RR <= 1.2886) or (RR3_rr > 0.90686 and RR1_RR <= 1.6331):
                            return 'N'
                        elif (RR3_rr <= 0.90686 and RR3_RR > 1.2886) or (RR3_rr > 0.90686 and RR1_RR > 1.6331):
                            return 'A'

                else:  # if RR2_RR3 > 1.15899
                    if RR3_RR <= 0.87088:
                        if RR1_rr <= 1.04058:
                            if RR2_rr <= 1.03272 and RR1_rr <= 0.95135:
                                return 'N'
                            elif RR2_rr <= 1.03272 and RR1_rr > 0.95135:
                                return 'A'
                            else:  # if RR2_rr > 1.03272
                                if w_50_cali <= 0.94945 and RR1_RR2 > 0.66357:
                                    return 'N'
                                else:  # if (w_50_cali_w50 <= 0.94945 and RR1_RR2 <= 0.66357) or (w_50_cali_w50 > 0.94945):
                                    return 'A'
                        else:  # if RR1_rr > 1.04058
                            if (RR1_rr <= 1.35549) or (RR1_rr > 1.35549 and w_20_cali <= 0.94635):
                                return 'N'
                            else:  # if w_20_cali_w20 > 0.94635
                                return 'A'
                    else:  # if RR3_RR > 0.87088
                        if (RR1_RR > 1.67393) and (RR1_rr <= 1.4 and RR2_RR <= 1.88304):
                            return 'A'
                        else:  # if (RR1_rr <= 1.4 and  RR2_RR > 1.88304) or (RR1_rr > 1.4):
                            return 'N'

    else:  # if w_50_cali_w50 > 1.2094
        if w_80_cali <= 1.388:
            if RR3_RR <= 0.83923:
                if w_50_cali <= 1.25229:
                    return 'A'
                else:  # if w_50_cali_w50 > 1.25229
                    if w_50_cali <= 1.26076:
                        return 'N'
                    else:  # if w_50_cali_w50 > 1.26076
                        if (RR2_RR <= 0.93424 and w_80_cali <= 1.17897) or (RR2_RR > 0.93424 and RR3_rr <= 0.88262):
                            return 'A'
                        elif RR2_RR <= 0.93424 and w_80_cali > 1.17897:
                            if RR1_rr <= 0.70664:
                                return 'N'
                            else:  # if RR1_rr > 0.70664
                                if (w_80_cali <= 1.23734 and RR1_RR2 <= 1.55616) or (
                                        w_80_cali > 1.23734 and (RR3_RR > 0.64563 and w_50_cali <= 1.30437)):
                                    return 'N'
                                else:  # if (w_80_cali_w80 <= 1.23734 and  RR1_RR2 > 1.55616) or (RR3_RR > 0.64563 and w_50_cali_w50 > 1.30437) or (RR3_RR <= 0.64563):
                                    return 'A'
                        else:  # if RR3_rr > 0.88262
                            return 'N'
            else:  # if RR3_RR > 0.83923
                if w_50_cali <= 1.35585:
                    if RR3_rr <= 0.91298:
                        if RR2_RR <= 1.39601 and w_80_cali <= 1.31186:
                            return 'N'
                        elif (RR2_RR <= 1.39601 and w_80_cali > 1.31186) or (RR2_RR > 1.39601):
                            return 'A'
                    else:  # if RR3_rr > 0.91298
                        return 'N'

                else:  # if w_50_cali_w50 > 1.35585
                    if (RR3_RR <= 0.92315) or (RR3_RR > 0.92315 and RR3_rr <= 0.95346):
                        return 'A'
                    else:  # if RR3_rr > 0.95346
                        return 'N'
        else:  # if w_80_cali_w80 > 1.388
            if RR3_rr <= 0.97212:
                return 'A'
            else:  # if RR3_rr > 0.97212
                if w_50_cali <= 2.40879:
                    if w_20_cali <= 1.5286:
                        if w_20_cali <= 1.375:
                            if RR3_RR <= 1.02886 or (RR3_RR > 1.02886 and RR1_RR <= 0.98094):
                                return 'A'
                            else:  # if RR1_RR > 0.98094
                                return 'N'
                        else:  # if w_20_cali_w20 > 1.375
                            return 'N'
                    else:  # if w_20_cali_w20 > 1.5286
                        if (w_50_cali <= 1.7891) or (w_50_cali > 1.7891 and RR1_RR2 <= 0.57504):
                            return 'A'
                        else:  # if w_50_cali_w50 > 1.7891 and RR1_RR2 > 0.57504
                            return 'N'
                else:  # if w_50_cali_w50 > 2.40879
                    return 'A'


# Normal ==> using WIDTH to classify Normal or Ves
def normal_W_AMP_2_predict(w_20_cali, w_50_cali, w_80_cali, w_20_local, w_50_local, w_80_local,
                           min_amp_cali, max_amp_cali, peak_peak_cali, min_amp_local,
                           max_amp_local, peak_peak_local):
    if max_amp_cali <= 0.33701:
        if w_50_cali <= 0.95951:
            if min_amp_local <= 1.57231:
                if max_amp_cali <= 0.33543 and ((peak_peak_local > 0.33338)
                                                or (peak_peak_local <= 0.32324)):
                    return 'N'
                else:  # if (max_amp_cali_max_amp > 0.33543) or
                    # (max_amp_cali_max_amp <= 0.33543 and peak_peak_local_peak_peak <= 0.33338
                    # and peak_peak_local_peak_peak > 0.32324):
                    return 'V'
            else:  # if min_amp_local_min_amp > 1.57231
                if ((min_amp_cali <= 3.00978 and w_80_cali > 0.88074 and min_amp_local > 3.4749)
                        or min_amp_cali > 3.00978 and max_amp_local <= 0.32261):
                    return 'V'
                else:  # if (min_amp_cali_min_amp > 3.00978 and max_amp_local_max_amp > 0.32261)
                    # or (min_amp_cali_min_amp <= 3.00978 and ((w_80_cali_w80 <= 0.88074)
                    # or (w_80_cali_w80 > 0.88074 and min_amp_local_min_amp <= 3.4749))):
                    return 'N'
        else:  # if w_50_cali_w50 > 0.95951
            if min_amp_cali <= 1.76494:
                if w_20_local <= 0.83557 and min_amp_cali > 0.9891 and w_80_cali > 1.75196:
                    return 'V'
                else:
                    return 'N'
            else:  # if min_amp_cali_min_amp > 1.76494
                if max_amp_local <= 0.06135:
                    return 'N'
                else:  # if max_amp_local_max_amp > 0.06135
                    if min_amp_local <= 9.65552 and w_20_cali <= 2.16555:
                        if ((peak_peak_local <= 0.55652 and w_20_local > 1.08792)
                                or (peak_peak_local > 0.55652 and max_amp_cali > 0.25426
                                    and (peak_peak_cali > 0.97875 and min_amp_local <= 4.18612))):
                            return 'N'
                        else:
                            return 'V'
                    else:  # if w_20_cali_w20 > 2.16555 and min_amp_local_min_amp > 9.65552
                        return 'N'

    else:  # if max_amp_cali_max_amp > 0.33701
        if min_amp_cali <= 2.82118:
            if w_80_local <= 1.6016:
                if peak_peak_cali <= 2.76304:
                    if w_80_local <= 1.17752:
                        if max_amp_cali <= 0.82203:
                            if max_amp_local <= 1.49178:
                                if max_amp_local <= 0.60943:
                                    if min_amp_cali <= 1.46607:
                                        if peak_peak_local <= 0.79706:
                                            if ((w_50_cali <= 0.39484) or (
                                                    (0.50359 < max_amp_local <= 0.50374)
                                                    or (max_amp_local <= 0.50359
                                                        and min_amp_cali > 1.23573
                                                        and 1.05627 < w_50_local <= 1.12325))):
                                                return 'V'
                                            else:  # if max_amp_local_max_amp > 0.50374
                                                return 'N'
                                        else:  # if peak_peak_local_peak_peak > 0.79706
                                            if (peak_peak_cali <= 0.88347 and w_50_cali > 0.82778
                                                    and max_amp_local > 0.4867 and w_20_cali > 0.82875):
                                                return 'V'
                                            else:
                                                return 'N'
                                    else:  # if min_amp_cali_min_amp > 1.46607
                                        if peak_peak_cali <= 1.04555:
                                            if ((w_80_local <= 0.73646 and min_amp_cali <= 1.50324)
                                                    or (w_80_local > 0.73646 and (
                                                            w_20_cali <= 1.8 and w_50_local > 0.77657
                                                            and w_50_cali > 0.67222 and max_amp_cali > 0.35813))):
                                                return 'V'
                                            else:
                                                return 'N'
                                        else:  # if peak_peak_cali_peak_peak > 1.04555
                                            if min_amp_local <= 2.24086:
                                                return 'V'
                                            else:  # if min_amp_local_min_amp > 2.24086
                                                return 'N'
                                else:  # if max_amp_local_max_amp > 0.60943
                                    if min_amp_local <= 1.3777:
                                        if ((w_50_local <= 0.62382 and 0.66215 < w_80_local <= 1.10672
                                             and w_20_local > 0.76969 and peak_peak_cali <= 0.7412)
                                                or (w_80_local > 1.10672
                                                    and peak_peak_cali > 0.99426
                                                    and w_80_cali > 1.16111)):
                                            return 'V'
                                        else:
                                            return 'N'
                                    else:  # if min_amp_local_min_amp > 1.3777
                                        if w_20_cali > 1.08921 and min_amp_local <= 1.38732:
                                            return 'V'
                                        else:  # if min_amp_local_min_amp > 1.38732
                                            return 'N'
                            else:  # if max_amp_local_max_amp > 1.49178
                                if w_80_cali <= 1.06026:
                                    return 'N'
                                else:  # if w_80_cali_w80 > 1.06026
                                    if (w_50_local <= 0.6522
                                            or (w_50_local > 0.6522
                                                and w_80_local > 0.95671
                                                and max_amp_cali <= 0.78334)):
                                        return 'N'
                                    else:  # if max_amp_cali_max_amp > 0.78334
                                        return 'V'
                        else:  # if max_amp_cali_max_amp > 0.82203
                            if min_amp_local <= 1.85715:
                                if w_80_local <= 1.12069:
                                    if max_amp_local <= 1.85518:
                                        if peak_peak_local <= 1.29444:
                                            if w_80_cali <= 0.66133:
                                                if max_amp_local <= 1.03119 and max_amp_cali > 0.98737:
                                                    if ((w_50_local <= 0.6083
                                                         and peak_peak_local > 0.89568
                                                         and w_50_cali > 0.62448)
                                                            or (w_50_local > 0.6083
                                                                and max_amp_local > 1.03042)):
                                                        return 'V'
                                                    else:
                                                        return 'N'
                                                else:  # if max_amp_local_max_amp > 1.03119
                                                    return 'N'
                                            else:  # if w_80_cali_w80 > 0.66133
                                                if w_50_cali <= 1.19783:
                                                    if peak_peak_cali <= 0.85828 and peak_peak_local <= 1.02042:
                                                        if (
                                                                0.99107 < w_20_local <= 1.00926 and max_amp_local <= 0.9008
                                                                and peak_peak_cali > 0.8449 and min_amp_cali <= 0.72407):
                                                            return 'V'
                                                        else:
                                                            return 'N'
                                                    elif (
                                                            peak_peak_cali <= 0.85828 and peak_peak_local > 1.02042
                                                            and max_amp_local > 1.55152 and w_80_cali <= 0.90991):
                                                        return 'V'
                                                    else:  # if peak_peak_cali_peak_peak > 0.85828
                                                        return 'N'
                                                else:  # if w_50_cali_w50 > 1.19783
                                                    if ((
                                                            w_50_local <= 0.69112 and 0.67496 >= min_amp_cali > 0.6696)
                                                            or (
                                                                    w_50_local > 0.69112 and min_amp_local <= 0.73674
                                                                    and 1.05142 >= max_amp_cali > 1.04978)):
                                                        return 'V'
                                                    else:
                                                        return 'N'
                                        else:  # if peak_peak_local_peak_peak > 1.29444
                                            if ((
                                                    w_80_local <= 1.08443 and w_20_cali <= 0.72269 and w_20_local > 0.7079)
                                                    or (
                                                            w_80_local > 1.08443 and max_amp_cali > 1.23205 and max_amp_local <= 1.29958)):
                                                return 'V'
                                            else:  # if max_amp_local_max_amp > 1.29958
                                                return 'N'
                                    else:  # if max_amp_local_max_amp > 1.85518
                                        if min_amp_cali <= 1.44043:
                                            if (w_50_cali <= 1.09396 and (
                                                    0.9799 < min_amp_local <= 0.99069)
                                                    or (
                                                            w_50_cali > 1.09396 and peak_peak_local <= 1.30913)):
                                                return 'V'
                                            else:  # if peak_peak_local_peak_peak > 1.30913
                                                return 'N'
                                        else:  # if min_amp_cali_min_amp > 1.44043
                                            return 'V'
                                else:  # if w_80_local_w80 > 1.12069
                                    if min_amp_cali <= 2.05393:
                                        if peak_peak_local <= 1.1527:
                                            if min_amp_cali <= 1.50715:
                                                if (w_80_local > 1.1561 and peak_peak_cali <= 0.99852
                                                        and 1.22986 < w_50_cali <= 1.23847 and min_amp_cali <= 0.92256):
                                                    return 'V'
                                                else:
                                                    return 'N'
                                            else:  # if min_amp_cali_min_amp > 1.50715
                                                if min_amp_cali <= 1.50972:
                                                    return 'V'
                                                else:  # if min_amp_cali_min_amp > 1.50972
                                                    return 'N'
                                        else:  # if peak_peak_local_peak_peak > 1.1527
                                            if peak_peak_local <= 1.15319:
                                                return 'V'
                                            else:  # if peak_peak_local_peak_peak > 1.15319
                                                if ((
                                                        min_amp_local <= 0.56901 and max_amp_local <= 1.45292 and w_50_cali > 1.13037)
                                                        or (min_amp_local > 0.56901 and (
                                                                peak_peak_local > 1.23307 and 1.23208 < w_80_cali <= 1.26803))):
                                                    return 'V'
                                                else:
                                                    return 'N'
                                    else:  # if min_amp_cali_min_amp > 2.05393
                                        if w_50_local <= 0.93507:
                                            return 'V'
                                        else:  # if w_50_local_w50 > 0.93507
                                            return 'N'
                            else:  # if min_amp_local_min_amp > 1.85715
                                if peak_peak_cali <= 1.46651:
                                    if ((min_amp_local <= 1.86434)
                                            or (min_amp_local > 1.86434
                                                and peak_peak_cali > 1.37124
                                                and w_20_local > 1.06783)):
                                        return 'V'
                                    else:
                                        return 'N'
                                else:  # if peak_peak_cali_peak_peak > 1.46651
                                    if w_80_local <= 1.16765:
                                        return 'V'
                                    else:  # if w_80_local_w80 > 1.16765
                                        return 'N'
                    else:  # if w_80_local_w80 > 1.17752
                        if min_amp_local <= 1.49427:
                            if max_amp_local <= 0.35182:
                                if peak_peak_local <= 0.61011 and peak_peak_cali > 0.62988:
                                    return 'V'
                                else:  # if peak_peak_local_peak_peak > 0.61011
                                    return 'N'
                            else:  # if max_amp_local_max_amp > 0.35182
                                if peak_peak_local <= 1.4581:
                                    if peak_peak_local <= 1.29809:
                                        if ((w_80_cali <= 1.87607 and 1.37501 < w_50_cali <= 1.3875)
                                                or ((w_80_cali > 1.87607 and max_amp_local > 1.23783)
                                                    and (
                                                            peak_peak_local <= 1.165 and min_amp_local > 0.91295))):
                                            return 'V'
                                        else:
                                            return 'N'
                                    else:  # if peak_peak_local_peak_peak > 1.29809
                                        if w_50_local <= 1.19939:
                                            if max_amp_cali <= 1.75956:
                                                if w_80_local <= 1.23083 and w_20_cali > 1.18749 and w_80_cali <= 1.51148:
                                                    return 'V'
                                                else:  # if w_80_local_w80 > 1.23083
                                                    return 'N'
                                            else:  # if max_amp_cali_max_amp > 1.75956
                                                return 'V'
                                        else:  # if w_50_local_w50 > 1.19939
                                            return 'N'
                                else:  # if peak_peak_local_peak_peak > 1.4581
                                    if min_amp_local <= 0.86018:
                                        if peak_peak_local <= 1.51499:
                                            if w_50_local <= 1.42252:
                                                if w_20_cali <= 1.38963:
                                                    return 'V'
                                                else:  # if w_20_cali_w20 > 1.38963
                                                    if w_80_cali <= 1.42593:
                                                        return 'N'
                                                    else:  # if w_80_cali_w80 > 1.42593
                                                        return 'V'
                                            else:  # if w_50_local_w50 > 1.42252
                                                return 'N'
                                        else:  # if peak_peak_local_peak_peak > 1.51499
                                            return 'N'
                                    else:  # if min_amp_local_min_amp > 0.86018
                                        return 'N'
                        else:  # if min_amp_local_min_amp > 1.49427
                            if peak_peak_cali <= 1.17499:
                                return 'N'
                            else:  # if peak_peak_cali_peak_peak > 1.17499
                                if w_20_cali <= 1.03921:
                                    if max_amp_cali <= 1.12596:
                                        return 'V'
                                    else:  # if max_amp_cali_max_amp > 1.12596
                                        return 'N'
                                else:  # if w_20_cali_w20 > 1.03921
                                    if w_80_cali <= 1.11087:
                                        return 'N'
                                    else:  # if w_80_cali_w80 > 1.11087
                                        return 'V'
                else:  # if peak_peak_cali_peak_peak > 2.76304
                    return 'V'
            else:  # if w_80_local_w80 > 1.6016
                if max_amp_cali <= 1.29593:
                    if min_amp_local <= 1.82048:
                        if w_80_local <= 1.65768:
                            if min_amp_local <= 1.18378:
                                return 'N'
                            else:  # if min_amp_local_min_amp > 1.18378
                                if min_amp_local <= 1.25551:
                                    return 'V'
                                else:  # if min_amp_local_min_amp > 1.25551
                                    return 'N'
                        else:  # if w_80_local_w80 > 1.65768
                            return 'N'
                    else:  # if min_amp_local_min_amp > 1.82048
                        if w_50_cali <= 1.06163:
                            return 'N'
                        else:  # if w_50_cali_w50 > 1.06163
                            return 'V'
                else:  # if max_amp_cali_max_amp > 1.29593
                    if peak_peak_local <= 1.3201:
                        if w_20_local <= 2.13396:
                            if peak_peak_local <= 1.12697:
                                return 'V'
                            else:  # if peak_peak_local_peak_peak > 1.12697
                                if peak_peak_local <= 1.2078:
                                    if w_50_local <= 1.73937:
                                        return 'N'
                                    else:  # if w_50_local_w50 > 1.73937
                                        if w_50_local <= 1.79353:
                                            return 'V'
                                        else:  # if w_50_local_w50 > 1.79353
                                            return 'N'
                                else:  # if peak_peak_local_peak_peak > 1.2078
                                    return 'N'
                        else:  # if w_20_local_w20 > 2.13396
                            if w_80_local <= 1.8731:
                                return 'N'
                            else:  # if w_80_local_w80 > 1.8731
                                return 'V'
                    else:  # if peak_peak_local_peak_peak > 1.3201
                        return 'V'
        else:  # if min_amp_cali_min_amp > 2.82118
            if max_amp_local <= 0.68421:
                if w_80_cali <= 0.78526:
                    return 'N'
                else:  # if w_80_cali_w80 > 0.78526
                    if max_amp_local <= 0.38835:
                        return 'N'
                    else:  # if max_amp_local_max_amp > 0.38835
                        if max_amp_cali <= 0.5588:
                            if peak_peak_local <= 0.98937:
                                if w_80_cali <= 0.96759:
                                    if w_50_local <= 0.72364:
                                        if max_amp_cali <= 0.39355:
                                            return 'N'
                                        else:  # if max_amp_cali_max_amp > 0.39355
                                            return 'V'
                                    else:  # if w_50_local_w50 > 0.72364
                                        return 'V'
                                else:  # if w_80_cali_w80 > 0.96759
                                    if w_20_local <= 0.81793:
                                        if w_20_local <= 0.56675:
                                            return 'N'
                                        else:  # if w_20_local_w20 > 0.56675
                                            return 'V'
                                    else:  # if w_20_local_w20 > 0.81793
                                        return 'N'
                            else:  # if peak_peak_local_peak_peak > 0.98937
                                if min_amp_local <= 5.03747:
                                    return 'N'
                                else:  # if min_amp_local_min_amp > 5.03747
                                    return 'V'
                        else:  # if max_amp_cali_max_amp > 0.5588
                            return 'V'
            else:  # if max_amp_local_max_amp > 0.68421
                if min_amp_cali <= 2.86302:
                    if peak_peak_local <= 1.19381:
                        return 'V'
                    else:  # if peak_peak_local_peak_peak > 1.19381
                        return 'N'
                else:  # if min_amp_cali_min_amp > 2.86302
                    if w_20_cali <= 2.22342:
                        if peak_peak_local <= 1.2523:
                            return 'N'
                        else:  # if peak_peak_local_peak_peak > 1.2523
                            if min_amp_local <= 4.37733:
                                return 'N'
                            else:  # if min_amp_local_min_amp > 4.37733
                                if w_20_cali <= 0.92307:
                                    return 'V'
                                else:  # if w_20_cali_w20 > 0.92307
                                    return 'N'
                    else:  # if w_20_cali_w20 > 2.22342
                        if min_amp_cali <= 6.26624:
                            return 'V'
                        else:  # if min_amp_cali_min_amp > 6.26624
                            return 'N'


# Normal ==> using RR to classify Normal or SVes
def normal_RR_W_RR_2_predict(RR1_RR2, RR2_RR3, RR1_RR, RR2_RR, RR3_RR, RR1_rr, RR2_rr, RR3_rr):
    if RR3_RR <= 0.80689:
        if RR2_RR <= 0.85398:
            if RR3_rr <= 0.50908:
                if RR2_rr <= 0.55516:
                    if RR2_RR <= 0.50372:
                        return 'S'
                    else:  # if RR2_RR > 0.50372
                        if RR3_rr <= 0.44924:
                            if RR2_RR <= 0.64472:
                                return 'S'
                            else:  # if RR2_RR > 0.64472
                                if RR1_rr <= 0.90078:
                                    return 'N'
                                else:  # if RR1_rr > 0.90078
                                    return 'S'
                        else:  # if RR3_rr > 0.44924
                            return 'N'
                else:  # if RR2_rr > 0.55516
                    return 'N'
            else:  # if RR3_rr > 0.50908
                if RR2_RR3 <= 0.63855:
                    if RR3_rr <= 0.88416:
                        if RR2_rr <= 0.45666:
                            return 'N'
                        else:  # if RR2_rr > 0.45666
                            return 'S'
                    else:  # if RR3_rr > 0.88416
                        return 'N'
                else:  # if RR2_RR3 > 0.63855
                    if RR3_RR <= 0.67469:
                        if RR1_rr <= 0.54359:
                            if RR2_RR3 <= 1.06222:
                                if RR3_RR <= 0.64994:
                                    return 'S'
                                else:  # if RR3_RR > 0.64994
                                    if RR1_rr <= 0.49442:
                                        return 'S'
                                    else:  # if RR1_rr > 0.49442
                                        return 'N'
                            else:  # if RR2_RR3 > 1.06222
                                return 'N'
                        else:  # if RR1_rr > 0.54359
                            if RR2_RR <= 0.45897:
                                return 'N'
                            else:  # if RR2_RR > 0.45897
                                if RR2_RR <= 0.45946:
                                    return 'S'
                                else:  # if RR2_RR > 0.45946
                                    if RR2_rr <= 1.01882:
                                        if RR1_RR <= 0.73831:
                                            if RR3_RR <= 0.48897:
                                                if RR2_rr <= 0.68439:
                                                    if RR2_RR3 <= 1.08962:
                                                        if RR3_rr <= 0.57677:
                                                            return 'N'
                                                        else:  # if RR3_rr > 0.57677
                                                            return 'S'
                                                    else:  # if RR2_RR3 > 1.08962
                                                        return 'N'
                                                else:  # if RR2_rr > 0.68439
                                                    return 'N'
                                            else:  # if RR3_RR > 0.48897
                                                if RR1_rr <= 0.96707:
                                                    return 'N'
                                                else:  # if RR1_rr > 0.96707
                                                    if RR1_RR2 <= 1.00265:
                                                        if RR3_RR <= 0.58612:
                                                            return 'N'
                                                        else:  # if RR3_RR > 0.58612
                                                            return 'S'
                                                    else:  # if RR1_RR2 > 1.00265
                                                        return 'N'
                                        else:  # if RR1_RR > 0.73831
                                            if RR2_rr <= 0.96163:
                                                if RR3_rr <= 0.82583:
                                                    if RR3_RR <= 0.64668:
                                                        return 'N'
                                                    else:  # if RR3_RR > 0.64668
                                                        if RR3_RR <= 0.6469:
                                                            return 'S'
                                                        else:  # if RR3_RR > 0.6469
                                                            return 'N'
                                                else:  # if RR3_rr > 0.82583
                                                    if RR1_rr <= 1.06815:
                                                        return 'S'
                                                    else:  # if RR1_rr > 1.06815
                                                        if RR3_rr <= 0.82669:
                                                            return 'S'
                                                        else:  # if RR3_rr > 0.82669
                                                            if RR3_RR <= 0.6648:
                                                                return 'N'
                                                            else:  # if RR3_RR > 0.6648
                                                                return 'S'
                                            else:  # if RR2_rr > 0.96163
                                                if RR1_RR2 <= 1.2269:
                                                    return 'S'
                                                else:  # if RR1_RR2 > 1.2269
                                                    return 'N'
                                    else:  # if RR2_rr > 1.01882
                                        return 'N'
                    else:  # if RR3_RR > 0.67469
                        if RR1_RR2 <= 1.38878:
                            return 'N'
                        else:  # if RR1_RR2 > 1.38878
                            if RR1_RR <= 0.97862:
                                if RR1_RR2 <= 1.40972:
                                    if RR2_rr <= 0.8891:
                                        return 'S'
                                    else:  # if RR2_rr > 0.8891
                                        return 'N'
                                else:  # if RR1_RR2 > 1.40972
                                    return 'N'
                            else:  # if RR1_RR > 0.97862
                                return 'N'
        else:  # if RR2_RR > 0.85398
            if RR1_rr <= 0.56353:
                if RR2_RR3 <= 1.44295:
                    return 'N'
                else:  # if RR2_RR3 > 1.44295
                    if RR1_RR <= 0.5582:
                        if RR1_RR <= 0.42656:
                            return 'N'
                        else:  # if RR1_RR > 0.42656
                            return 'S'
                    else:  # if RR1_RR > 0.5582
                        return 'N'
            else:  # if RR1_rr > 0.56353
                if RR3_rr <= 0.45583:
                    if RR1_RR <= 1.94836:
                        return 'S'
                    else:  # if RR1_RR > 1.94836
                        return 'N'
                else:  # if RR3_rr > 0.45583
                    if RR2_RR3 <= 1.16203:
                        if RR2_RR3 <= 1.1409:
                            if RR2_RR3 <= 1.11222:
                                return 'N'
                            else:  # if RR2_RR3 > 1.11222
                                if RR1_RR <= 0.91836:
                                    if RR2_rr <= 1.00713:
                                        if RR1_rr <= 0.97534:
                                            return 'N'
                                        else:  # if RR1_rr > 0.97534
                                            return 'S'
                                    else:  # if RR2_rr > 1.00713
                                        return 'N'
                                else:  # if RR1_RR > 0.91836
                                    return 'N'
                        else:  # if RR2_RR3 > 1.1409
                            if RR2_RR <= 0.867:
                                return 'S'
                            else:  # if RR2_RR > 0.867
                                if RR3_RR <= 0.80466:
                                    if RR2_rr <= 0.98602:
                                        if RR1_RR2 <= 1.04803:
                                            return 'N'
                                        else:  # if RR1_RR2 > 1.04803
                                            if RR1_RR <= 0.9737:
                                                return 'S'
                                            else:  # if RR1_RR > 0.9737
                                                return 'N'
                                    else:  # if RR2_rr > 0.98602
                                        return 'N'
                                else:  # if RR3_RR > 0.80466
                                    if RR3_RR <= 0.80636:
                                        return 'S'
                                    else:  # if RR3_RR > 0.80636
                                        if RR3_rr <= 0.92895:
                                            return 'N'
                                        else:  # if RR3_rr > 0.92895
                                            return 'S'
                    else:  # if RR2_RR3 > 1.16203
                        if RR3_rr <= 0.71872:
                            return 'N'
                        else:  # if RR3_rr > 0.71872
                            if RR1_rr <= 0.65788:
                                if RR2_RR <= 1.26158:
                                    if RR1_rr <= 0.64606:
                                        return 'N'
                                    else:  # if RR1_rr > 0.64606
                                        return 'S'
                                else:  # if RR2_RR > 1.26158
                                    return 'S'
                            else:  # if RR1_rr > 0.65788
                                if RR1_RR <= 0.94511:
                                    if RR1_RR <= 0.68956:
                                        if RR2_rr <= 0.96512:
                                            return 'S'
                                        else:  # if RR2_rr > 0.96512
                                            if RR1_RR2 <= 0.70502:
                                                if RR2_RR3 <= 1.55266:
                                                    return 'N'
                                                else:  # if RR2_RR3 > 1.55266
                                                    if RR2_RR <= 1.20306:
                                                        return 'N'
                                                    else:  # if RR2_RR > 1.20306
                                                        return 'S'
                                            else:  # if RR1_RR2 > 0.70502
                                                if RR1_RR <= 0.66271:
                                                    return 'S'
                                                else:  # if RR1_RR > 0.66271
                                                    return 'N'
                                    else:  # if RR1_RR > 0.68956
                                        return 'N'
                                else:  # if RR1_RR > 0.94511
                                    if RR2_RR <= 1.0007:
                                        if RR3_rr <= 0.7203:
                                            return 'S'
                                        else:  # if RR3_rr > 0.7203
                                            if RR2_RR3 <= 1.2107:
                                                return 'N'
                                            else:  # if RR2_RR3 > 1.2107
                                                if RR1_RR <= 1.25616:
                                                    if RR2_rr <= 1.03284:
                                                        return 'S'
                                                    else:  # if RR2_rr > 1.03284
                                                        if RR1_RR <= 0.95502:
                                                            if RR3_RR <= 0.68011:
                                                                return 'N'
                                                            else:  # if RR3_RR > 0.68011
                                                                return 'S'
                                                        else:  # if RR1_RR > 0.95502
                                                            if RR3_rr <= 0.73349:
                                                                if RR3_RR <= 0.64617:
                                                                    return 'N'
                                                                else:  # if RR3_RR > 0.64617
                                                                    return 'S'
                                                            else:  # if RR3_rr > 0.73349
                                                                return 'N'
                                                else:  # if RR1_RR > 1.25616
                                                    return 'N'
                                    else:  # if RR2_RR > 1.0007
                                        return 'N'
    else:  # if RR3_RR > 0.80689
        if RR2_RR3 <= 1.1088:
            if RR2_RR <= 0.79558:
                if RR2_RR <= 0.79535:
                    if RR3_RR <= 0.92065:
                        if RR2_RR <= 0.73156:
                            if RR3_RR <= 0.92028:
                                if RR2_RR3 <= 0.74138:
                                    if RR3_rr <= 0.85105:
                                        if RR3_rr <= 0.84644:
                                            return 'N'
                                        else:  # if RR3_rr > 0.84644
                                            return 'S'
                                    else:  # if RR3_rr > 0.85105
                                        return 'N'
                                else:  # if RR2_RR3 > 0.74138
                                    if RR2_RR3 <= 0.74275:
                                        return 'S'
                                    else:  # if RR2_RR3 > 0.74275
                                        if RR1_RR <= 0.62878:
                                            if RR1_RR <= 0.59949:
                                                return 'N'
                                            else:  # if RR1_RR > 0.59949
                                                if RR1_RR <= 0.60568:
                                                    return 'S'
                                                else:  # if RR1_RR > 0.60568
                                                    if RR1_RR <= 0.61424:
                                                        return 'N'
                                                    else:  # if RR1_RR > 0.61424
                                                        return 'S'
                                        else:  # if RR1_RR > 0.62878
                                            if RR1_RR <= 1.04611:
                                                return 'N'
                                            else:  # if RR1_RR > 1.04611
                                                if RR2_RR <= 0.72789:
                                                    return 'N'
                                                else:  # if RR2_RR > 0.72789
                                                    return 'S'
                            else:  # if RR3_RR > 0.92028
                                return 'S'
                        else:  # if RR2_RR > 0.73156
                            if RR1_RR <= 0.91148:
                                return 'N'
                            else:  # if RR1_RR > 0.91148
                                if RR1_RR2 <= 1.18615:
                                    return 'S'
                                else:  # if RR1_RR2 > 1.18615
                                    return 'N'
                    else:  # if RR3_RR > 0.92065
                        return 'N'
                else:  # if RR2_RR > 0.79535
                    if RR3_rr <= 1.2431:
                        return 'S'
                    else:  # if RR3_rr > 1.2431
                        return 'N'
            else:  # if RR2_RR > 0.79558
                if RR3_rr <= 0.94643:
                    if RR2_RR <= 1.00455:
                        if RR2_rr <= 0.75383:
                            if RR1_RR <= 1.0736:
                                return 'N'
                            else:  # if RR1_RR > 1.0736
                                if RR1_RR <= 1.08006:
                                    return 'S'
                                else:  # if RR1_RR > 1.08006
                                    return 'N'
                        else:  # if RR2_rr > 0.75383
                            return 'N'
                    else:  # if RR2_RR > 1.00455
                        if RR3_RR <= 0.94562:
                            if RR3_RR <= 0.94532:
                                if RR3_rr <= 0.81094:
                                    if RR1_RR <= 1.14897:
                                        return 'S'
                                    else:  # if RR1_RR > 1.14897
                                        return 'N'
                                else:  # if RR3_rr > 0.81094
                                    if RR1_RR2 <= 0.98683:
                                        if RR2_rr <= 0.96887:
                                            if RR1_RR2 <= 0.97335:
                                                return 'N'
                                            else:  # if RR1_RR2 > 0.97335
                                                return 'S'
                                        else:  # if RR2_rr > 0.96887
                                            if RR3_rr <= 0.94354:
                                                return 'N'
                                            else:  # if RR3_rr > 0.94354
                                                if RR1_RR2 <= 0.95191:
                                                    return 'N'
                                                else:  # if RR1_RR2 > 0.95191
                                                    return 'S'
                                    else:  # if RR1_RR2 > 0.98683
                                        return 'N'
                            else:  # if RR3_RR > 0.94532
                                if RR1_RR <= 0.96762:
                                    return 'S'
                                else:  # if RR1_RR > 0.96762
                                    return 'N'
                        else:  # if RR3_RR > 0.94562
                            if RR2_rr <= 0.97142:
                                return 'N'
                            else:  # if RR2_rr > 0.97142
                                if RR3_rr <= 0.89737:
                                    if RR2_RR3 <= 1.08552:
                                        return 'S'
                                    else:  # if RR2_RR3 > 1.08552
                                        return 'N'
                                else:  # if RR3_rr > 0.89737
                                    if RR1_RR2 <= 1.03081:
                                        return 'N'
                                    else:  # if RR1_RR2 > 1.03081
                                        if RR1_RR <= 1.11007:
                                            return 'N'
                                        else:  # if RR1_RR > 1.11007
                                            return 'N'
                else:  # if RR3_rr > 0.94643
                    return 'N'
        else:  # if RR2_RR3 > 1.1088
            if RR1_RR2 <= 0.9626:
                if RR3_rr <= 0.8544:
                    if RR2_rr <= 1.6733:
                        if RR1_rr <= 1.05948:
                            if RR1_RR <= 0.55375:
                                if RR1_rr <= 0.51826:
                                    return 'N'
                                else:  # if RR1_rr > 0.51826
                                    return 'S'
                            else:  # if RR1_RR > 0.55375
                                if RR1_RR <= 0.68857:
                                    if RR3_RR <= 0.82707:
                                        if RR1_RR <= 0.67735:
                                            return 'N'
                                        else:  # if RR1_RR > 0.67735
                                            return 'S'
                                    else:  # if RR3_RR > 0.82707
                                        return 'N'
                                else:  # if RR1_RR > 0.68857
                                    if RR1_rr <= 0.78604:
                                        if RR1_RR <= 1.13898:
                                            return 'N'
                                        else:  # if RR1_RR > 1.13898
                                            if RR1_rr <= 0.78316:
                                                return 'N'
                                            else:  # if RR1_rr > 0.78316
                                                return 'S'
                                    else:  # if RR1_rr > 0.78604
                                        return 'N'
                        else:  # if RR1_rr > 1.05948
                            if RR2_RR3 <= 1.36646:
                                return 'S'
                            else:  # if RR2_RR3 > 1.36646
                                return 'N'
                    else:  # if RR2_rr > 1.6733
                        return 'S'
                else:  # if RR3_rr > 0.8544
                    if RR2_RR3 <= 1.46347:
                        if RR3_rr <= 0.94235:
                            if RR3_RR <= 0.99463:
                                return 'N'
                            else:  # if RR3_RR > 0.99463
                                if RR2_RR <= 1.12994:
                                    if RR1_rr <= 0.89598:
                                        return 'S'
                                    else:  # if RR1_rr > 0.89598
                                        return 'N'
                                else:  # if RR2_RR > 1.12994
                                    if RR2_RR <= 1.19478:
                                        if RR3_rr <= 0.92141:
                                            if RR1_rr <= 0.94605:
                                                return 'N'
                                            else:  # if RR1_rr > 0.94605
                                                if RR1_RR <= 1.10671:
                                                    return 'S'
                                                else:  # if RR1_RR > 1.10671
                                                    return 'N'
                                        else:  # if RR3_rr > 0.92141
                                            return 'N'
                                    else:  # if RR2_RR > 1.19478
                                        return 'N'
                        else:  # if RR3_rr > 0.94235
                            return 'N'
                    else:  # if RR2_RR3 > 1.46347
                        if RR2_RR3 <= 1.46365:
                            return 'S'
                        else:  # if RR2_RR3 > 1.46365
                            if RR2_RR <= 1.30486:
                                if RR1_rr <= 0.65997:
                                    if RR3_RR <= 0.88656:
                                        return 'N'
                                    else:  # if RR3_RR > 0.88656
                                        return 'S'
                                else:  # if RR1_rr > 0.65997
                                    return 'N'
                            else:  # if RR2_RR > 1.30486
                                return 'N'
            else:  # if RR1_RR2 > 0.9626
                if RR1_rr <= 1.02011:
                    if RR1_RR <= 1.23728:
                        if RR1_RR <= 0.91746:
                            return 'S'
                        else:  # if RR1_RR > 0.91746
                            if RR2_RR3 <= 1.1183:
                                return 'N'
                            else:  # if RR2_RR3 > 1.1183
                                if RR2_rr <= 0.94337:
                                    if RR3_rr <= 0.83676:
                                        if RR1_RR <= 0.99111:
                                            if RR1_rr <= 0.88973:
                                                return 'N'
                                            else:  # if RR1_rr > 0.88973
                                                if RR1_RR2 <= 1.04673:
                                                    return 'S'
                                                else:  # if RR1_RR2 > 1.04673
                                                    return 'N'
                                        else:  # if RR1_RR > 0.99111
                                            if RR1_rr <= 0.85408:
                                                return 'S'
                                            else:  # if RR1_rr > 0.85408
                                                if RR2_RR <= 1.1362:
                                                    if RR2_rr <= 0.84585:
                                                        if RR1_RR2 <= 1.11768:
                                                            return 'N'
                                                        else:  # if RR1_RR2 > 1.11768
                                                            if RR3_rr <= 0.71939:
                                                                return 'S'
                                                            else:  # if RR3_rr > 0.71939
                                                                return 'N'
                                                    else:  # if RR2_rr > 0.84585
                                                        return 'N'
                                                else:  # if RR2_RR > 1.1362
                                                    if RR2_RR3 <= 1.16037:
                                                        if RR1_RR <= 1.14771:
                                                            return 'S'
                                                        else:  # if RR1_RR > 1.14771
                                                            return 'N'
                                                    else:  # if RR2_RR3 > 1.16037
                                                        return 'N'
                                    else:  # if RR3_rr > 0.83676
                                        return 'S'
                                else:  # if RR2_rr > 0.94337
                                    if RR2_RR3 <= 1.11858:
                                        return 'S'
                                    else:  # if RR2_RR3 > 1.11858
                                        if RR2_RR <= 0.99581:
                                            return 'N'
                                        else:  # if RR2_RR > 0.99581
                                            if RR2_RR <= 0.99808:
                                                if RR2_rr <= 0.98914:
                                                    return 'N'
                                                else:  # if RR2_rr > 0.98914
                                                    return 'S'
                                            else:  # if RR2_RR > 0.99808
                                                if RR3_RR <= 0.91585:
                                                    return 'N'
                                                else:  # if RR3_RR > 0.91585
                                                    if RR1_RR2 <= 1.0271:
                                                        if RR1_RR <= 1.01125:
                                                            return 'S'
                                                        else:  # if RR1_RR > 1.01125
                                                            return 'N'
                                                    else:  # if RR1_RR2 > 1.0271
                                                        if RR2_rr <= 0.95749:
                                                            return 'N'
                                                        else:  # if RR2_rr > 0.95749
                                                            if RR2_RR <= 1.09106:
                                                                return 'S'
                                                            else:  # if RR2_RR > 1.09106
                                                                if RR2_rr <= 0.98:
                                                                    return 'N'
                                                                else:  # if RR2_rr > 0.98
                                                                    return 'S'
                    else:  # if RR1_RR > 1.23728
                        if RR2_rr <= 0.86946:
                            return 'N'
                        else:  # if RR2_rr > 0.86946
                            if RR3_RR <= 0.90728:
                                return 'N'
                            else:  # if RR3_RR > 0.90728
                                return 'S'
                else:  # if RR1_rr > 1.02011
                    if RR2_RR3 <= 1.10887:
                        if RR2_RR <= 1.02989:
                            return 'S'
                        else:  # if RR2_RR > 1.02989
                            return 'N'
                    else:  # if RR2_RR3 > 1.10887
                        if RR3_rr <= 0.59828:
                            if RR2_RR3 <= 1.80423:
                                if RR1_RR2 <= 1.20522:
                                    return 'S'
                                else:  # if RR1_RR2 > 1.20522
                                    return 'N'
                            else:  # if RR2_RR3 > 1.80423
                                return 'N'
                        else:  # if RR3_rr > 0.59828
                            if RR1_rr <= 1.10791:
                                if RR2_RR3 <= 1.17589:
                                    if RR1_RR <= 0.96147:
                                        if RR3_rr <= 0.91337:
                                            return 'S'
                                        else:  # if RR3_rr > 0.91337
                                            return 'N'
                                    else:  # if RR1_RR > 0.96147
                                        if RR2_rr <= 1.0823:
                                            if RR1_RR <= 1.02214:
                                                if RR1_rr <= 1.02641:
                                                    if RR1_RR <= 1.01441:
                                                        return 'N'
                                                    else:  # if RR1_RR > 1.01441
                                                        return 'S'
                                                else:  # if RR1_rr > 1.02641
                                                    return 'N'
                                            else:  # if RR1_RR > 1.02214
                                                return 'N'
                                        else:  # if RR2_rr > 1.0823
                                            if RR1_RR2 <= 1.01614:
                                                return 'N'
                                            else:  # if RR1_RR2 > 1.01614
                                                return 'S'
                                else:  # if RR2_RR3 > 1.17589
                                    if RR2_rr <= 1.12358:
                                        if RR2_RR3 <= 1.17629:
                                            return 'S'
                                        else:  # if RR2_RR3 > 1.17629
                                            if RR1_rr <= 1.06656:
                                                if RR2_RR3 <= 1.38087:
                                                    if RR2_RR3 <= 1.20516:
                                                        if RR3_rr <= 0.86011:
                                                            return 'N'
                                                        else:  # if RR3_rr > 0.86011
                                                            if RR1_RR2 <= 0.9714:
                                                                return 'N'
                                                            else:  # if RR1_RR2 > 0.9714
                                                                if RR3_RR <= 0.92739:
                                                                    return 'S'
                                                                else:  # if RR3_RR > 0.92739
                                                                    return 'N'
                                                    else:  # if RR2_RR3 > 1.20516
                                                        return 'N'
                                                else:  # if RR2_RR3 > 1.38087
                                                    if RR3_rr <= 0.70728:
                                                        return 'N'
                                                    else:  # if RR3_rr > 0.70728
                                                        return 'S'
                                            else:  # if RR1_rr > 1.06656
                                                return 'N'
                                    else:  # if RR2_rr > 1.12358
                                        return 'S'
                            else:  # if RR1_rr > 1.10791
                                if RR3_RR <= 0.93945:
                                    if RR2_rr <= 1.45139:
                                        if RR3_rr <= 0.85787:
                                            if RR3_RR <= 0.93849:
                                                return 'N'
                                            else:  # if RR3_RR > 0.93849
                                                return 'S'
                                        else:  # if RR3_rr > 0.85787
                                            return 'N'
                                    else:  # if RR2_rr > 1.45139
                                        if RR3_rr <= 1.05628:
                                            return 'N'
                                        else:  # if RR3_rr > 1.05628
                                            return 'S'
                                else:  # if RR3_RR > 0.93945
                                    return 'N'


# Abnormal ==> using WIDTH + AMP to classify SVes, Ves
def abnormal_W_AMP_2_predict(w_20_cali, w_50_cali, w_80_cali, w_20_local, w_50_local, w_80_local,
                             min_amp_cali, max_amp_cali, peak_peak_cali,
                             min_amp_local, max_amp_local, peak_peak_local):
    if w_50_local <= 1.22793:
        if peak_peak_cali <= 0.81197:
            if min_amp_local <= 1.10317:
                if w_20_cali <= 1.50861:
                    if w_50_cali <= 0.88848:
                        if min_amp_cali <= 0.41704:
                            return 'S'
                        else:  # if min_amp_cali_min_amp > 0.41704
                            if min_amp_local <= 0.84495:
                                if w_50_cali <= 0.67799:
                                    return 'V'
                                else:  # if w_50_cali_w50 > 0.67799
                                    if min_amp_local <= 0.76669:
                                        return 'V'
                                    else:  # if min_amp_local_min_amp > 0.76669
                                        if max_amp_cali <= 0.62879:
                                            return 'V'
                                        else:  # if max_amp_cali_max_amp > 0.62879
                                            return 'S'
                            else:  # if min_amp_local_min_amp > 0.84495
                                return 'V'
                    else:  # if w_50_cali_w50 > 0.88848
                        if w_80_cali <= 1.24194:
                            if w_50_cali <= 0.97014:
                                if min_amp_cali <= 1.09588:
                                    return 'S'
                                else:  # if min_amp_cali_min_amp > 1.09588
                                    return 'V'
                            else:  # if w_50_cali_w50 > 0.97014
                                if max_amp_local <= 1.44435:
                                    if max_amp_local <= 0.58922:
                                        if w_80_cali <= 1.06698:
                                            return 'V'
                                        else:  # if w_80_cali_w80 > 1.06698
                                            return 'S'
                                    else:  # if max_amp_local_max_amp > 0.58922
                                        return 'S'
                                else:  # if max_amp_local_max_amp > 1.44435
                                    return 'V'
                        else:  # if w_80_cali_w80 > 1.24194
                            if w_80_local <= 0.97134:
                                return 'V'
                            else:  # if w_80_local_w80 > 0.97134
                                if w_50_cali <= 1.25926:
                                    return 'V'
                                else:  # if w_50_cali_w50 > 1.25926
                                    return 'S'
                else:  # if w_20_cali_w20 > 1.50861
                    if w_50_cali <= 1.30625:
                        return 'S'
                    else:  # if w_50_cali_w50 > 1.30625
                        return 'V'
            else:  # if min_amp_local_min_amp > 1.10317
                if max_amp_local <= 0.79582:
                    return 'V'
                else:  # if max_amp_local_max_amp > 0.79582
                    if w_20_cali <= 1.08511:
                        if peak_peak_local <= 1.00399:
                            return 'S'
                        else:  # if peak_peak_local_peak_peak > 1.00399
                            return 'V'
                    else:  # if w_20_cali_w20 > 1.08511
                        return 'V'
        else:  # if peak_peak_cali_peak_peak > 0.81197
            if min_amp_cali <= 1.5571:
                if w_80_local <= 1.15124:
                    if w_80_cali <= 0.72658:
                        if w_80_local <= 0.60451:
                            return 'V'
                        else:  # if w_80_local_w80 > 0.60451
                            if max_amp_local <= 0.94749:
                                if w_80_cali <= 0.70254:
                                    return 'V'
                                else:  # if w_80_cali_w80 > 0.70254
                                    return 'S'
                            else:  # if max_amp_local_max_amp > 0.94749
                                if max_amp_local <= 1.22544:
                                    if w_80_local <= 0.86946:
                                        if w_50_cali <= 0.84478:
                                            return 'S'
                                        else:  # if w_50_cali_w50 > 0.84478
                                            return 'V'
                                    else:  # if w_80_local_w80 > 0.86946
                                        if max_amp_local <= 0.97191:
                                            return 'S'
                                        else:  # if max_amp_local_max_amp > 0.97191
                                            return 'V'
                                else:  # if max_amp_local_max_amp > 1.22544
                                    if peak_peak_cali <= 0.93178:
                                        return 'S'
                                    else:  # if peak_peak_cali_peak_peak > 0.93178
                                        return 'V'
                    else:  # if w_80_cali_w80 > 0.72658
                        if w_50_cali <= 1.32704:
                            if max_amp_cali <= 2.30845:
                                if peak_peak_local <= 1.33662:
                                    if min_amp_cali <= 0.57752:
                                        if w_80_cali <= 0.8253:
                                            return 'V'
                                        else:  # if w_80_cali_w80 > 0.8253
                                            if max_amp_cali <= 1.69521:
                                                return 'S'
                                            else:  # if max_amp_cali_max_amp > 1.69521
                                                return 'V'
                                    else:  # if min_amp_cali_min_amp > 0.57752
                                        if max_amp_cali <= 0.82243:
                                            if max_amp_cali <= 0.81729:
                                                if w_50_cali <= 0.88899:
                                                    if w_50_local <= 0.90244:
                                                        return 'S'
                                                    else:  # if w_50_local_w50 > 0.90244
                                                        return 'V'
                                                else:  # if w_50_cali_w50 > 0.88899
                                                    return 'S'
                                            else:  # if max_amp_cali_max_amp > 0.81729
                                                return 'V'
                                        else:  # if max_amp_cali_max_amp > 0.82243
                                            if min_amp_local <= 0.3994:
                                                return 'V'
                                            else:  # if min_amp_local_min_amp > 0.3994
                                                if max_amp_local <= 1.48313:
                                                    return 'S'
                                                else:  # if max_amp_local_max_amp > 1.48313
                                                    return 'V'
                                else:  # if peak_peak_local_peak_peak > 1.33662
                                    if w_50_cali <= 1.18669:
                                        if w_50_cali <= 0.9992:
                                            return 'S'
                                        else:  # if w_50_cali_w50 > 0.9992
                                            return 'V'
                                    else:  # if w_50_cali_w50 > 1.18669
                                        return 'S'
                            else:  # if max_amp_cali_max_amp > 2.30845
                                return 'V'
                        else:  # if w_50_cali_w50 > 1.32704
                            return 'V'
                else:  # if w_80_local_w80 > 1.15124
                    if w_80_cali <= 1.6996:
                        if max_amp_local <= 1.37047:
                            if max_amp_local <= 0.51539:
                                return 'V'
                            else:  # if max_amp_local_max_amp > 0.51539
                                if peak_peak_local <= 1.18154:
                                    if min_amp_cali <= 1.38495:
                                        if min_amp_local <= 0.53191:
                                            return 'V'
                                        else:  # if min_amp_local_min_amp > 0.53191
                                            return 'S'
                                    else:  # if min_amp_cali_min_amp > 1.38495
                                        return 'V'
                                else:  # if peak_peak_local_peak_peak > 1.18154
                                    if w_20_local <= 1.27111:
                                        if w_20_cali <= 0.98889:
                                            return 'V'
                                        else:  # if w_20_cali_w20 > 0.98889
                                            if w_50_cali <= 1.25828:
                                                return 'V'
                                            else:  # if w_50_cali_w50 > 1.25828
                                                return 'S'
                                    else:  # if w_20_local_w20 > 1.27111
                                        if w_50_cali <= 1.32704:
                                            return 'S'
                                        else:  # if w_50_cali_w50 > 1.32704
                                            return 'V'
                        else:  # if max_amp_local_max_amp > 1.37047
                            if w_50_cali <= 1.21512:
                                if max_amp_cali <= 1.34875:
                                    return 'S'
                                else:  # if max_amp_cali_max_amp > 1.34875
                                    return 'V'
                            else:  # if w_50_cali_w50 > 1.21512
                                if peak_peak_cali <= 1.19248:
                                    return 'V'
                                else:  # if peak_peak_cali_peak_peak > 1.19248
                                    if w_50_cali <= 1.37287:
                                        return 'S'
                                    else:  # if w_50_cali_w50 > 1.37287
                                        return 'V'
                    else:  # if w_80_cali_w80 > 1.6996
                        if w_20_cali <= 1.50861:
                            return 'S'
                        else:  # if w_20_cali_w20 > 1.50861
                            return 'V'
            else:  # if min_amp_cali_min_amp > 1.5571
                if max_amp_local <= 0.82967:
                    if peak_peak_local <= 0.69804:
                        return 'S'
                    else:  # if peak_peak_local_peak_peak > 0.69804
                        if w_20_cali <= 0.5473:
                            return 'S'
                        else:  # if w_20_cali_w20 > 0.5473
                            if w_50_local <= 0.53486:
                                return 'S'
                            else:  # if w_50_local_w50 > 0.53486
                                if min_amp_local <= 1.25972:
                                    if max_amp_cali <= 0.81808:
                                        return 'V'
                                    else:  # if max_amp_cali_max_amp > 0.81808
                                        return 'S'
                                else:  # if min_amp_local_min_amp > 1.25972
                                    return 'V'
                else:  # if max_amp_local_max_amp > 0.82967
                    if peak_peak_local <= 1.11585:
                        if w_80_cali <= 1.44786:
                            return 'S'
                        else:  # if w_80_cali_w80 > 1.44786
                            return 'V'
                    else:  # if peak_peak_local_peak_peak > 1.11585
                        if max_amp_cali <= 1.20872:
                            if w_50_local <= 0.73342:
                                if peak_peak_local <= 1.23414:
                                    if w_50_local <= 0.62149:
                                        return 'V'
                                    else:  # if w_50_local_w50 > 0.62149
                                        return 'S'
                                else:  # if peak_peak_local_peak_peak > 1.23414
                                    return 'V'
                            else:  # if w_50_local_w50 > 0.73342
                                if w_20_local <= 0.54445:
                                    return 'S'
                                else:  # if w_20_local_w20 > 0.54445
                                    return 'V'
                        else:  # if max_amp_cali_max_amp > 1.20872
                            if max_amp_cali <= 2.85904:
                                if peak_peak_local <= 1.33251:
                                    if peak_peak_cali <= 1.42498:
                                        return 'V'
                                    else:  # if peak_peak_cali_peak_peak > 1.42498
                                        return 'S'
                                else:  # if peak_peak_local_peak_peak > 1.33251
                                    if w_50_local <= 1.19783:
                                        if max_amp_local <= 1.42577:
                                            if w_20_local <= 0.9366:
                                                return 'S'
                                            else:  # if w_20_local_w20 > 0.9366
                                                return 'V'
                                        else:  # if max_amp_local_max_amp > 1.42577
                                            if w_80_cali <= 0.95584:
                                                return 'V'
                                            else:  # if w_80_cali_w80 > 0.95584
                                                if min_amp_local <= 1.40832:
                                                    return 'V'
                                                else:  # if min_amp_local_min_amp > 1.40832
                                                    return 'S'
                                    else:  # if w_50_local_w50 > 1.19783
                                        return 'V'
                            else:  # if max_amp_cali_max_amp > 2.85904
                                return 'V'
    else:  # if w_50_local_w50 > 1.22793
        if min_amp_cali <= 1.47594:
            if w_50_local <= 1.43763:
                if w_80_cali <= 1.70271:
                    if w_50_cali <= 1.31176:
                        if w_50_cali <= 1.27172:
                            if max_amp_cali <= 1.00182:
                                return 'V'
                            else:  # if max_amp_cali_max_amp > 1.00182
                                if min_amp_cali <= 0.92629:
                                    return 'V'
                                else:  # if min_amp_cali_min_amp > 0.92629
                                    if min_amp_local <= 1.25078:
                                        if w_50_local <= 1.23769:
                                            return 'V'
                                        else:  # if w_50_local_w50 > 1.23769
                                            return 'S'
                                    else:  # if min_amp_local_min_amp > 1.25078
                                        return 'V'
                        else:  # if w_50_cali_w50 > 1.27172
                            return 'S'
                    else:  # if w_50_cali_w50 > 1.31176
                        if peak_peak_cali <= 1.32749:
                            if w_20_cali <= 1.50489:
                                if w_20_cali <= 1.48334:
                                    if w_50_local <= 1.39833:
                                        if w_20_cali <= 1.13511:
                                            if peak_peak_cali <= 0.67101:
                                                if min_amp_cali <= 0.82103:
                                                    return 'V'
                                                else:  # if min_amp_cali_min_amp > 0.82103
                                                    return 'S'
                                            else:  # if peak_peak_cali_peak_peak > 0.67101
                                                return 'V'
                                        else:  # if w_20_cali_w20 > 1.13511
                                            return 'V'
                                    else:  # if w_50_local_w50 > 1.39833
                                        if w_20_local <= 1.41:
                                            return 'S'
                                        else:  # if w_20_local_w20 > 1.41
                                            return 'V'
                                else:  # if w_20_cali_w20 > 1.48334
                                    if w_50_local <= 1.41739:
                                        return 'S'
                                    else:  # if w_50_local_w50 > 1.41739
                                        return 'V'
                            else:  # if w_20_cali_w20 > 1.50489
                                return 'V'
                        else:  # if peak_peak_cali_peak_peak > 1.32749
                            if max_amp_cali <= 2.10737:
                                if w_20_cali <= 1.57499:
                                    return 'S'
                                else:  # if w_20_cali_w20 > 1.57499
                                    return 'V'
                            else:  # if max_amp_cali_max_amp > 2.10737
                                return 'V'
                else:  # if w_80_cali_w80 > 1.70271
                    if w_50_local <= 1.42473:
                        return 'V'
                    else:  # if w_50_local_w50 > 1.42473
                        if min_amp_local <= 1.1309:
                            return 'V'
                        else:  # if min_amp_local_min_amp > 1.1309
                            return 'S'
            else:  # if w_50_local_w50 > 1.43763
                if w_20_cali <= 1.50489:
                    if w_50_cali <= 1.54088:
                        if w_50_cali <= 1.52369:
                            if w_80_local <= 1.11897:
                                return 'S'
                            else:  # if w_80_local_w80 > 1.11897
                                return 'V'
                        else:  # if w_50_cali_w50 > 1.52369
                            return 'S'
                    else:  # if w_50_cali_w50 > 1.54088
                        return 'V'
                else:  # if w_20_cali_w20 > 1.50489
                    return 'V'
        else:  # if min_amp_cali_min_amp > 1.47594
            return 'V'


# Abnormal (Normal-SVes) ==> using RR to classify Normal, SVes
def abnormal_RR_NS_2_predict(RR1_RR2, RR2_RR3, RR1_RR, RR2_RR, RR3_RR, RR1_rr, RR2_rr, RR3_rr):
    if RR3_RR <= 0.89119:
        if RR2_RR <= 0.68294:
            if RR1_rr <= 0.90869:
                if RR3_rr <= 0.58869:
                    if RR2_rr <= 0.62677:
                        if RR1_RR <= 0.66132:
                            if RR3_RR <= 0.4987:
                                return 'N'
                            else:  # if RR3_RR > 0.4987
                                return 'S'
                        else:  # if RR1_RR > 0.66132
                            if RR1_RR2 <= 1.6499:
                                if RR1_rr <= 0.51084:
                                    return 'S'
                                else:  # if RR1_rr > 0.51084
                                    if RR3_rr <= 0.58209:
                                        return 'N'
                                    else:  # if RR3_rr > 0.58209
                                        if RR1_rr <= 0.74025:
                                            return 'S'
                                        else:  # if RR1_rr > 0.74025
                                            return 'N'
                            else:  # if RR1_RR2 > 1.6499
                                return 'S'
                    else:  # if RR2_rr > 0.62677
                        return 'N'
                else:  # if RR3_rr > 0.58869
                    if RR1_RR2 <= 0.87268:
                        return 'N'
                    else:  # if RR1_RR2 > 0.87268
                        if RR3_RR <= 0.56657:
                            return 'N'
                        else:  # if RR3_RR > 0.56657
                            if RR1_RR <= 0.77082:
                                if RR3_RR <= 0.72009:
                                    if RR1_RR <= 0.47381:
                                        return 'N'
                                    else:  # if RR1_RR > 0.47381
                                        return 'S'
                                else:  # if RR3_RR > 0.72009
                                    return 'N'
                            else:  # if RR1_RR > 0.77082
                                return 'N'
            else:  # if RR1_rr > 0.90869
                if RR2_RR3 <= 1.07217:
                    if RR2_rr <= 0.95731:
                        if RR1_rr <= 1.43494:
                            if RR3_rr <= 1.01099:
                                if RR1_rr <= 0.91091:
                                    return 'N'
                                else:  # if RR1_rr > 0.91091
                                    if RR2_RR3 <= 0.89089:
                                        if RR2_RR3 <= 0.6975:
                                            return 'S'
                                        else:  # if RR2_RR3 > 0.6975
                                            if RR2_rr <= 0.58877:
                                                if RR2_RR3 <= 0.86488:
                                                    return 'S'
                                                else:  # if RR2_RR3 > 0.86488
                                                    return 'N'
                                            else:  # if RR2_rr > 0.58877
                                                return 'N'
                                    else:  # if RR2_RR3 > 0.89089
                                        if RR3_RR <= 0.52239:
                                            return 'N'
                                        else:  # if RR3_RR > 0.52239
                                            if RR2_rr <= 0.65299:
                                                return 'S'
                                            else:  # if RR2_rr > 0.65299
                                                if RR3_rr <= 0.90157:
                                                    if RR1_RR <= 0.96489:
                                                        if RR2_rr <= 0.84873:
                                                            if RR2_RR <= 0.61237:
                                                                return 'N'
                                                            else:  # if RR2_RR > 0.61237
                                                                return 'S'
                                                        else:  # if RR2_rr > 0.84873
                                                            return 'S'
                                                    else:  # if RR1_RR > 0.96489
                                                        return 'S'
                                                else:  # if RR3_rr > 0.90157
                                                    return 'S'
                            else:  # if RR3_rr > 1.01099
                                return 'N'
                        else:  # if RR1_rr > 1.43494
                            if RR2_RR3 <= 1.01394:
                                return 'S'
                            else:  # if RR2_RR3 > 1.01394
                                return 'N'
                    else:  # if RR2_rr > 0.95731
                        return 'S'
                else:  # if RR2_RR3 > 1.07217
                    if RR2_rr <= 0.76534:
                        if RR1_RR <= 0.87646:
                            return 'N'
                        else:  # if RR1_RR > 0.87646
                            if RR3_rr <= 0.62769:
                                if RR2_RR3 <= 1.22385:
                                    if RR2_rr <= 0.54256:
                                        return 'N'
                                    else:  # if RR2_rr > 0.54256
                                        if RR1_RR <= 1.11247:
                                            if RR3_rr <= 0.51461:
                                                if RR2_RR <= 0.48465:
                                                    return 'N'
                                                else:  # if RR2_RR > 0.48465
                                                    return 'S'
                                            else:  # if RR3_rr > 0.51461
                                                return 'S'
                                        else:  # if RR1_RR > 1.11247
                                            if RR1_RR2 <= 2.48157:
                                                return 'N'
                                            else:  # if RR1_RR2 > 2.48157
                                                return 'S'
                                else:  # if RR2_RR3 > 1.22385
                                    return 'S'
                            else:  # if RR3_rr > 0.62769
                                return 'N'
                    else:  # if RR2_rr > 0.76534
                        if RR1_rr <= 1.05335:
                            if RR3_RR <= 0.46801:
                                return 'N'
                            else:  # if RR3_RR > 0.46801
                                return 'S'
                        else:  # if RR1_rr > 1.05335
                            if RR1_RR2 <= 1.85892:
                                return 'N'
                            else:  # if RR1_RR2 > 1.85892
                                if RR1_rr <= 1.66966:
                                    return 'S'
                                else:  # if RR1_rr > 1.66966
                                    return 'N'
        else:  # if RR2_RR > 0.68294
            if RR2_rr <= 0.93358:
                if RR1_RR2 <= 1.0182:
                    if RR2_RR <= 0.73933:
                        if RR2_RR <= 0.68473:
                            return 'N'
                        else:  # if RR2_RR > 0.68473
                            return 'S'
                    else:  # if RR2_RR > 0.73933
                        if RR1_rr <= 0.90749:
                            if RR2_RR <= 0.89063:
                                return 'N'
                            else:  # if RR2_RR > 0.89063
                                if RR3_rr <= 0.71014:
                                    return 'N'
                                else:  # if RR3_rr > 0.71014
                                    return 'S'
                        else:  # if RR1_rr > 0.90749
                            return 'S'
                else:  # if RR1_RR2 > 1.0182
                    if RR2_RR3 <= 1.15001:
                        if RR1_RR <= 1.57862:
                            if RR3_rr <= 0.50431:
                                return 'S'
                            else:  # if RR3_rr > 0.50431
                                if RR2_RR <= 0.71126:
                                    if RR2_rr <= 0.74991:
                                        return 'N'
                                    else:  # if RR2_rr > 0.74991
                                        if RR1_rr <= 0.95991:
                                            return 'N'
                                        else:  # if RR1_rr > 0.95991
                                            if RR3_RR <= 0.64743:
                                                return 'N'
                                            else:  # if RR3_RR > 0.64743
                                                return 'S'
                                else:  # if RR2_RR > 0.71126
                                    return 'N'
                        else:  # if RR1_RR > 1.57862
                            return 'S'
                    else:  # if RR2_RR3 > 1.15001
                        if RR2_RR <= 0.74966:
                            if RR2_RR <= 0.69254:
                                return 'N'
                            else:  # if RR2_RR > 0.69254
                                return 'S'
                        else:  # if RR2_RR > 0.74966
                            if RR3_rr <= 0.58471:
                                if RR2_RR3 <= 1.4154:
                                    return 'N'
                                else:  # if RR2_RR3 > 1.4154
                                    if RR3_RR <= 0.64511:
                                        if RR1_RR2 <= 1.05357:
                                            return 'N'
                                        else:  # if RR1_RR2 > 1.05357
                                            if RR3_rr <= 0.33238:
                                                return 'N'
                                            else:  # if RR3_rr > 0.33238
                                                return 'S'
                                    else:  # if RR3_RR > 0.64511
                                        return 'N'
                            else:  # if RR3_rr > 0.58471
                                if RR1_rr <= 1.05449:
                                    if RR2_RR3 <= 1.24435:
                                        return 'S'
                                    else:  # if RR2_RR3 > 1.24435
                                        return 'N'
                                else:  # if RR1_rr > 1.05449
                                    return 'N'
            else:  # if RR2_rr > 0.93358
                if RR2_RR3 <= 1.99597:
                    if RR3_RR <= 0.53012:
                        if RR2_rr <= 1.29759:
                            if RR3_rr <= 0.53077:
                                return 'N'
                            else:  # if RR3_rr > 0.53077
                                if RR1_RR <= 1.09874:
                                    return 'N'
                                else:  # if RR1_RR > 1.09874
                                    return 'S'
                        else:  # if RR2_rr > 1.29759
                            return 'N'
                    else:  # if RR3_RR > 0.53012
                        if RR1_RR <= 0.88747:
                            if RR2_RR3 <= 1.35654:
                                if RR3_rr <= 0.87211:
                                    if RR1_RR <= 0.62457:
                                        return 'N'
                                    else:  # if RR1_RR > 0.62457
                                        if RR1_RR <= 0.86879:
                                            if RR3_RR <= 0.61498:
                                                return 'N'
                                            else:  # if RR3_RR > 0.61498
                                                return 'S'
                                        else:  # if RR1_RR > 0.86879
                                            if RR1_RR2 <= 0.96192:
                                                return 'N'
                                            else:  # if RR1_RR2 > 0.96192
                                                if RR2_RR3 <= 1.17516:
                                                    return 'N'
                                                else:  # if RR2_RR3 > 1.17516
                                                    return 'S'
                                else:  # if RR3_rr > 0.87211
                                    if RR1_RR <= 0.69769:
                                        if RR1_RR2 <= 0.87416:
                                            return 'N'
                                        else:  # if RR1_RR2 > 0.87416
                                            if RR2_RR <= 0.68565:
                                                return 'N'
                                            else:  # if RR2_RR > 0.68565
                                                if RR3_RR <= 0.63455:
                                                    return 'N'
                                                else:  # if RR3_RR > 0.63455
                                                    return 'S'
                                    else:  # if RR1_RR > 0.69769
                                        return 'N'
                            else:  # if RR2_RR3 > 1.35654
                                if RR1_rr <= 0.64761:
                                    if RR2_RR <= 1.01861:
                                        return 'N'
                                    else:  # if RR2_RR > 1.01861
                                        if RR2_rr <= 1.48092:
                                            return 'S'
                                        else:  # if RR2_rr > 1.48092
                                            return 'N'
                                else:  # if RR1_rr > 0.64761
                                    if RR2_RR <= 1.27345:
                                        if RR1_rr <= 0.81077:
                                            if RR2_rr <= 1.08963:
                                                return 'N'
                                            else:  # if RR2_rr > 1.08963
                                                if RR2_RR3 <= 1.84523:
                                                    if RR3_rr <= 0.87203:
                                                        return 'S'
                                                    else:  # if RR3_rr > 0.87203
                                                        return 'N'
                                                else:  # if RR2_RR3 > 1.84523
                                                    return 'N'
                                        else:  # if RR1_rr > 0.81077
                                            if RR3_RR <= 0.63911:
                                                if RR2_rr <= 1.34705:
                                                    if RR1_RR2 <= 0.99422:
                                                        if RR1_RR <= 0.80428:
                                                            if RR2_RR <= 1.05933:
                                                                return 'N'
                                                            else:  # if RR2_RR > 1.05933
                                                                return 'S'
                                                        else:  # if RR1_RR > 0.80428
                                                            return 'S'
                                                    else:  # if RR1_RR2 > 0.99422
                                                        return 'N'
                                                else:  # if RR2_rr > 1.34705
                                                    return 'N'
                                            else:  # if RR3_RR > 0.63911
                                                if RR2_RR <= 1.23421:
                                                    if RR3_RR <= 0.74732:
                                                        return 'N'
                                                    else:  # if RR3_RR > 0.74732
                                                        if RR1_RR2 <= 0.71789:
                                                            return 'S'
                                                        else:  # if RR1_RR2 > 0.71789
                                                            return 'N'
                                                else:  # if RR2_RR > 1.23421
                                                    return 'S'
                                    else:  # if RR2_RR > 1.27345
                                        return 'N'
                        else:  # if RR1_RR > 0.88747
                            if RR2_rr <= 1.05634:
                                if RR3_RR <= 0.61005:
                                    if RR2_RR3 <= 1.80164:
                                        if RR3_RR <= 0.59449:
                                            if RR1_rr <= 1.01808:
                                                return 'N'
                                            else:  # if RR1_rr > 1.01808
                                                if RR1_RR2 <= 1.10818:
                                                    return 'S'
                                                else:  # if RR1_RR2 > 1.10818
                                                    return 'N'
                                        else:  # if RR3_RR > 0.59449
                                            return 'N'
                                    else:  # if RR2_RR3 > 1.80164
                                        return 'S'
                                else:  # if RR3_RR > 0.61005
                                    if RR2_RR <= 1.13165:
                                        if RR3_RR <= 0.79316:
                                            if RR1_RR <= 1.15912:
                                                return 'S'
                                            else:  # if RR1_RR > 1.15912
                                                if RR1_rr <= 1.72602:
                                                    if RR3_rr <= 0.66201:
                                                        if RR1_RR <= 1.34176:
                                                            return 'S'
                                                        else:  # if RR1_RR > 1.34176
                                                            return 'N'
                                                    else:  # if RR3_rr > 0.66201
                                                        return 'N'
                                                else:  # if RR1_rr > 1.72602
                                                    return 'S'
                                        else:  # if RR3_RR > 0.79316
                                            if RR1_RR2 <= 1.08837:
                                                if RR1_rr <= 1.0432:
                                                    return 'S'
                                                else:  # if RR1_rr > 1.0432
                                                    if RR2_RR <= 1.05439:
                                                        return 'N'
                                                    else:  # if RR2_RR > 1.05439
                                                        if RR1_RR <= 1.11583:
                                                            return 'N'
                                                        else:  # if RR1_RR > 1.11583
                                                            return 'S'
                                            else:  # if RR1_RR2 > 1.08837
                                                if RR1_RR2 <= 1.35979:
                                                    return 'N'
                                                else:  # if RR1_RR2 > 1.35979
                                                    return 'S'
                                    else:  # if RR2_RR > 1.13165
                                        if RR2_RR3 <= 1.73658:
                                            return 'N'
                                        else:  # if RR2_RR3 > 1.73658
                                            return 'S'
                            else:  # if RR2_rr > 1.05634
                                if RR1_RR <= 2.07125:
                                    if RR2_RR <= 1.24335:
                                        if RR3_RR <= 0.61025:
                                            if RR2_RR <= 0.99021:
                                                return 'N'
                                            else:  # if RR2_RR > 0.99021
                                                return 'S'
                                        else:  # if RR3_RR > 0.61025
                                            if RR1_RR <= 1.07855:
                                                if RR2_rr <= 1.07936:
                                                    if RR1_RR2 <= 0.94619:
                                                        return 'N'
                                                    else:  # if RR1_RR2 > 0.94619
                                                        if RR1_rr <= 1.06688:
                                                            return 'S'
                                                        else:  # if RR1_rr > 1.06688
                                                            return 'N'
                                                else:  # if RR2_rr > 1.07936
                                                    return 'N'
                                            else:  # if RR1_RR > 1.07855
                                                if RR1_rr <= 1.74746:
                                                    if RR3_RR <= 0.6386:
                                                        return 'N'
                                                    else:  # if RR3_RR > 0.6386
                                                        if RR1_RR <= 1.38997:
                                                            if RR2_rr <= 1.10939:
                                                                if RR1_RR <= 1.11528:
                                                                    return 'S'
                                                                else:  # if RR1_RR > 1.11528
                                                                    if RR3_RR <= 0.71461:
                                                                        if RR1_RR <= 1.18403:
                                                                            return 'N'
                                                                        else:  # if RR1_RR > 1.18403
                                                                            return 'S'
                                                                    else:  # if RR3_RR > 0.71461
                                                                        return 'N'
                                                            else:  # if RR2_rr > 1.10939
                                                                return 'S'
                                                        else:  # if RR1_RR > 1.38997
                                                            return 'N'
                                                else:  # if RR1_rr > 1.74746
                                                    return 'N'
                                    else:  # if RR2_RR > 1.24335
                                        if RR2_rr <= 1.41676:
                                            return 'N'
                                        else:  # if RR2_rr > 1.41676
                                            if RR3_RR <= 0.7841:
                                                if RR3_rr <= 0.82857:
                                                    if RR2_RR <= 1.44758:
                                                        return 'S'
                                                    else:  # if RR2_RR > 1.44758
                                                        return 'N'
                                                else:  # if RR3_rr > 0.82857
                                                    return 'N'
                                            else:  # if RR3_RR > 0.7841
                                                return 'N'
                                else:  # if RR1_RR > 2.07125
                                    return 'S'
                else:  # if RR2_RR3 > 1.99597
                    if RR3_RR <= 0.41781:
                        return 'N'
                    else:  # if RR3_RR > 0.41781
                        if RR1_RR <= 0.53311:
                            if RR1_rr <= 0.3851:
                                return 'N'
                            else:  # if RR1_rr > 0.3851
                                if RR1_RR <= 0.53058:
                                    return 'S'
                                else:  # if RR1_RR > 0.53058
                                    return 'N'
                        else:  # if RR1_RR > 0.53311
                            if RR3_RR <= 0.81782:
                                return 'S'
                            else:  # if RR3_RR > 0.81782
                                if RR2_RR3 <= 2.82654:
                                    return 'N'
                                else:  # if RR2_RR3 > 2.82654
                                    return 'S'
    else:  # if RR3_RR > 0.89119
        if RR2_RR3 <= 1.31709:
            return 'N'
        else:  # if RR2_RR3 > 1.31709
            if RR1_RR2 <= 1.19983:
                if RR1_RR2 <= 0.56757:
                    if RR1_RR <= 0.99621:
                        return 'N'
                    else:  # if RR1_RR > 0.99621
                        if RR2_RR <= 2.21207:
                            return 'S'
                        else:  # if RR2_RR > 2.21207
                            return 'N'
                else:  # if RR1_RR2 > 0.56757
                    return 'N'
            else:  # if RR1_RR2 > 1.19983
                return 'S'


# Abnormal (Normal-Ves) ==> using WIDTH + AMP to classify Normal, Ves
def abnormal_RR_NV_2_predict(w_20_cali, w_50_cali, w_80_cali, w_20_local, w_50_local,
                             w_80_local0, min_amp_cali, max_amp_cali, peak_peak_cali,
                             min_amp_local, max_amp_local, peak_peak_local):
    if w_50_local <= 0.79614:
        if w_50_cali <= 0.64609:
            if min_amp_cali <= 0.77692:
                if max_amp_cali <= 0.78864:
                    if w_50_local <= 0.6082:
                        if w_50_local <= 0.54188:
                            if min_amp_local <= 0.83092:
                                return 'N'
                            else:  # if min_amp_local_min_amp > 0.83092
                                return 'V'
                        else:  # if w_50_local_w50 > 0.54188
                            return 'V'
                    else:  # if w_50_local_w50 > 0.6082
                        return 'N'
                else:  # if max_amp_cali_max_amp > 0.78864
                    return 'V'
            else:  # if min_amp_cali_min_amp > 0.77692
                if w_50_local <= 0.54461:
                    if w_50_cali <= 0.55843:
                        if w_20_local <= 0.38941:
                            return 'N'
                        else:  # if w_20_local_w20 > 0.38941
                            if min_amp_local <= 2.14318:
                                return 'V'
                            else:  # if min_amp_local_min_amp > 2.14318
                                return 'N'
                    else:  # if w_50_cali_w50 > 0.55843
                        if peak_peak_cali <= 0.63413:
                            return 'N'
                        else:  # if peak_peak_cali_peak_peak > 0.63413
                            if min_amp_cali <= 0.79684:
                                return 'N'
                            else:  # if min_amp_cali_min_amp > 0.79684
                                if w_20_cali <= 0.47849:
                                    if min_amp_local <= 1.42623:
                                        return 'N'
                                    else:  # if min_amp_local_min_amp > 1.42623
                                        return 'V'
                                else:  # if w_20_cali_w20 > 0.47849
                                    return 'V'
                else:  # if w_50_local_w50 > 0.54461
                    if max_amp_local <= 0.43034:
                        if min_amp_local <= 1.46593:
                            return 'N'
                        else:  # if min_amp_local_min_amp > 1.46593
                            return 'V'
                    else:  # if max_amp_local_max_amp > 0.43034
                        return 'N'
        else:  # if w_50_cali_w50 > 0.64609
            if w_50_cali <= 0.69185:
                return 'V'
            else:  # if w_50_cali_w50 > 0.69185
                if min_amp_local <= 0.79005:
                    if min_amp_cali <= 0.64085:
                        return 'N'
                    else:  # if min_amp_cali_min_amp > 0.64085
                        if min_amp_local <= 0.73719:
                            if peak_peak_local <= 1.15964:
                                if min_amp_local <= 0.3796:
                                    return 'N'
                                else:  # if min_amp_local_min_amp > 0.3796
                                    return 'V'
                            else:  # if peak_peak_local_peak_peak > 1.15964
                                return 'N'
                        else:  # if min_amp_local_min_amp > 0.73719
                            return 'N'
                else:  # if min_amp_local_min_amp > 0.79005
                    if min_amp_cali <= 1.32517:
                        if max_amp_local <= 0.30574:
                            if w_80_local0 <= 1.59477:
                                return 'V'
                            else:  # if w_80_local_w80 > 1.59477
                                return 'N'
                        else:  # if max_amp_local_max_amp > 0.30574
                            if max_amp_cali <= 1.02989:
                                return 'N'
                            else:  # if max_amp_cali_max_amp > 1.02989
                                if min_amp_local <= 1.23699:
                                    if max_amp_local <= 1.00193:
                                        return 'V'
                                    else:  # if max_amp_local_max_amp > 1.00193
                                        return 'N'
                                else:  # if min_amp_local_min_amp > 1.23699
                                    return 'N'
                    else:  # if min_amp_cali_min_amp > 1.32517
                        if peak_peak_local <= 1.85861:
                            if w_20_local <= 0.28127:
                                return 'N'
                            else:  # if w_20_local_w20 > 0.28127
                                if max_amp_local <= 1.24146:
                                    if w_20_cali <= 11.5357:
                                        return 'V'
                                    else:  # if w_20_cali_w20 > 11.5357
                                        return 'N'
                                else:  # if max_amp_local_max_amp > 1.24146
                                    if w_20_cali <= 0.91404:
                                        return 'N'
                                    else:  # if w_20_cali_w20 > 0.91404
                                        return 'V'
                        else:  # if peak_peak_local_peak_peak > 1.85861
                            return 'N'
    else:  # if w_50_local_w50 > 0.79614
        if min_amp_cali <= 0.03679:
            return 'N'
        else:  # if min_amp_cali_min_amp > 0.03679
            if min_amp_local <= 1.05234:
                if w_50_cali <= 1.64257:
                    if max_amp_local <= 1.36647:
                        if w_80_cali <= 1.45163:
                            if w_80_local0 <= 1.19936:
                                if peak_peak_local <= 0.92044:
                                    if max_amp_local <= 0.66597:
                                        if w_20_local <= 0.81494:
                                            if w_50_cali <= 1.11265:
                                                return 'N'
                                            else:  # if w_50_cali_w50 > 1.11265
                                                return 'V'
                                        else:  # if w_20_local_w20 > 0.81494
                                            if w_80_local0 <= 1.16471:
                                                return 'V'
                                            else:  # if w_80_local_w80 > 1.16471
                                                return 'N'
                                    else:  # if max_amp_local_max_amp > 0.66597
                                        if min_amp_cali <= 0.59331:
                                            if min_amp_local <= 0.36131:
                                                return 'N'
                                            else:  # if min_amp_local_min_amp > 0.36131
                                                if min_amp_cali <= 0.39076:
                                                    return 'N'
                                                else:  # if min_amp_cali_min_amp > 0.39076
                                                    if peak_peak_cali <= 0.77002:
                                                        if w_20_local <= 1.14513:
                                                            return 'N'
                                                        else:  # if w_20_local_w20 > 1.14513
                                                            return 'V'
                                                    else:  # if peak_peak_cali_peak_peak > 0.77002
                                                        return 'V'
                                        else:  # if min_amp_cali_min_amp > 0.59331
                                            if w_80_local0 <= 1.00541:
                                                if w_80_cali <= 0.83955:
                                                    return 'N'
                                                else:  # if w_80_cali_w80 > 0.83955
                                                    if min_amp_cali <= 0.8024:
                                                        return 'N'
                                                    else:  # if min_amp_cali_min_amp > 0.8024
                                                        return 'V'
                                            else:  # if w_80_local_w80 > 1.00541
                                                return 'N'
                                else:  # if peak_peak_local_peak_peak > 0.92044
                                    if peak_peak_cali <= 1.16081:
                                        if min_amp_cali <= 1.06165:
                                            if w_20_cali <= 1.64414:
                                                return 'N'
                                            else:  # if w_20_cali_w20 > 1.64414
                                                return 'V'
                                        else:  # if min_amp_cali_min_amp > 1.06165
                                            if max_amp_cali <= 1.09612:
                                                return 'V'
                                            else:  # if max_amp_cali_max_amp > 1.09612
                                                return 'N'
                                    else:  # if peak_peak_cali_peak_peak > 1.16081
                                        if peak_peak_local <= 1.02022:
                                            return 'N'
                                        else:  # if peak_peak_local_peak_peak > 1.02022
                                            if w_20_cali <= 1.43749:
                                                return 'V'
                                            else:  # if w_20_cali_w20 > 1.43749
                                                return 'N'
                            else:  # if w_80_local_w80 > 1.19936
                                if w_80_local0 <= 1.67515:
                                    if w_50_cali <= 1.22986:
                                        if w_50_local <= 1.3261:
                                            return 'N'
                                        else:  # if w_50_local_w50 > 1.3261
                                            return 'V'
                                    else:  # if w_50_cali_w50 > 1.22986
                                        if max_amp_cali <= 0.50448:
                                            return 'N'
                                        else:  # if max_amp_cali_max_amp > 0.50448
                                            if w_20_local <= 1.00207:
                                                if peak_peak_local <= 0.82583:
                                                    return 'V'
                                                else:  # if peak_peak_local_peak_peak > 0.82583
                                                    return 'N'
                                            else:  # if w_20_local_w20 > 1.00207
                                                if peak_peak_local <= 1.21731:
                                                    if max_amp_cali <= 1.4495:
                                                        return 'V'
                                                    else:  # if max_amp_cali_max_amp > 1.4495
                                                        return 'N'
                                                else:  # if peak_peak_local_peak_peak > 1.21731
                                                    return 'N'
                                else:  # if w_80_local_w80 > 1.67515
                                    return 'N'
                        else:  # if w_80_cali_w80 > 1.45163
                            if w_20_cali <= 1.49814:
                                if w_80_local0 <= 2.18863:
                                    return 'N'
                                else:  # if w_80_local_w80 > 2.18863
                                    if w_50_cali <= 1.27311:
                                        return 'N'
                                    else:  # if w_50_cali_w50 > 1.27311
                                        return 'V'
                            else:  # if w_20_cali_w20 > 1.49814
                                if w_20_cali <= 1.60486:
                                    if min_amp_local <= 0.42118:
                                        return 'N'
                                    else:  # if min_amp_local_min_amp > 0.42118
                                        if peak_peak_local <= 0.97639:
                                            if w_50_cali <= 1.3902:
                                                return 'N'
                                            else:  # if w_50_cali_w50 > 1.3902
                                                return 'V'
                                        else:  # if peak_peak_local_peak_peak > 0.97639
                                            return 'N'
                                else:  # if w_20_cali_w20 > 1.60486
                                    if min_amp_cali <= 1.07866:
                                        if w_50_local <= 1.3812:
                                            return 'N'
                                        else:  # if w_50_local_w50 > 1.3812
                                            if w_80_local0 <= 1.47883:
                                                return 'V'
                                            else:  # if w_80_local_w80 > 1.47883
                                                return 'N'
                                    else:  # if min_amp_cali_min_amp > 1.07866
                                        return 'V'
                    else:  # if max_amp_local_max_amp > 1.36647
                        if w_20_cali <= 0.95338:
                            return 'N'
                        else:  # if w_20_cali_w20 > 0.95338
                            if w_20_local <= 0.7178:
                                return 'N'
                            else:  # if w_20_local_w20 > 0.7178
                                if peak_peak_cali <= 1.78183:
                                    if w_50_cali <= 1.22986:
                                        if min_amp_local <= 0.69712:
                                            if w_50_local <= 0.88668:
                                                if w_20_local <= 0.90428:
                                                    return 'V'
                                                else:  # if w_20_local_w20 > 0.90428
                                                    return 'N'
                                            else:  # if w_50_local_w50 > 0.88668
                                                return 'V'
                                        else:  # if min_amp_local_min_amp > 0.69712
                                            if w_50_local <= 1.06163:
                                                if w_20_local <= 1.0911:
                                                    return 'N'
                                                else:  # if w_20_local_w20 > 1.0911
                                                    return 'V'
                                            else:  # if w_50_local_w50 > 1.06163
                                                return 'N'
                                    else:  # if w_50_cali_w50 > 1.22986
                                        return 'V'
                                else:  # if peak_peak_cali_peak_peak > 1.78183
                                    return 'N'
                else:  # if w_50_cali_w50 > 1.64257
                    if w_20_cali <= 1.49325:
                        if max_amp_local <= 0.39944:
                            if peak_peak_cali <= 0.43806:
                                return 'N'
                            else:  # if peak_peak_cali_peak_peak > 0.43806
                                return 'V'
                        else:  # if max_amp_local_max_amp > 0.39944
                            if w_50_cali <= 2.49466:
                                if max_amp_local <= 1.11427:
                                    if min_amp_cali <= 1.20288:
                                        if w_80_cali <= 1.59847:
                                            if max_amp_local <= 0.65756:
                                                return 'V'
                                            else:  # if max_amp_local_max_amp > 0.65756
                                                return 'N'
                                        else:  # if w_80_cali_w80 > 1.59847
                                            return 'V'
                                    else:  # if min_amp_cali_min_amp > 1.20288
                                        return 'N'
                                else:  # if max_amp_local_max_amp > 1.11427
                                    return 'N'
                            else:  # if w_50_cali_w50 > 2.49466
                                return 'N'
                    else:  # if w_20_cali_w20 > 1.49325
                        if w_80_local0 <= 3.46573:
                            if w_20_cali <= 7.09577:
                                if min_amp_cali <= 1.28259:
                                    if max_amp_cali <= 3.93265:
                                        if min_amp_local <= 1.04491:
                                            if max_amp_local <= 0.59651:
                                                if w_50_cali <= 2.48573:
                                                    if max_amp_cali <= 0.59517:
                                                        return 'V'
                                                    else:  # if max_amp_cali_max_amp > 0.59517
                                                        if w_80_local0 <= 2.6699:
                                                            return 'N'
                                                        else:  # if w_80_local_w80 > 2.6699
                                                            return 'V'
                                                else:  # if w_50_cali_w50 > 2.48573
                                                    if min_amp_local <= 0.49688:
                                                        return 'V'
                                                    else:  # if min_amp_local_min_amp > 0.49688
                                                        return 'N'
                                            else:  # if max_amp_local_max_amp > 0.59651
                                                return 'V'
                                        else:  # if min_amp_local_min_amp > 1.04491
                                            if peak_peak_local <= 1.18527:
                                                if w_50_cali <= 1.90507:
                                                    return 'V'
                                                else:  # if w_50_cali_w50 > 1.90507
                                                    return 'N'
                                            else:  # if peak_peak_local_peak_peak > 1.18527
                                                if min_amp_local <= 1.0462:
                                                    return 'N'
                                                else:  # if min_amp_local_min_amp > 1.0462
                                                    return 'V'
                                    else:  # if max_amp_cali_max_amp > 3.93265
                                        return 'N'
                                else:  # if min_amp_cali_min_amp > 1.28259
                                    if max_amp_local <= 0.678:
                                        return 'N'
                                    else:  # if max_amp_local_max_amp > 0.678
                                        return 'V'
                            else:  # if w_20_cali_w20 > 7.09577
                                return 'N'
                        else:  # if w_80_local_w80 > 3.46573
                            return 'N'
            else:  # if min_amp_local_min_amp > 1.05234
                if max_amp_cali <= -0.06448:
                    return 'N'
                else:  # if max_amp_cali_max_amp > -0.06448
                    if w_20_cali <= 6.16377:
                        if w_20_local <= 0.51765:
                            return 'N'
                        else:  # if w_20_local_w20 > 0.51765
                            if min_amp_cali <= 1.06795:
                                if w_20_cali <= 1.54431:
                                    if max_amp_local <= 0.89873:
                                        if min_amp_local <= 1.14914:
                                            if min_amp_local <= 1.14484:
                                                if w_20_local <= 1.17521:
                                                    return 'V'
                                                else:  # if w_20_local_w20 > 1.17521
                                                    return 'N'
                                            else:  # if min_amp_local_min_amp > 1.14484
                                                return 'N'
                                        else:  # if min_amp_local_min_amp > 1.14914
                                            return 'V'
                                    else:  # if max_amp_local_max_amp > 0.89873
                                        if peak_peak_cali <= 0.86077:
                                            return 'N'
                                        else:  # if peak_peak_cali_peak_peak > 0.86077
                                            if w_80_local0 <= 1.25773:
                                                if w_20_cali <= 1.19623:
                                                    if w_50_cali <= 1.37855:
                                                        if min_amp_cali <= 0.91902:
                                                            return 'V'
                                                        else:  # if min_amp_cali_min_amp > 0.91902
                                                            return 'N'
                                                    else:  # if w_50_cali_w50 > 1.37855
                                                        return 'N'
                                                else:  # if w_20_cali_w20 > 1.19623
                                                    return 'N'
                                            else:  # if w_80_local_w80 > 1.25773
                                                return 'V'
                                else:  # if w_20_cali_w20 > 1.54431
                                    if max_amp_cali <= 1.3649:
                                        if w_50_local <= 1.49783:
                                            if min_amp_local <= 1.05857:
                                                return 'N'
                                            else:  # if min_amp_local_min_amp > 1.05857
                                                if w_80_local0 <= 1.12894:
                                                    return 'N'
                                                else:  # if w_80_local_w80 > 1.12894
                                                    if peak_peak_cali <= 0.92389:
                                                        return 'V'
                                                    else:  # if peak_peak_cali_peak_peak > 0.92389
                                                        if w_20_local <= 1.49111:
                                                            return 'V'
                                                        else:  # if w_20_local_w20 > 1.49111
                                                            return 'N'
                                        else:  # if w_50_local_w50 > 1.49783
                                            return 'V'
                                    else:  # if max_amp_cali_max_amp > 1.3649
                                        if w_20_cali <= 2.23792:
                                            return 'V'
                                        else:  # if w_20_cali_w20 > 2.23792
                                            if w_20_local <= 2.21035:
                                                return 'N'
                                            else:  # if w_20_local_w20 > 2.21035
                                                if w_50_local <= 2.32673:
                                                    return 'V'
                                                else:  # if w_50_local_w50 > 2.32673
                                                    return 'N'
                            else:  # if min_amp_cali_min_amp > 1.06795
                                if w_50_local <= 1.08502:
                                    if w_50_cali <= 0.67222:
                                        return 'N'
                                    else:  # if w_50_cali_w50 > 0.67222
                                        if max_amp_cali <= 0.62144:
                                            return 'V'
                                        else:  # if max_amp_cali_max_amp > 0.62144
                                            if peak_peak_cali <= 1.17314:
                                                if w_80_local0 <= 0.86714:
                                                    if min_amp_cali <= 1.71061:
                                                        return 'V'
                                                    else:  # if min_amp_cali_min_amp > 1.71061
                                                        return 'N'
                                                else:  # if w_80_local_w80 > 0.86714
                                                    if peak_peak_local <= 0.8115:
                                                        return 'N'
                                                    else:  # if peak_peak_local_peak_peak > 0.8115
                                                        if w_80_cali <= 0.84513:
                                                            if max_amp_local <= 0.82535:
                                                                return 'V'
                                                            else:  # if max_amp_local_max_amp > 0.82535
                                                                return 'N'
                                                        else:  # if w_80_cali_w80 > 0.84513
                                                            if peak_peak_local <= 1.04523:
                                                                if w_50_local <= 0.89206:
                                                                    return 'V'
                                                                else:  # if w_50_local_w50 > 0.89206
                                                                    return 'N'
                                                            else:  # if peak_peak_local_peak_peak > 1.04523
                                                                if peak_peak_cali <= 0.96644:
                                                                    return 'N'
                                                                else:  # if peak_peak_cali_peak_peak > 0.96644
                                                                    if peak_peak_local <= 1.49653:
                                                                        return 'V'
                                                                    else:  # if peak_peak_local_peak_peak > 1.49653
                                                                        return 'N'
                                            else:  # if peak_peak_cali_peak_peak > 1.17314
                                                if peak_peak_local <= 1.00877:
                                                    return 'N'
                                                else:  # if peak_peak_local_peak_peak > 1.00877
                                                    if min_amp_local <= 1.22521:
                                                        if w_50_cali <= 1.07863:
                                                            if min_amp_cali <= 1.30748:
                                                                return 'V'
                                                            else:  # if min_amp_cali_min_amp > 1.30748
                                                                return 'N'
                                                        else:  # if w_50_cali_w50 > 1.07863
                                                            return 'V'
                                                    else:  # if min_amp_local_min_amp > 1.22521
                                                        return 'V'
                                else:  # if w_50_local_w50 > 1.08502
                                    if min_amp_local <= 1.45328:
                                        if w_80_cali <= 1.52317:
                                            if w_80_local0 <= 1.42088:
                                                if w_20_cali <= 1.48334:
                                                    if max_amp_cali <= 1.12195:
                                                        return 'V'
                                                    else:  # if max_amp_cali_max_amp > 1.12195
                                                        if w_50_cali <= 1.25828:
                                                            return 'V'
                                                        else:  # if w_50_cali_w50 > 1.25828
                                                            return 'N'
                                                else:  # if w_20_cali_w20 > 1.48334
                                                    if min_amp_cali <= 1.91663:
                                                        if w_20_local <= 1.27206:
                                                            return 'N'
                                                        else:  # if w_20_local_w20 > 1.27206
                                                            if w_20_local <= 1.59615:
                                                                return 'V'
                                                            else:  # if w_20_local_w20 > 1.59615
                                                                return 'N'
                                                    else:  # if min_amp_cali_min_amp > 1.91663
                                                        return 'N'
                                            else:  # if w_80_local_w80 > 1.42088
                                                if min_amp_local <= 1.2767:
                                                    return 'N'
                                                else:  # if min_amp_local_min_amp > 1.2767
                                                    if max_amp_cali <= 1.47451:
                                                        return 'N'
                                                    else:  # if max_amp_cali_max_amp > 1.47451
                                                        return 'V'
                                        else:  # if w_80_cali_w80 > 1.52317
                                            if min_amp_local <= 1.45209:
                                                if w_20_local <= 1.06543:
                                                    if min_amp_cali <= 1.1994:
                                                        return 'N'
                                                    else:  # if min_amp_cali_min_amp > 1.1994
                                                        if w_80_local0 <= 2.10063:
                                                            return 'V'
                                                        else:  # if w_80_local_w80 > 2.10063
                                                            return 'N'
                                                else:  # if w_20_local_w20 > 1.06543
                                                    if max_amp_local <= 0.55777:
                                                        if max_amp_cali <= 0.41603:
                                                            return 'V'
                                                        else:  # if max_amp_cali_max_amp > 0.41603
                                                            return 'N'
                                                    else:  # if max_amp_local_max_amp > 0.55777
                                                        return 'V'
                                            else:  # if min_amp_local_min_amp > 1.45209
                                                return 'N'
                                    else:  # if min_amp_local_min_amp > 1.45328
                                        if w_80_local0 <= 4.22795:
                                            if w_20_local <= 4.18621:
                                                return 'V'
                                            else:  # if w_20_local_w20 > 4.18621
                                                if w_20_local <= 4.21538:
                                                    return 'N'
                                                else:  # if w_20_local_w20 > 4.21538
                                                    if min_amp_local <= 6.77986:
                                                        return 'V'
                                                    else:  # if min_amp_local_min_amp > 6.77986
                                                        return 'N'
                                        else:  # if w_80_local_w80 > 4.22795
                                            return 'N'
                    else:  # if w_20_cali_w20 > 6.16377
                        return 'N'

