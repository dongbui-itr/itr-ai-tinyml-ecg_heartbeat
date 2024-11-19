import traceback

import numpy as np

from typing import (
    Final,
    List,
)

from numpy.typing import (
    NDArray,
)


class QRSAmplitudeSpecs:
    index = 0
    sample = 0
    rr = 0.0
    w20 = 0
    w50 = 0
    w80 = 0
    max_amp = 0.0
    min_amp = 0.0
    peak_peak = 0.0

    def __eq__(
            self,
            other
    ):
        if isinstance(other, QRSAmplitudeSpecs):

            for key in self.__dict__:
                if key != 'methods':
                    if self.__dict__[key] != other.__dict__[key]:
                        return False
            return True
        return False


class QRSRRSpecs:
    RR1 = 0.0
    RR2 = 0.0
    RR3 = 0.0

    R12 = 0.0
    R23 = 0.0

    RR1_CAL = 0.0
    RR2_CAL = 0.0
    RR3_CAL = 0.0

    RR1_UDT = 0.0
    RR2_UDT = 0.0
    RR3_UDT = 0.0


class DecisionTree:
    THR_RR_MIN: Final[float] = 0.7
    THR_RR_MAX: Final[float] = 1.25
    THR_W_LOCAL: Final[float] = 1.6
    THR_AMP_MAX: Final[float] = 1.8
    THR_AMP_MIN: Final[float] = 0.55

    # S_RUN
    THR_RUN_RR_MIN: Final[float] = 0.85
    THR_RUN_RR_MAX: Final[float] = 1.15
    THR_RUN_RR1_MIN: Final[float] = 0.6
    THR_RUN_RR1_MAX: Final[float] = 1.28

    def check_function(
            self,
            args: List,
            multiplier=1
    ):
        check = np.array(list(map(
            lambda x: self.THR_AMP_MIN * multiplier < x < self.THR_AMP_MAX * multiplier,
            args
        )))

        return check

    def check_ratio_condition(
            self,
            qrs_ratio_to_calibrate: QRSAmplitudeSpecs,
            qrs_ratio_to_updated: QRSAmplitudeSpecs,
    ) -> [bool, bool, bool]:

        check_width = False
        check_peaks = False
        check_amp_max = False
        check_amp_min = False

        ind_amp_max = list()
        ind_amp_min = list()
        try:
            width_data = np.array([qrs_ratio_to_calibrate.w20, qrs_ratio_to_calibrate.w50, qrs_ratio_to_calibrate.w80,
                                   qrs_ratio_to_updated.w20, qrs_ratio_to_updated.w50, qrs_ratio_to_updated.w80])

            check_width = np.count_nonzero(width_data < self.THR_W_LOCAL) > 2
            check_peaks = all(self.check_function([qrs_ratio_to_calibrate.peak_peak, qrs_ratio_to_updated.peak_peak]))

            ind_amp_max = self.check_function([qrs_ratio_to_calibrate.max_amp, qrs_ratio_to_updated.max_amp])
            check_amp_max = all(ind_amp_max)

            ind_amp_min = self.check_function([qrs_ratio_to_calibrate.min_amp, qrs_ratio_to_updated.min_amp])
            check_amp_min = all(ind_amp_min)

        except (Exception,) as error:
            exc_traceback = traceback.sys.exc_info()
            print(f'Error {error} at {exc_traceback[-1].tb_lineno}')

        return check_width, check_peaks, check_amp_max, check_amp_min, ind_amp_max, ind_amp_min

    def check_amp(
            self,
            amp: float,
            ind_amp: NDArray,
            args: List
    ):
        condition = any([
            np.abs(amp) > 0.2 and any(~ind_amp),
            np.abs(amp) <= 0.2 and any(~self.check_function(args, multiplier=2)),
        ])

        return condition

    # @df.timeit
    def width_amp_predict(
            self,
            qrs_label,
            qrs_beat: QRSAmplitudeSpecs,
            qrs_ratio_to_calibrate: QRSAmplitudeSpecs,
            qrs_ratio_to_updated: QRSAmplitudeSpecs
    ) -> str:

        """
            Abnormal ==> using WIDTH + AMP to classify SVE, VES
            Normal ==> using WIDTH + AMP to classify Normal, VES

        """

        symbol = None
        try:
            (check_width, check_peak, check_amp_max, check_amp_min,
             ind_amp_max, ind_amp_min) = self.check_ratio_condition(qrs_ratio_to_calibrate=qrs_ratio_to_calibrate,
                                                                    qrs_ratio_to_updated=qrs_ratio_to_updated)
            if all([check_width, check_amp_max, check_amp_min, check_peak]):
                symbol = 'S' if qrs_label != 'N' else 'N'

            else:
                if qrs_label != 'N':
                    if not check_amp_min:
                        check_amp = self.check_amp(amp=qrs_beat.min_amp,
                                                   ind_amp=ind_amp_min,
                                                   args=[qrs_ratio_to_calibrate.min_amp, qrs_ratio_to_updated.min_amp])
                        symbol = 'V' if check_amp else 'S'

                    elif not check_amp_max:
                        check_amp = self.check_amp(amp=qrs_beat.max_amp,
                                                   ind_amp=ind_amp_max,
                                                   args=[qrs_ratio_to_calibrate.max_amp, qrs_ratio_to_updated.max_amp])
                        symbol = 'V' if check_amp else 'S'

                    else:
                        symbol = 'V'
                else:
                    symbol = 'N' if not all([check_amp_max, check_amp_min]) else 'V'

        except (Exception,) as error:
            exc_traceback = traceback.sys.exc_info()
            print(f'Error {error} at {exc_traceback[-1].tb_lineno}')

        return symbol

    # @df.timeit
    def normal_rr_predict(
            self,
            symbol_bef,
            qrs_rr: QRSRRSpecs
    ) -> str:

        """
            Normal ==> using RR to classify Normal or SVes
        """

        symbol = None
        try:
            condition = any([
                qrs_rr.R23 > self.THR_RR_MAX,
                all(x < self.THR_RR_MIN for x in [qrs_rr.RR3_CAL, qrs_rr.RR3_UDT]),
                all(x > self.THR_RR_MAX for x in [qrs_rr.RR2_CAL, qrs_rr.RR2_UDT]),
                all(x <= self.THR_RR_MAX / 2 for x in [qrs_rr.RR3_CAL, qrs_rr.RR2_UDT])
            ])

            condition_2 = all([
                all(x > self.THR_RR_MAX for x in [qrs_rr.RR1_CAL, qrs_rr.RR1_UDT]),
                self.THR_RR_MIN < qrs_rr.R23 < self.THR_RR_MAX,
                symbol_bef == 'S'
            ])

            if condition or condition_2:
                symbol = 'S'
            else:
                symbol = 'N'
        except (Exception,) as error:
            exc_traceback = traceback.sys.exc_info()
            print(f'Error {error} at {exc_traceback[-1].tb_lineno}')

        return symbol

    @staticmethod
    def logic_function(
            value: float,
            ranges: List = None
    ) -> bool:
        cond = ranges is None
        if not cond:
            thr_from, thr_to = ranges
            if thr_from == 0:
                cond = value <= thr_to
            elif thr_to == 0:
                cond = value > thr_from
            else:
                cond = thr_from < value <= thr_to

        return cond

    # @df.timeit
    def abnormal_rr_predict(
            self,
            qrs_rr: QRSRRSpecs
    ) -> str:

        """
            Abnormal (Normal-SVes) ==> using RR to classify Normal, SVes
        """

        symbol = None
        try:
            if any(x < self.THR_RUN_RR1_MIN for x in [qrs_rr.RR3_CAL, qrs_rr.RR3_UDT]) and 0.9 < qrs_rr.R23 < 1.1:
                if (
                        any(x < self.THR_RR_MIN for x in [qrs_rr.RR3_CAL, qrs_rr.RR3_UDT])
                        and any(x < self.THR_RR_MIN for x in [qrs_rr.RR2_CAL, qrs_rr.RR2_UDT])
                ):
                    symbol = 'N'
                else:
                    symbol = 'S'

            elif any(x <= self.THR_RUN_RR_MIN for x in [qrs_rr.RR3_CAL, qrs_rr.RR3_UDT]):
                if qrs_rr.R23 >= self.THR_RUN_RR_MAX:
                    symbol = 'S'

                elif (
                        all(x < self.THR_RUN_RR_MIN for x in [qrs_rr.RR2_CAL, qrs_rr.RR2_UDT])
                        and 0.9 <= qrs_rr.R23 <= 1.1
                ):
                    symbol = 'S'

                else:
                    symbol = 'N'

            else:
                symbol = 'N'

        except (Exception,) as error:
            exc_traceback = traceback.sys.exc_info()
            print(f'Error {error} at {exc_traceback[-1].tb_lineno}')

        return symbol

    # @df.timeit
    def rr_width_predict(
            self,
            qrs_rr: QRSRRSpecs,
            qrs_amp: QRSAmplitudeSpecs,
    ):
        sym = None
        try:
            rr_check = any([
                all(x < self.THR_RR_MIN for x in [qrs_rr.RR3_CAL, qrs_rr.RR3_UDT]),
                all(x <= self.THR_RR_MAX / 2 for x in [qrs_rr.RR3_CAL, qrs_rr.RR2_UDT]),
                all(x > self.THR_RR_MAX for x in [qrs_rr.RR2_CAL, qrs_rr.RR2_UDT]),
                qrs_rr.R23 > self.THR_RR_MAX,
                qrs_amp.w20 > self.THR_W_LOCAL,
                qrs_amp.w50 > self.THR_W_LOCAL,
                qrs_amp.w80 > self.THR_W_LOCAL * 1.5,
            ])

            sym = 'S' if rr_check else 'N'

        except (Exception,) as error:
            exc_traceback = traceback.sys.exc_info()
            print(f'Error {error} at {exc_traceback[-1].tb_lineno}')

        return sym

    # @df.timeit
    def process(
            self,
            sym_bef,
            qrs_rr: QRSRRSpecs,
            qrs_beat: QRSAmplitudeSpecs,
            ratio_to_updated: QRSAmplitudeSpecs,
            ratio_to_calibrate: QRSAmplitudeSpecs,
    ):
        sym = None
        try:
            # Normal-N vs Abnormal-A
            sym = self.rr_width_predict(qrs_rr=qrs_rr, qrs_amp=ratio_to_calibrate)

            # N: N (N-S) vs V
            if sym == 'N':
                sym = self.width_amp_predict(qrs_label=sym,
                                             qrs_beat=qrs_beat,
                                             qrs_ratio_to_calibrate=ratio_to_calibrate,
                                             qrs_ratio_to_updated=ratio_to_updated)
                # N: N vs S
                if sym == 'N':
                    sym = self.normal_rr_predict(symbol_bef=sym_bef, qrs_rr=qrs_rr)

            else:
                # A: S (N-S) vs V
                sym = self.width_amp_predict(qrs_label=sym,
                                             qrs_beat=qrs_beat,
                                             qrs_ratio_to_calibrate=ratio_to_calibrate,
                                             qrs_ratio_to_updated=ratio_to_updated)
                # S: N vs S
                if sym == 'S':
                    sym = self.abnormal_rr_predict(qrs_rr=qrs_rr)

        except (Exception,) as error:
            exc_traceback = traceback.sys.exc_info()
            print(f'Error {error} at {exc_traceback[-1].tb_lineno}')

        return sym
