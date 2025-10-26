import numpy as np
from pymoo.indicators.hv import HV

from src.core.abstract import OptimizerAbstract


def hypervolume_change(optimizer: OptimizerAbstract):
    previous_hv = 0.0
    hv_indicator: HV | None = None

    def get_change():
        nonlocal previous_hv, hv_indicator

        current_front = np.array([sol.objectives for sol in optimizer.get_solutions()])
        if not hv_indicator:
            reference_point = np.max(current_front, axis=0) + 1e-6
            hv_indicator = HV(ref_point=reference_point)

        current_hv = hv_indicator.do(current_front) or 0.0
        delta = current_hv - previous_hv
        previous_hv = current_hv
        return delta

    return get_change
