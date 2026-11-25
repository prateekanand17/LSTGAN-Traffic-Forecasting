from dataclasses import dataclass

import numpy as np

from pipeline.inputs import PipelineInput
from pipeline.parse import ParsedData


@dataclass
class GateResult:
    passed: bool
    reason: str


class ForecastIntegrityGate:
    """Stage 2: plausibility checks on the forecast tensor. The result is recorded; it does not stop the run."""
    MAX_EXTREME_SENSORS = 50
    MAX_DEVIATION_MPH = 40.0

    def check(self, inp: PipelineInput, parsed: ParsedData) -> GateResult:
        if not np.all(np.isfinite(inp.pred_speed)):
            return GateResult(False, "Forecast contains NaN or inf.")
        if parsed.num_severe > self.MAX_EXTREME_SENSORS:
            return GateResult(False, f"{parsed.num_severe} sensors forecast HIGH, above {self.MAX_EXTREME_SENSORS}.")
        observed = getattr(inp, "observed_speed", None)
        if observed is not None and len(inp.pred_speed):
            last = observed[-1]
            ok = last > 0
            if ok.any():
                jump = float(np.max(np.abs(inp.pred_speed[0][ok] - last[ok])))
                if jump > self.MAX_DEVIATION_MPH:
                    return GateResult(False, f"Speed jump of {jump:.1f} mph between last observation and first forecast step.")
        return GateResult(True, "Forecast integrity checks passed.")
