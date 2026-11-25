import numpy as np
from dataclasses import dataclass
from pipeline import config as C
from pipeline.inputs import PipelineInput


@dataclass
class ParsedData:
    z_scores:      np.ndarray   # (N,)
    speed_levels:  np.ndarray   # (N,) HIGH | MEDIUM | LOW (config severity scale on mean forecast speed)
    speed_trends:  np.ndarray   # (N,) WORSENING | STABLE | IMPROVING
    faulty:        np.ndarray   # (N,) bool - sensor reported 0 (missing) in the observed hour
    is_rush_hour:  bool
    network_avg:   float
    num_severe:    int          # sensors at HIGH
    num_heavy:     int          # sensors at MEDIUM


class Parser:
    """Stage 1: per-sensor severity, trend and z-score from the forecast tensor (thresholds from pipeline.config)."""

    def parse(self, inp: PipelineInput) -> ParsedData:
        avg_future_speed = np.mean(inp.pred_speed, axis=0)
        stds = np.where(inp.hist_stds == 0, 1e-6, inp.hist_stds)
        z_scores = (avg_future_speed - inp.hist_means) / stds

        speed_levels = np.array([C.speed_to_severity(s) for s in avg_future_speed], dtype=object)

        diff = np.mean(inp.pred_speed[-3:, :], axis=0) - np.mean(inp.pred_speed[:3, :], axis=0)
        speed_trends = np.full(avg_future_speed.shape, "STABLE", dtype=object)
        speed_trends[diff < -C.TREND_DELTA_MPH] = "WORSENING"
        speed_trends[diff > C.TREND_DELTA_MPH] = "IMPROVING"

        observed = getattr(inp, "observed_speed", None)
        if observed is not None:
            faulty = np.any(observed <= C.SENSOR_FAULT_SPEED, axis=0)
        else:
            faulty = np.zeros(avg_future_speed.shape, dtype=bool)

        is_rush_hour = inp.day_of_week < 5 and any(lo <= inp.hour <= hi for lo, hi in C.RUSH_HOURS)

        return ParsedData(
            z_scores=z_scores,
            speed_levels=speed_levels,
            speed_trends=speed_trends,
            faulty=faulty,
            is_rush_hour=bool(is_rush_hour),
            network_avg=float(np.mean(avg_future_speed)),
            num_severe=int(np.sum(speed_levels == "HIGH")),
            num_heavy=int(np.sum(speed_levels == "MEDIUM")),
        )
