import numpy as np
from typing import List

from pipeline import config as C
from pipeline.inputs import PipelineInput
from pipeline.zone_detection import CongestionZone
from pipeline.graph.zone_state import ZoneMessage, EnrichedZoneState

TREND_SIGN = {"WORSENING": -1, "STABLE": 0, "IMPROVING": 1}


class GraphMessagePassingLayer:
    """Stage 4: two-round deterministic message passing between zones over a communication topology.

    Round 1 (broadcast): zone i emits m_i = (s_i, v_i, tau_i, p_i) with s_i = SEV_SCORES[severity],
        v_i = mean forecast speed, tau_i in {-1,0,+1} (trend), p_i = min(1, |cascade sensors| / |zone|).
    Round 2 (aggregate): w_ji = agg_{a in z_j, b in z_i} A[a, b] over the topology A (inp.comm_adj),
        agg = mean or max (config GMP_WEIGHT_AGG). Messages pass when w_ji > GMP_EDGE_THRESHOLD.
        Pressure (config GMP_PRESSURE):
          normalized: pi_i = sum_j w_ji s_j / sum_j w_ji
          additive:   pi_i = sum_j w_ji (s_j + p_j + max(0, -tau_j))
        Updated risk: p'_i = min(1, p_i + GMP_LAMBDA * pi_i).
    A[a, b] is read as "influence of sensor b on sensor a" for the row-normalized learned graph, so the weight of
    the message from z_j to z_i uses rows of z_i and columns of z_j.
    """

    def __init__(self, weight_agg: str = C.GMP_WEIGHT_AGG, pressure: str = C.GMP_PRESSURE,
                 lam: float = C.GMP_LAMBDA, edge_threshold: float = C.GMP_EDGE_THRESHOLD):
        if weight_agg not in ("mean", "max") or pressure not in ("normalized", "additive"):
            raise ValueError("bad GMP configuration")
        self.weight_agg, self.pressure, self.lam, self.edge_threshold = weight_agg, pressure, lam, edge_threshold

    def run(self, zones: List[CongestionZone], inp: PipelineInput) -> List[EnrichedZoneState]:
        A = getattr(inp, "comm_adj", None)
        if A is None:
            raise ValueError("PipelineInput.comm_adj (communication topology) is required for GMP")

        messages = {}
        for z in zones:
            messages[z.zone_id] = ZoneMessage(
                zone_id=z.zone_id,
                severity=z.severity,
                severity_score=C.SEV_SCORES[z.severity],
                avg_speed_60min=z.avg_speed_60min,
                trend=z.trend,
                cascade_prob_raw=min(1.0, len(z.cascade_risk_raw) / max(1, len(z.sensor_indices))),
            )

        enriched = []
        for z_i in zones:
            received, weights = [], []
            num = den = 0.0
            for z_j in zones:
                if z_j.zone_id == z_i.zone_id:
                    continue
                sub = A[np.ix_(z_i.sensor_indices, z_j.sensor_indices)]
                w = float(sub.max() if self.weight_agg == "max" else sub.mean()) if sub.size else 0.0
                if w <= self.edge_threshold:
                    continue
                m = messages[z_j.zone_id]
                received.append(m)
                weights.append(w)
                if self.pressure == "normalized":
                    num += w * m.severity_score
                    den += w
                else:
                    num += w * (m.severity_score + m.cascade_prob_raw + max(0, -TREND_SIGN[m.trend]))
            pi = (num / den if den > 0 else 0.0) if self.pressure == "normalized" else num
            p0 = messages[z_i.zone_id].cascade_prob_raw
            p_new = min(1.0, p0 + self.lam * pi)

            if received:
                order = np.argsort(weights)[::-1]
                lines = ["Messages from zones connected to this zone in the communication graph "
                         "(strongest connection first):"]
                for k in order:
                    m = received[k]
                    lines.append(f"- {m.zone_id}: severity={m.severity}, trend={m.trend}, "
                                 f"forecast speed={m.avg_speed_60min:.1f} mph, link weight={weights[k]:.4f}")
                text = "\n".join(lines)
            else:
                text = "No connected congested zones in the communication graph."
            enriched.append(EnrichedZoneState(zone=z_i, neighbor_messages=received, updated_cascade_risk=p_new,
                                              network_pressure=float(pi), neighbor_context_text=text,
                                              neighbor_weights=weights))
        return enriched


def no_communication(zones: List[CongestionZone]) -> List[EnrichedZoneState]:
    """Variant without GMP: same zone state, no neighbour information, p' = p."""
    return [EnrichedZoneState(zone=z, neighbor_messages=[],
                              updated_cascade_risk=min(1.0, len(z.cascade_risk_raw) / max(1, len(z.sensor_indices))),
                              network_pressure=0.0,
                              neighbor_context_text="No neighbour information available.",
                              neighbor_weights=[]) for z in zones]
