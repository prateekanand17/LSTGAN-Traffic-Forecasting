from dataclasses import dataclass, field
from typing import List
from pipeline.zone_detection import CongestionZone


@dataclass
class ZoneMessage:
    zone_id:          str
    severity:         str
    severity_score:   float    # pipeline.config.SEV_SCORES
    avg_speed_60min:  float
    trend:            str
    cascade_prob_raw: float


@dataclass
class EnrichedZoneState:
    zone:                  CongestionZone
    neighbor_messages:     List[ZoneMessage]
    updated_cascade_risk:  float
    network_pressure:      float
    neighbor_context_text: str
    neighbor_weights:      List[float] = field(default_factory=list)
