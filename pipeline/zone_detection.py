import hashlib
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List

from sklearn.cluster import DBSCAN

from pipeline import config as C
from pipeline.inputs import PipelineInput
from pipeline.parse import ParsedData


@dataclass
class CongestionZone:
    zone_id:           str
    sensor_indices:    List[int]
    center_lat:        float
    center_lng:        float
    avg_speed_60min:   float        # mean forecast speed over the zone and the 60-min horizon
    min_speed_60min:   float
    severity:          str          # detector severity (config scale on avg_speed_60min): HIGH|MEDIUM|LOW
    trend:             str          # majority vote: WORSENING|STABLE|IMPROVING
    cascade_risk_raw:  List[int]    # sensors within BFS_HOPS downstream on the distance graph, outside the zone
    anomaly_score:     float
    roads:             List[str] = field(default_factory=list)   # corridors (road-direction) covered by the zone
    n_faulty:          int = 0      # sensors in the zone with missing readings in the observed hour


class ZoneDetector:
    """Stage 3: DBSCAN on lat/lng of sensors whose mean forecast speed is below ZONE_SPEED_THRESHOLD.

    Clustering uses Euclidean distance on (lat, lng) with eps=DBSCAN_EPS_DEG; this is what is implemented and
    what the paper must describe. Zone IDs are a deterministic hash of the member station IDs.
    `eps` defaults to DBSCAN_EPS_DEG; other values are used only by the F2 sensitivity runs.
    """

    def __init__(self, eps: float = C.DBSCAN_EPS_DEG):
        self.eps = eps

    def detect(self, inp: PipelineInput, parsed: ParsedData) -> List[CongestionZone]:
        avg_future_speed = np.mean(inp.pred_speed, axis=0)
        congested_idx = np.where(avg_future_speed < C.ZONE_SPEED_THRESHOLD)[0]
        if len(congested_idx) == 0:
            return []

        coords = np.column_stack((inp.lats[congested_idx], inp.lngs[congested_idx]))
        labels = DBSCAN(eps=self.eps, min_samples=C.DBSCAN_MIN_SAMPLES).fit(coords).labels_
        roads = getattr(inp, "roads", None)

        zones = []
        for label in sorted(set(labels)):
            if label == -1:
                continue
            idx = congested_idx[labels == label]
            z_speeds = avg_future_speed[idx]
            z_avg = float(np.mean(z_speeds))
            trends = parsed.speed_trends[idx]
            counts = {t: int(np.sum(trends == t)) for t in ("WORSENING", "STABLE", "IMPROVING")}
            members = sorted(inp.station_ids[i] for i in idx)
            zid = "zone_" + hashlib.sha1(",".join(members).encode()).hexdigest()[:8]
            zones.append(CongestionZone(
                zone_id=zid,
                sensor_indices=idx.tolist(),
                center_lat=float(np.mean(inp.lats[idx])),
                center_lng=float(np.mean(inp.lngs[idx])),
                avg_speed_60min=z_avg,
                min_speed_60min=float(np.min(z_speeds)),
                severity=C.speed_to_severity(z_avg),
                trend=max(counts, key=counts.get),
                cascade_risk_raw=self._bfs(inp.adj_mx, idx.tolist(), C.BFS_HOPS),
                anomaly_score=float(np.mean(np.abs(parsed.z_scores[idx]))),
                roads=sorted({roads[i] for i in idx}) if roads else [],
                n_faulty=int(parsed.faulty[idx].sum()),
            ))
        return zones

    @staticmethod
    def _bfs(adj_mx: np.ndarray, sources: List[int], hops: int) -> List[int]:
        visited = set(sources)
        queue = deque((s, 0) for s in sources)
        at_risk = set()
        while queue:
            curr, depth = queue.popleft()
            if depth >= hops:
                continue
            for n in np.where(adj_mx[curr, :] > C.BFS_ADJ_THRESHOLD)[0]:
                if n not in visited:
                    visited.add(int(n))
                    at_risk.add(int(n))
                    queue.append((int(n), depth + 1))
        return sorted(at_risk)
