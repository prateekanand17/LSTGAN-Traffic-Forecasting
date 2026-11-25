from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class LineageTracker:
    source_model: str
    run_timestamp: str
    scenario_id: str

@dataclass
class PipelineInput:
    """Standardized input to the MASTRA pipeline."""
    pred_speed:    np.ndarray    # [12, 325]
    actual_speed:  np.ndarray    # [12, 325]
    adj_mx:        np.ndarray    # [325, 325]
    station_ids:   List[str]     # 325 strings
    lats:          np.ndarray    # [325]
    lngs:          np.ndarray    # [325]
    hist_means:    np.ndarray    # [325]
    hist_stds:     np.ndarray    # [325]
    day_of_week:   int           # 0-6
    hour:          int           # 0-23
    timestamp_idx: int

@dataclass
class PipelineResult:
    """Output of the MASTRA pipeline."""
    status:          str         # SUCCESS | BLOCKED_BY_* | NO_CONGESTION
    run_id:          str         # timestamp-based
    lineage:         LineageTracker
    outputs:         Optional[Dict[str, Any]] = None
    evals:           Optional[Dict[str, Any]] = None
    gt_result:       Optional[Any] = None
    integrated_plan: Optional[Dict[str, Any]] = None
    network_avg:     Optional[float] = None
