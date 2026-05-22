from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class MetaheuristicContext:
    """Shared context passed to each metaheuristic method implementation."""
    df_walkability: pd.DataFrame
    df_hex_time_matrix: pd.DataFrame
    budget: int
    method_code: str
    method_name: str
    seeds: List[int]
    walking_profile: str
    dimensions: List[str]
    source_hex_ids: List[str]
    baseline_iqc_total: Optional[float]
    allocations: List[Dict[str, object]]
