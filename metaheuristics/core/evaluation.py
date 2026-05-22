import math
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from walkability.iqc import calculate_walkability_index


ID_COLUMNS = ['h3_id', 'latitude', 'longitude']
CORE_INDICATOR_COLUMNS = [
    'S_saude',
    'S_educacao',
    'S_abastecimento',
    'S_lazer',
    'S_servicos',
    'I_seguranca',
    'A_vegetacao',
    'A_agua',
    'C_conectividade',
    'T_transporte',
    'U_urbanidade',
]

HEX_TIME_MATRIX_REQUIRED_COLUMNS = ['source_h3_id', 'target_h3_id', 'time_min']


def get_available_dimensions(df_walkability: pd.DataFrame,
                             candidate_dimensions: Iterable[str]) -> List[str]:
    """Return dimensions that exist in the provided walkability dataframe."""
    if df_walkability is None or df_walkability.empty:
        return []
    return [col for col in candidate_dimensions if col in df_walkability.columns]


def compute_baseline_iqc_total(df_walkability: pd.DataFrame) -> Optional[float]:
    """Compute baseline global IQC (sum of IQC) when IQC column exists."""
    if df_walkability is None or df_walkability.empty:
        return None
    if 'IQC' not in df_walkability.columns:
        return None
    return float(df_walkability['IQC'].sum())


def calculate_time_decay_weight(time_min: float, max_time: float = 20.0) -> float:
    """Cosine-decay accessibility weight."""
    if pd.isna(time_min):
        return 0.0
    t = float(time_min)
    if t < 0 or t > max_time:
        return 0.0
    return float((1 + math.cos(math.pi * t / max_time)) / 2)


def validate_hex_time_matrix(df_hex_time_matrix: pd.DataFrame,
                             max_time: float = 20.0) -> pd.DataFrame:
    """
    Validate and normalize hex impact matrix.

    Required columns: source_h3_id, target_h3_id, time_min.
    If alpha_20 is absent, it is computed from time_min.
    """
    if df_hex_time_matrix is None or df_hex_time_matrix.empty:
        raise ValueError("df_hex_time_matrix is empty.")

    missing_cols = [col for col in HEX_TIME_MATRIX_REQUIRED_COLUMNS if col not in df_hex_time_matrix.columns]
    if missing_cols:
        raise ValueError(f"df_hex_time_matrix is missing required columns: {missing_cols}")

    df_matrix = df_hex_time_matrix.copy()
    df_matrix['source_h3_id'] = df_matrix['source_h3_id'].astype(str)
    df_matrix['target_h3_id'] = df_matrix['target_h3_id'].astype(str)
    df_matrix['time_min'] = pd.to_numeric(df_matrix['time_min'], errors='coerce')
    df_matrix = df_matrix.dropna(subset=['source_h3_id', 'target_h3_id', 'time_min'])
    df_matrix = df_matrix[df_matrix['time_min'] <= max_time].copy()
    if df_matrix.empty:
        raise ValueError("df_hex_time_matrix has no rows with time_min <= max_time.")

    if 'alpha_20' in df_matrix.columns:
        df_matrix['alpha_20'] = pd.to_numeric(df_matrix['alpha_20'], errors='coerce')
    else:
        df_matrix['alpha_20'] = df_matrix['time_min'].apply(
            lambda t: calculate_time_decay_weight(t, max_time=max_time)
        )

    df_matrix['alpha_20'] = df_matrix['alpha_20'].fillna(0.0)
    df_matrix = df_matrix[df_matrix['alpha_20'] > 0].copy()
    if df_matrix.empty:
        raise ValueError("df_hex_time_matrix has no positive alpha_20 rows.")

    # Keep one row per pair (best/shortest path).
    df_matrix = (
        df_matrix.sort_values(['source_h3_id', 'target_h3_id', 'time_min'])
        .drop_duplicates(subset=['source_h3_id', 'target_h3_id'], keep='first')
        .reset_index(drop=True)
    )
    return df_matrix


def normalize_spatial_allocation(allocation_items: Iterable[Dict[str, object]],
                                 valid_dimensions: Iterable[str]) -> pd.DataFrame:
    """Normalize allocation items to a compact dataframe."""
    if allocation_items is None:
        return pd.DataFrame(columns=['h3_id', 'dimension', 'quantity'])

    valid_dimensions = set(valid_dimensions)
    df_alloc = pd.DataFrame(list(allocation_items))
    if df_alloc.empty:
        return pd.DataFrame(columns=['h3_id', 'dimension', 'quantity'])

    required_cols = ['h3_id', 'dimension', 'quantity']
    missing_cols = [col for col in required_cols if col not in df_alloc.columns]
    if missing_cols:
        raise ValueError(f"Allocation items are missing required fields: {missing_cols}")

    df_alloc['h3_id'] = df_alloc['h3_id'].astype(str)
    df_alloc['dimension'] = df_alloc['dimension'].astype(str)
    df_alloc['quantity'] = pd.to_numeric(df_alloc['quantity'], errors='coerce').fillna(0.0)
    df_alloc = df_alloc[df_alloc['quantity'] > 0].copy()
    df_alloc = df_alloc[df_alloc['dimension'].isin(valid_dimensions)].copy()
    if df_alloc.empty:
        return pd.DataFrame(columns=['h3_id', 'dimension', 'quantity'])

    df_alloc = (
        df_alloc.groupby(['h3_id', 'dimension'], as_index=False)['quantity']
        .sum()
        .sort_values(['h3_id', 'dimension'])
        .reset_index(drop=True)
    )
    return df_alloc


def apply_spatial_allocation_with_time(df_walkability: pd.DataFrame,
                                       df_hex_time_matrix: pd.DataFrame,
                                       allocation_items: Iterable[Dict[str, object]],
                                       candidate_dimensions: Iterable[str],
                                       max_time: float = 20.0) -> pd.DataFrame:
    """
    Apply a spatial POI allocation using source-target alpha impact matrix.
    """
    if df_walkability is None or df_walkability.empty:
        raise ValueError("df_walkability is empty.")

    if 'h3_id' not in df_walkability.columns:
        raise ValueError("df_walkability is missing required column 'h3_id'.")

    available_dimensions = get_available_dimensions(df_walkability, candidate_dimensions)
    if not available_dimensions:
        raise ValueError("No candidate POI dimensions found in df_walkability.")

    df_matrix = validate_hex_time_matrix(df_hex_time_matrix, max_time=max_time)
    df_alloc = normalize_spatial_allocation(allocation_items, available_dimensions)

    df_updated = df_walkability.copy()
    df_updated['h3_id'] = df_updated['h3_id'].astype(str)

    # Ensure indicator columns are numeric before updates.
    for dim in available_dimensions:
        df_updated[dim] = pd.to_numeric(df_updated[dim], errors='coerce').fillna(0.0)

    if df_alloc.empty:
        return df_updated

    # Keep only source hexagons represented in the matrix.
    df_alloc = df_alloc[df_alloc['h3_id'].isin(df_matrix['source_h3_id'].unique())].copy()
    if df_alloc.empty:
        return df_updated

    merged = df_alloc.merge(
        df_matrix[['source_h3_id', 'target_h3_id', 'alpha_20']],
        left_on='h3_id',
        right_on='source_h3_id',
        how='inner',
    )
    if merged.empty:
        return df_updated

    merged['delta'] = merged['quantity'] * merged['alpha_20']
    grouped = (
        merged.groupby(['target_h3_id', 'dimension'], as_index=False)['delta']
        .sum()
    )
    if grouped.empty:
        return df_updated

    delta_pivot = grouped.pivot(index='target_h3_id', columns='dimension', values='delta').fillna(0.0)
    for dim in delta_pivot.columns:
        dim_delta = delta_pivot[dim]
        df_updated[dim] = df_updated[dim] + df_updated['h3_id'].map(dim_delta).fillna(0.0)

    return df_updated


def recalculate_iqc_and_critic(df_final: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Recalculate CRITIC weights and IQC from an updated dataframe.

    Expected input is the current solution dataframe (`df_final`), usually based on
    `df_walkability` after new POI allocations. Existing IQC is discarded and
    recomputed for all hexagons.
    """
    if df_final is None or df_final.empty:
        raise ValueError("df_final is empty.")

    missing_id_cols = [col for col in ID_COLUMNS if col not in df_final.columns]
    if missing_id_cols:
        raise ValueError(f"df_final is missing required ID columns: {missing_id_cols}")

    available_indicator_cols = [col for col in CORE_INDICATOR_COLUMNS if col in df_final.columns]
    if not available_indicator_cols:
        raise ValueError("df_final has no core indicator columns for IQC recalculation.")

    df_indicators = df_final[ID_COLUMNS + available_indicator_cols].copy()
    if 'IQC' in df_indicators.columns:
        df_indicators = df_indicators.drop(columns=['IQC'])

    for col in available_indicator_cols:
        df_indicators[col] = pd.to_numeric(df_indicators[col], errors='coerce')
    df_indicators[available_indicator_cols] = df_indicators[available_indicator_cols].fillna(0.0)

    df_recomputed, critic_weights = calculate_walkability_index(df_indicators)

    df_updated = df_final.copy()
    df_updated['IQC'] = df_recomputed['IQC'].values

    return df_updated, critic_weights


def objective_function(df_final: pd.DataFrame) -> Dict[str, object]:
    """
    Common objective function for all metaheuristic methods.

    Recomputes CRITIC weights and IQC first, then returns an objective scalar.
    """
    df_updated, critic_weights = recalculate_iqc_and_critic(df_final)

    objective_value = float(df_updated['IQC'].sum())

    return {
        'objective_metric': 'sum_iqc',
        'objective_value': objective_value,
        'optimization_direction': 'maximize',
        'df_final_updated': df_updated,
        'critic_weights': critic_weights,
    }


def objective_function_with_time(df_walkability: pd.DataFrame,
                                 df_hex_time_matrix: pd.DataFrame,
                                 allocation_items: Iterable[Dict[str, object]],
                                 candidate_dimensions: Iterable[str],
                                 max_time: float = 20.0) -> Dict[str, object]:
    """
    Apply spatial allocation with time decay, then evaluate objective.
    """
    allocation_list = list(allocation_items) if allocation_items is not None else []

    df_candidate = apply_spatial_allocation_with_time(
        df_walkability=df_walkability,
        df_hex_time_matrix=df_hex_time_matrix,
        allocation_items=allocation_list,
        candidate_dimensions=candidate_dimensions,
        max_time=max_time,
    )
    eval_result = objective_function(df_candidate)
    eval_result['applied_allocation_size'] = len(allocation_list)
    return eval_result
