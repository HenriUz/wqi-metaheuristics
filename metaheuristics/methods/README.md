# Metaheuristics Methods - Quick Start

This folder is for method-specific implementations.

Current method modules:
- `ils.py`
- `grasp.py`
- `brkga.py`
- `pso.py`
- `hybrid_grasp_vns_pr.py`

## Required function contract
Each method file must expose one function:

```python
def run_<method_name>(context: MetaheuristicContext) -> dict:
    ...
```

The `context` object includes:
- `df_walkability`
- `df_hex_time_matrix`
- `budget`
- `seeds`
- `dimensions`
- `source_hex_ids`
- `walking_profile`
- method metadata (`method_code`, `method_name`)

Plain-language note:
- `MetaheuristicContext` is only a container with prepared inputs.
- Your method receives it already filled by `walk_meta_opt(...)`.
- `context.allocations[0]` is simply the first candidate allocation generated from seeds.

## Objective function (shared)
For the spatial allocation scenario (POI insertion by source hexagon),
evaluate candidates with:

```python
from ..core import objective_function_with_time

eval_result = objective_function_with_time(
    df_walkability=context.df_walkability,
    df_hex_time_matrix=context.df_hex_time_matrix,
    allocation_items=candidate_allocation,
    candidate_dimensions=context.dimensions,
)
score = eval_result["objective_value"]  # maximize
```

This objective applies source-target temporal impact (`alpha_20`), then
recalculates CRITIC + IQC and uses:
- `objective_value = sum(IQC)` (global IQC)
- `optimization_direction = "maximize"`

## Minimal method skeleton
```python
from ..core.types import MetaheuristicContext
from ..core import objective_function_with_time

def run_ils(context: MetaheuristicContext) -> dict:
    first_candidate = context.allocations[0]
    eval_result = objective_function_with_time(
        df_walkability=context.df_walkability,
        df_hex_time_matrix=context.df_hex_time_matrix,
        allocation_items=first_candidate["allocation"],
        candidate_dimensions=context.dimensions,
    )
    return {
        "method_code": context.method_code,
        "method_name": context.method_name,
        "status": "ok",
        "best_objective_value": eval_result["objective_value"],
        "message": "Baseline spatial-time evaluation executed."
    }
```

## Registration
After implementing a method:
1. Export/import the function in `metaheuristics/methods/__init__.py`.
2. Register it in `METHOD_RUNNERS`.

Without registration, the optimizer cannot dispatch to the method.

## Quick checklist
- Function compiles.
- Function uses `objective_function(...)`.
- Method is registered in `METHOD_RUNNERS`.
- Return dict has clear `status` and objective value.
