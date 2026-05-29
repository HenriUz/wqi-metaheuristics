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
- `allocations`
- `objective_state_nd` (precompiled numeric state for fast evaluation)
- method metadata (`method_code`, `method_name`)

What `objective_state_nd` is:
- One-time compiled state created before the optimization loop.
- Contains sequential hex indexing, `baseline_matrix`, dimension mapping, and precompiled source-target `alpha` impacts.
- Lets `objective_function(candidate_matrix=..., objective_state=...)` run with ndarray operations only in the hot loop.

Plain-language note:
- `MetaheuristicContext` is only a container with prepared inputs.
- Your method receives it already filled by `walk_meta_opt(...)`.
- `context.allocations[0]` is simply the first candidate allocation generated from seeds.

## Objective function (shared, ndarray entry point)
The low-overhead objective is:

```python
from ..core import allocation_items_to_candidate_matrix, objective_function

candidate_matrix = allocation_items_to_candidate_matrix(
    allocation_items=candidate_allocation,
    objective_state=context.objective_state_nd,
)
eval_result = objective_function(
    candidate_matrix=candidate_matrix,
    objective_state=context.objective_state_nd,
)
score = eval_result["objective_value"]  # maximize
```

This objective:
- uses the source-target time-decay impact (`alpha_20`) precompiled outside the hot loop,
- updates indicators using ndarray operations only,
- recalculates CRITIC + IQC,
- returns `objective_value = sum(IQC)` and `optimization_direction = "maximize"`.

## Debug Matrix Exports (temporary control)
Each method currently writes two `.txt` files per run under:
- `metaheuristics/debug_nd_arrays/`

Saved files:
- `*_baseline_matrix.txt` (baseline ndarray compatible with walkability indicators)
- `*_initial_candidate_matrix.txt` (initial solution matrix converted to ndarray)

Returned fields in method result:
- `debug_baseline_matrix_file`
- `debug_initial_candidate_matrix_file`

## Minimal method skeleton
```python
from ..core.types import MetaheuristicContext
from ..core import allocation_items_to_candidate_matrix, objective_function

def run_ils(context: MetaheuristicContext) -> dict:
    first_candidate = context.allocations[0]
    candidate_matrix = allocation_items_to_candidate_matrix(
        allocation_items=first_candidate["allocation"],
        objective_state=context.objective_state_nd,
    )
    eval_result = objective_function(
        candidate_matrix=candidate_matrix,
        objective_state=context.objective_state_nd,
    )
    return {
        "method_code": context.method_code,
        "method_name": context.method_name,
        "status": "ok",
        "best_objective_value": eval_result["objective_value"],
        "message": "Baseline ndarray spatial-time evaluation executed.",
    }
```

## Registration
After implementing a method:
1. Export/import the function in `metaheuristics/methods/__init__.py`.
2. Register it in `METHOD_RUNNERS`.

Without registration, the optimizer cannot dispatch to the method.

## Quick checklist
- Function compiles.
- Function uses `objective_function(candidate_matrix=..., objective_state=...)`.
- Method is registered in `METHOD_RUNNERS`.
- Return dict has clear `status` and objective value.
