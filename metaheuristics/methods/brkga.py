from ..core import objective_function_with_time
from ..core.types import MetaheuristicContext


def run_brkga(context: MetaheuristicContext) -> dict:
    """BRKGA baseline evaluation over the shared spatial-time objective."""
    if not context.allocations:
        return {
            'method_code': context.method_code,
            'method_name': context.method_name,
            'status': 'error',
            'message': 'No allocation candidates available.',
        }

    first_candidate = context.allocations[0]
    eval_result = objective_function_with_time(
        df_walkability=context.df_walkability,
        df_hex_time_matrix=context.df_hex_time_matrix,
        allocation_items=first_candidate['allocation'],
        candidate_dimensions=context.dimensions,
    )

    return {
        'method_code': context.method_code,
        'method_name': context.method_name,
        'status': 'baseline_ready',
        'seed_used': first_candidate['seed'],
        'best_objective_value': eval_result['objective_value'],
        'applied_allocation_size': eval_result['applied_allocation_size'],
        'message': 'BRKGA placeholder currently evaluates the first spatial candidate using shared objective.',
    }
