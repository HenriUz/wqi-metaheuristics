from ..core import (
    allocation_items_to_candidate_matrix,
    objective_function,
)
from ..core.types import MetaheuristicContext
from .debug_nd_io import save_nd_debug_matrices


def run_pso(context: MetaheuristicContext) -> dict:
    """PSO baseline evaluation over the shared spatial-time objective."""
    if not context.allocations:
        return {
            'method_code': context.method_code,
            'method_name': context.method_name,
            'status': 'error',
            'message': 'No allocation candidates available.',
        }

    first_candidate = context.allocations[0]
    candidate_matrix = allocation_items_to_candidate_matrix(
        allocation_items=first_candidate['allocation'],
        objective_state=context.objective_state_nd,
    )
    debug_files = save_nd_debug_matrices(
        context=context,
        seed=first_candidate['seed'],
        candidate_matrix=candidate_matrix,
    )
    eval_result = objective_function(
        candidate_matrix=candidate_matrix,
        objective_state=context.objective_state_nd,
    )

    return {
        'method_code': context.method_code,
        'method_name': context.method_name,
        'status': 'baseline_ready',
        'seed_used': first_candidate['seed'],
        'best_objective_value': eval_result['objective_value'],
        'applied_allocation_size': eval_result['applied_allocation_size'],
        **debug_files,
        'message': 'PSO placeholder currently evaluates the first spatial candidate using shared objective.',
    }
