import numpy as np
import numpy.typing as npt

from ..core import build_final_indicator_matrix_nd, objective_function
from ..core.types import ObjectiveStateND, MetaheuristicContext
from dataclasses import dataclass
from math import floor
from random import random, sample, seed

@dataclass
class Particle():
    """
    Represents a particle in the swarm, containing essential and auxiliary attributes.

    Attributes:
        x (set[int]): Selected allocations.
        objective (float): Value of the objective function.
        number (int): Number of allocations.
        pbest (set[int]): Best allocations found.
        pbest_objective (float): Value of the objective function for the best allocations.
    """
    
    x              : set[int]
    objective      : float
    number         : int
    pbest          : set[int]
    pbest_objective: float

def generate_initial_swarm(
    size: int,
    budget: int,
    baseline: npt.NDArray[np.uint8],
    objective_state_nd: ObjectiveStateND
) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.uint8]]:
    """
    Populates the swarm with randomly generated particles. Each particle has a 50% chance of selecting an allocation

    Returns an array containing all the particles, and a 3-dimensional matrix. The first dimension represents a particle, and the other two dimensions are the binary allocation matrix for that particle.
    
    Args:
        size (int): Swarm size.
        budget (int): Limitation on the number of allocations.
        baseline (NDArray[uint8]): Current status of allocations.
        objective_state_nd (ObjectiveStateND): Required metadata.
    
    Returns:
        swarm (tuple[NDArray[object_], NDArray[uint8]]): Particles and allocations.
    """

    n_hex, n_dim = baseline.shape
    
    swarm = np.empty(size, dtype = Particle)
    allocations = np.zeros((size, n_hex, n_dim), dtype = np.uint8)

    for i in range(size):
        particle = Particle(
            x               = {e for e in range(n_hex * n_dim) if random() <= 0.5},
            objective       = 0.0,
            number          = 0,
            pbest           = set(),
            pbest_objective = 0.0
        )

        particle.pbest = particle.x.copy()
        for e in particle.x:
            hs, ds = np.divmod(e, n_dim)
            allocations[i][hs][ds] = 1
            particle.number += 1

        if particle.number > budget:
            particle.objective = 0
        else:
            matrix = build_final_indicator_matrix_nd(
                candidate_matrix = allocations[i],
                objective_state  = objective_state_nd
            )
            particle.objective = objective_function(final_indicator_matrix=matrix)

        particle.pbest_objective = particle.objective
        swarm[i] = particle

    return swarm, allocations

def scalar_multiplication(
    scalar: float,
    velocity: set[tuple[str, int]]
) -> set[tuple[str, int]]:
    """
    Multiplies a velocity by a scalar value. The result is a new velocity with `floor(scalar * |velocity|)` random elements of `velocity`.

    The scalar must be in [0, 1]: a value of 1 returns the full velocity, and 0 returns an empty set.

    Args:
        scalar (float): Scaling factor in [0, 1].
        velocity (set[tuple[str, int]]): Set of (operator, aisle) operations.
    
    Returns:
        subset (set[tuple[str, int]]): Randomly sampled subset of the velocity.
    """

    if scalar < 0 or scalar > 1:
        return set()
    
    k = floor(scalar * len(velocity))
    return set(sample(sorted(velocity), k=k))

def difference_in_positions(
    target: set[int],
    current: set[int]
) -> set[tuple[str, int]]:
    """
    Computes the velocity needed to transform `current` into `target`.

    Elements only in `target` become additions; elements only in `current` become removals.

    Args:
        target (set[int]): Desired position.
        current (set[int]): Position to be transformed.

    Returns:
        velocity (set[tuple[str, int]]): Set of (operator, aisle) operations. 
    """
    
    additions = {("+", aisle) for aisle in target  - current}
    removals  = {("-", aisle) for aisle in current - target}
    return additions | removals

def number_of_elements(
    beta: float,
    reference_set: set[int]
) -> int:
    """
    Stochastically determines how many elements to operate on, bounded by the set size. Uses `floor(beta)` with a probabilistic +1.
    
    Args:
        beta (float): Expected number of elements.
        reference_set (set[int]): Set whose size serves as the upper bound.
    
    Returns:
        n_elements (int): Number of elements to operate on.
    """
    
    count = floor(beta)
    if random() < beta - count:
        count += 1
    
    length = len(reference_set)
    if count < length:
        return count
    return length

def k_tournament_selection(
    candidates: set[int],
    n_to_add: int,
    k: int,
    budget: int,
    number: int,
    intervention: npt.NDArray[np.uint8],
    objective_state_nd: ObjectiveStateND
) -> set[tuple[str, int]]:
    """
    Greedily selects `n_to_add` aisles from `candidates` using tournament selection.

    For each slot, `k` candidates are sampled and the one that best improves the objective is chosen.

    Args:
        problem (Problem): Dataset information.
        particle (SetParticle): Current particle.
        candidates (set[int]): Aisles not present in `x union pbest union G`.
        n_to_add (int): Number of aisles to select.
        k (int): Tournament size.
    
    Returns:
        additions (set[tuple[str, int]]): Addition operations for the selected aisles.
    """
    
    additions = set()
    remaining = sorted(candidates)
    _, n_dim  = intervention.shape

    for _ in range(n_to_add):
        length = len(remaining)
        if k < length:
            tournament_size = k
        else:
            tournament_size = length
        
        contestants = sample(remaining, tournament_size)
        
        best_allocation = -1
        best_objective  = -1
        
        for allocation in contestants:
            hs, ds = np.divmod(allocation, n_dim)

            if number + 1 > budget:
                objective = 0
            else:
                intervention[hs][ds] = 1
                objective = objective_function(
                    candidate_matrix = intervention,
                    objective_state  = objective_state_nd
                )["objective_value"]
                intervention[hs][ds] = 0

            if objective >= best_objective:
                best_allocation = allocation
                best_objective  = objective

        additions.add(("+", best_allocation))
        remaining.remove(best_allocation)
        
        # Commit the selected aisle so the next round evaluates on top of it.
        hs, ds = np.divmod(best_allocation, n_dim)
        intervention[hs][ds] = 1
        number += 1

    return additions

def removal_of_elements(
    consensus_set: set[int],
    n_to_remove: int
) -> set[tuple[str, int]]:
    """
    Randomly selects `n_to_remove` aisles from the consensus intersection to remove.
    
    Args:
        consensus_set (set[int]): Aisles present in `x intersect pbest intersect G`.
        n_ro_remove (int): Number of aisles to remove.
    
    Returns:
        removals (set[tuple[str, int]]): Removal operations for the selected aisles.
    """
    
    selected = sample(sorted(consensus_set), k=n_to_remove)
    return {("-", aisle) for aisle in selected}

def run_pso(context: MetaheuristicContext) -> dict:
    """
    PSO baseline evaluation over the shared spatial-time objective.
    """
    
    if not context.allocations or not context.objective_state_nd:
        return {
            'method_code': context.method_code,
            'method_name': context.method_name,
            'status': 'error',
            'message': 'No allocation candidates available.',
        }
    
    """
    Estratégias:
    - Na criação do enxame limitar a quantidade para ser menor do que o budget.
    - Usar o baseline como melhor global.
    
    Estruturas:
    - Allocations: matriz de 3 dimensões (partícula, hexágonos, dimensões), representando as alocação para cada partícula.
    - Swarm: lista com as informações da partícula (Particle).
    """

    seed(context.seeds[0])

    # Default parameters SBPSO.
    size = 50
    generations = 600
    c1 = 0.9297
    c2 = 0.2266
    c3 = 1.3086
    c4 = 2.1526
    k = 7
    
    # Problem parameters.
    n_hex    = len(context.source_hex_ids)
    n_dim    = len(context.dimensions)
    budget   = context.budget
    baseline = context.objective_state_nd.baseline_matrix[:, :n_dim]
    universe = set(range(n_hex * n_dim))

    # Initializing swarm.
    swarm, allocations = generate_initial_swarm(size, budget, baseline, context.objective_state_nd)

    # Searching for the best global.
    best_position  = set()
    best_objective = objective_function(
        candidate_matrix=baseline,
        objective_state=context.objective_state_nd
    )["objective_value"]
    
    print(f"DEBUG: {best_objective}")

    for i in range(size):
        if swarm[i].objective >= best_objective:
            best_position  = swarm[i].x.copy()
            best_objective = swarm[i].objective

    # SBPSO.
    for _ in range(generations):
        for i in range(size):
            particle = swarm[i]

            # Cognitive component: pull toward personal best.
            cognitive_velocity = scalar_multiplication(
                c1 * random(),
                difference_in_positions(particle.pbest, particle.x)
            )

            # Social component: pull toward global best.
            social_velocity = scalar_multiplication(
                c2 * random(),
                difference_in_positions(best_position, particle.x)
            )

            # Exploration: add aisles absent from all three reference sets.
            external_aisles  = universe - (particle.x | particle.pbest | best_position)
            random_additions = k_tournament_selection(
                external_aisles,
                number_of_elements(c3 * random(), external_aisles),
                k,
                budget,
                particle.number,
                baseline + allocations[i],
                context.objective_state_nd
            )

            # Diversity: remove aisles present in all three reference sets.
            consensus_aisles = particle.x & particle.pbest & best_position
            random_removals  = removal_of_elements(
                consensus_aisles,
                number_of_elements(c4 * random(), consensus_aisles)
            )

            velocity = cognitive_velocity | social_velocity | random_additions | random_removals

            for op, allocation in velocity:
                if op == "+":
                    particle.number += 1
                    particle.x.add(allocation)
                else:
                    particle.number -= 1
                    particle.x.remove(allocation)
        
        for i in range(size):
            particle = swarm[i]

            if particle.number > budget:
                particle.objective = 0
            else:
                particle.objective = objective_function(
                    candidate_matrix = baseline + allocations[i],
                    objective_state  = context.objective_state_nd
                )["objective_value"]

            if particle.objective >= particle.pbest_objective:
                particle.pbest           = particle.x.copy()
                particle.pbest_objective = particle.objective

            if particle.objective >= best_objective:
                best_position  = particle.pbest.copy()
                best_objective = particle.objective
                
    print(f"DEBUG: {best_objective}")
    print(len(best_position))

    return {
        'method_code': context.method_code,
        'method_name': context.method_name,
        'status': 'baseline_ready',
        'seed_used': context.seeds[0],
        'best_objective_value': best_objective,
        'message': 'PSO placeholder currently evaluates the first spatial candidate using shared objective.',
    }