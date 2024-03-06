import math
import numpy as np

from itertools import combinations
from kron_vec_product import kron_vec_prod

#######################################################################################################
#######################################################################################################
########################################## Constructors ###############################################
#######################################################################################################
#######################################################################################################

def fitness_fn_constructor(exp_val_func, arguments):
    """
    Parameters:
        - exp_val_func: expectation value function.
        - arguments: arguments for exp_val_func. 
        
    Return:
        - lambda function: fitness function, as required by the constructor of PyGAD's GA class.           
    """    
    # Evaluation of expectation value.
    f_v = lambda vec: exp_val_func(vec, *arguments)
    
    # Define fitness function as the negative of the expectation value.
    return lambda ga_instance, solution, solution_idx: -1 * f_v(np.array(solution))

def crossover_fn_constructor(crossover_func, arguments):
    """
    Parameters:
        - crossover_func: function that implements crossover.
        - arguments: arguments needed for crossover_func. 
        
    Return:
        - lambda function: crossover function, as required by the constructor of PyGAD's GA class.           
    """    
    # Evaluation of crossover function.
    return lambda parents, offspring_size, ga_instance: crossover_func(parents, offspring_size, ga_instance, *arguments)

def mutation_fn_constructor(crossover_func, arguments):
    """ 
    NOT IMPLEMENTED CAUSE NOT NEEDED
    Parameters:
        - crossover_func: expectation value function.
        - arguments: arguments needed for crossover_func. 
        
    Return:
        - lambda function: crossover function, as required by the constructor of PyGAD's GA class.           
    """    
    # Evaluation of crossover function.
    return

def style_summary(summary_text, phase_max, coeff_max):
    """
    Replaces Gene Space in summary with something more appropriate
    """
    text_split = summary_text.split('\n')
    for index, item in enumerate(text_split):
        if 'Gene Space' in item:
            text_split[index] = f'Gene Space: Phases -> ({-phase_max:.5f}, {phase_max:.5f}), Coefficients -> ({-coeff_max}, {coeff_max})'
            
    return '\n'.join(text_split)

#######################################################################################################
#######################################################################################################
############################################ Crossover ################################################
#######################################################################################################
#######################################################################################################

def type_sp_crossover(parents, offspring_size, ga_instance, N, m, unitaries):
    """
    Applies single-point crossover by parameter 'type'.
    """
    n_phases, n_unit, n_deform, n_superpos = int(N-1), 3*N*unitaries, 2*m*N, 2*m
    
    # List for storing offspring
    offspring = []
    # Index to loop through parents
    idx = 0
    
    while len(offspring) != offspring_size[0]:
        # Select parents.
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        
        # Create parameter arrays.
        phases_1 = parent1[:n_phases]
        unitaries_1 = parent1[n_phases: n_phases + n_unit]
        deformation_1 = parent1[n_phases + n_unit: n_phases + n_unit + n_deform]
        superposition_1 = parent1[n_phases + n_unit + n_deform:]
        
        phases_2 = parent2[:n_phases]
        unitaries_2 = parent2[n_phases: n_phases + n_unit]
        deformation_2 = parent2[n_phases + n_unit: n_phases + n_unit + n_deform]
        superposition_2 = parent2[n_phases + n_unit + n_deform:]
        
        # Select split by percentage. This is calculated from the number of
        # deformation parameters, since there are always more of them (provided
        # m > 1).
        split = np.random.choice(range(n_deform+1))/n_deform

        # Crossover step.
        phases_1[int(n_phases*split):] = phases_2[int(n_phases*split):]
        unitaries_1[int(n_unit*split):] = unitaries_2[int(n_unit*split):]
        deformation_1[int(n_deform*split):] = deformation_2[int(n_deform*split):]
        superposition_1[int(n_superpos*split):] = superposition_2[int(n_superpos*split):]
        
        # Append offspring.
        offspring.append(np.array([*phases_1, *unitaries_1, *deformation_1, *superposition_1]))
        
        # Up index by 1.
        idx += 1

    return np.array(offspring)

#######################################################################################################
#######################################################################################################
############################################ Mutation #################################################
#######################################################################################################
#######################################################################################################

def one_gene_mut(offspring, ga_instance):
    """
    Mutates a single gene by at most 5% of it's value.
    """
    # Iterate over offspring.
    for chromosome_idx in range(offspring.shape[0]):
        # Generate random gene index.
        random_gene_idx = np.random.choice(range(offspring.shape[1]))
        # Mutate.
        offspring[chromosome_idx, random_gene_idx] *= (1 + np.random.uniform(-0.5, 0.5))
        # If out of bounds, bring back down.
        max_value = ga_instance.gene_space[random_gene_idx]['high']
        if np.abs(offspring[chromosome_idx, random_gene_idx]) > max_value:
            offspring[chromosome_idx, random_gene_idx] = np.sign(offspring[chromosome_idx, random_gene_idx])*max_value*np.random.uniform(0.99, 1)

    return offspring