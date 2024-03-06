import math
import numpy as np

from itertools import combinations
from kron_vec_product import kron_vec_prod

#######################################################################################################
#######################################################################################################
############################################# Helpers #################################################
#######################################################################################################
#######################################################################################################

def bitset(number, position, N):
    """ 
    This function is to improve readability of later code.
    Checks if the `position`-th bit from the left in the `N` bit binary representation
    of `number` is set (i.e, if it's 1).
    Example: For N = 4, 5 is 0101 in binary. So:
             - bitset(5, 0, 4) is False, since -> [0]101
             - bitset(5, 1, 4) is True,  since -> 0[1]01  
             - bitset(5, 2, 4) is False, since -> 01[0]1
             - bitset(5, 3, 4) is True,  since -> 010[1]
    
    Parameters
    ----------
    number : int
        The number in question.
    position : int
        Position in `number`'s binary representation.
        Leftmost bit is considered the 0th position.
        In case of overflow (>= `N`), this number is used modulo N.
    N : int
        Length of the binary representation of `number` to be considered.
        
    Returns
    -------
    True if `position`-th bit is set. False if it's not set.
    """
    # To check if "position"-th bit of "number" is set, we AND it with 2^(N-1) rightshifted "position" times.
    return bool(number & (2**(N-1) >> (position % N)))

def flipbits(number, position, N):
    """ 
    This function is to improve readability of later code.
    Flips `position`-th and `position`+1-th bits from the left in the `N` bit binary 
    representation of `number`.
    Example: For N = 4, 5 is 0101 in binary. So:
             - flipbits(5, 0, 4) returns 9, since [01]01 goes to [10]01, which is 9.
             - flipbits(5, 1, 4) returns 3, since 0[10]1 goes to 0[01]1, which is 3.  
             - flipbits(5, 2, 4) returns 6, since 01[01] goes to 01[10], which is 6
    
    Parameters
    ----------
    number : int
        The number in question.
    position : int
        Position in `number`'s binary representation.
        Leftmost bit is considered the 0th position.
    N : int 
        Length of the binary representation of `number` to be considered.
        
    Returns
    -------
    new_number : int 
        Integer that results from flipping the `position`-th and `position`+1-th bits from the left.
    """
    ### Flip "position"-th and "position"+1-th bits from the left of "number".
    # To flip the "position"-th bit of "number", we XOR it with 2^(N-1) rightshifted "position" times.
    new_number = number ^ (2**(N-1) >> position)
    # To flip the "position"+1-th bit of "number", we XOR it with 2^(N-1) rightshifted "position"+1 times.
    new_number = new_number ^ (2**(N-1) >> (position+1))
    return new_number

def flipendbits(number, N):
    """ 
    This function is to improve readability of later code.
    Flips first and last bits in the `N` bit binary representation of `number`.
    Example: For N = 4, 5 is 0101 in binary. So:
             - flipendbits(5, 4) returns 12, since [0]10[1] goes to [1]10[0], which is 12.
    
    Parameters
    ----------
    number : int
        The number in question.
    N : int 
        Length of the binary representation of `number` to be considered.
        
    Returns
    -------
    new_number : int 
        Integer that results from flipping the first and last bits.
    """
    # To flip the first bit of "number", we XOR it with 2^(N-1).
    new_number = number ^ 2**(N-1)
    # To flip the last bit of "number", we XOR it with 1.
    new_number = new_number ^ 1
    return new_number

def flipanybits(number, position1, position2, N):
    """ 
    This function is to improve readability of later code.
    Flips `position1`-th and `position2`-th bits from the left in the `N` bit binary 
    representation of `number`.
    Example: For N = 4, 5 is 0101 in binary. So:
             - flipanybits(5, 0, 1, 4) returns 9, since [01]01 goes to [10]01, which is 9.
             - flipanybits(5, 1, 3, 4) returns 0, since 0[1]0[1] goes to 0[0]0[0], which is 0.  
             - flipanybits(5, 2, 2, 4) returns 5, since 01[0]1 goes to 01[0]1, which is 5.
             - flipanybits(5, 1, 4, 4) returns 9, since [01]01 goes to [10]01, which is 9.
             - flipanybits(5, 2, 8, 4) returns 15, since [0]1[0]1 goes to [1]1[1]1, which is 15.
            
    Parameters
    ----------
    number : int
        The number in question.
    position1 : int
        Position in `number`'s binary representation. Must be less than `N`.
        Leftmost bit is considered the 0th position.
    position2 : int
        Position in `number`'s binary representation.
        Leftmost bit is considered the 0th position.
        In case of overflow (>= `N`), this number is used modulo N.
    N : int 
        Length of the binary representation of `number` to be considered.
        
    Returns
    -------
    new_number : int 
        Integer that results from flipping the `position1`-th and `position2`+1-th bits from the left.
    """
    ### Flip "position1"-th and "position2"-th bits from the left of "number".
    # To flip the "position1"-th bit of "number", we XOR it with 2^(N-1) rightshifted "position1" times.
    new_number = number ^ (2**(N-1) >> position1)
    # To flip the "position2"-th bit of "number", we XOR it with 2^(N-1) rightshifted "position2" times.
    new_number = new_number ^ (2**(N-1) >> (position2 % N))
    return new_number


#######################################################################################################
#######################################################################################################
########################################## Hamiltonians ###############################################
#######################################################################################################
#######################################################################################################

def H_xx(eta_arr, closed = True):
    """ 
    This function implements H|eta⟩, for the XX Hamiltonian, without the -J/8 factor.
    
    Parameters
    ----------
    eta_arr : array_like
        A 1-D NumPy ndarray for |eta⟩. Indices represent basis vectors, elements are their coefficients.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
        
    Returns
    -------
    Heta_arr : array_like
        A 1-D NumPy ndarray for H|eta⟩. Indices represent basis vectors, elements are their coefficient.
        (!)Does not include the -J/8 factor from the Hamiltonian.
    """
    # Get number of spins in chain.
    N = len(eta_arr).bit_length() - 1
    
    # Set arr for H|eta⟩ with zeros.
    Heta_arr = np.zeros(len(eta_arr)).astype(complex)
    
    ### Loop for open chain terms.
    # Apply the (open chain terms) Hamiltonian to each basis vector, individually, and store results in Heta_arr.
    for vector, coefficient in enumerate(eta_arr):        
        
        # This for loop goes from site = 0 to site = N-2. For the expression of H_{xx}, i = site + 1.
        for site in range(0, N-1):            
            # Application of terms in the Hamiltonian to basis vectors will be null, unless neighbouring sites
            # are in different states.
            if bitset(vector, site, N) != bitset(vector, site + 1, N):                
                # sigma_{i}^(+) sigma_{i+1}^(-) |state⟩ is null, unless |state⟩ ~ |1⟩_{i} |0⟩_{i+1}. In that case:
                # sigma_{i}^(+) sigma_{i+1}^(-) |1⟩_{i} |0⟩_{i+1} = 4 |0⟩_{i} |1⟩_{i+1}.
                    
                # sigma_{i}^(-) sigma_{i+1}^(+) |state⟩ is null, unless |state⟩ ~ |0⟩_{i} |1⟩_{i+1}. In that case:
                # sigma_{i}^(-) sigma_{i+1}^(+) |0⟩_{i} |1⟩_{i+1} = 4 |1⟩_{i} |0⟩_{i+1}.
                
                # Since the flipbits function flips bits no matter what, and both of these operators give the same
                # result in terms of the coefficient, we don't need to differentiate between them.
                # We just get the resulting vector and multiply the coefficient by 4
                
                # Add calculated coefficient to array.
                Heta_arr[flipbits(vector, site, N)] += 4*coefficient
            
    ### Loop for closed chain terms, if necessary.
    # Apply sigma_{N}^(+) sigma_{1}^(-) and sigma_{N}^(-) sigma_{1}^(+) to |eta⟩.
    if closed:
        # Apply the (closed chain terms) Hamiltonian to each basis vector, individually, and store results in Heta_arr.
        for vector, coefficient in enumerate(eta_arr):
            # Check if sites are in different states.
            if bitset(vector, N-1, N) != bitset(vector, 0, N):
                # Flip bits and multiply coefficient by 4.                
                # Add calculated coefficient to array.
                Heta_arr[flipendbits(vector, N)] += 4*coefficient
                
    return Heta_arr

def H_ising(eta_arr, closed = True):
    """ 
    This function implements H|eta⟩, for the Ising Hamiltonian, without the -J/8 factor.
    
    Parameters
    ----------
    eta_arr : array_like
        A 1-D NumPy ndarray for |eta⟩. Indices represent basis vectors, elements are their coefficients.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
        
    Returns
    -------
    Heta_arr : array_like
        A 1-D NumPy ndarray for H|eta⟩. Indices represent basis vectors, elements are their coefficient.
        (!)Does not include the -J/8 factor from the Hamiltonian.
    """
    # Get number of spins in chain.
    N = len(eta_arr).bit_length() - 1
    
    # Set array for H|eta⟩ with zeros.
    Heta_arr = np.zeros(len(eta_arr)).astype(complex)
    
    ### Loop for open chain terms.
    # Apply the (open chain terms) Hamiltonian to each basis vector, individually, and store results in Heta_arr.
    for vector, coefficient in enumerate(eta_arr):        
        
        # This for loop goes from site = 0 to site = N-2. For the expression of H_{ising}, i = site + 1.
        for site in range(0, N-1):            
            # Application of terms in the Hamiltonian to basis vectors will never be null. Moreover, given the
            # expression for the Ising Hamiltonian, for each vector, and for each site, one, and only one, of the
            # sigma combinations will act.
            # The result will always be to multiply the coefficient by 4 and flip neighbouring bits.
            # But, since we have a 1/2 factor, we just multiply by 2.

            # Add calculated coefficient to array.
            Heta_arr[flipbits(vector, site, N)] += 2*coefficient
            
    ### Loop for closed chain terms, if necessary.
    # Apply closed chain terms to |eta⟩.
    if closed:
        # Apply the (closed chain terms) Hamiltonian to each basis vector, individually, and store results in Heta_arr.
        for vector, coefficient in enumerate(eta_arr):
            # Flip bits and multiply coefficient by 2.

            # Add calculated coefficient to array.
            Heta_arr[flipendbits(vector, N)] += 2*coefficient
                
    return Heta_arr

def H_xy(eta_arr, closed = True, gamma = 0.5):
    """ 
    This function implements H|eta⟩, for the XY Hamiltonian, without the -J/8 factor.
    
    Parameters
    ----------
    eta_arr : array_like
        A 1-D NumPy ndarray for |eta⟩. Indices represent basis vectors, elements are their coefficients.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    gamma :  float, optional
        Degree of anisotropy. Must be in the interval [0, 1]. Default is 0.5.
        
    Returns
    -------
    Heta_arr : array_like
        A 1-D NumPy ndarray for H|eta⟩. Indices represent basis vectors, elements are their coefficient.
        (!)Does not include the -J/8 factor from the Hamiltonian.
    """
    # Get number of spins in chain.
    N = len(eta_arr).bit_length() - 1
    
    # Set array for H|eta⟩ with zeros.
    Heta_arr = np.zeros(len(eta_arr)).astype(complex)
    
    ### Loop for open chain terms.
    # Apply the (open chain terms) Hamiltonian to each basis vector, individually, and store results in Heta_arr.
    for vector, coefficient in enumerate(eta_arr):        
        
        # This for loop goes from site = 0 to site = N-2. For the expression of H_{xy}, i = site + 1.
        for site in range(0, N-1):            
            # Application of terms in the Hamiltonian to basis vectors will never be null. Moreover, given the
            # expression for the XY Hamiltonian, for each vector, and for each site, one, and only one, of the
            # sigma combinations will act.
            # The result will always be to multiply the coefficient by 4 and flip neighbouring bits if neighbouring
            # sites are in different states, and by 4*gamma if they are in the same state.
            
            # If neighbouring sites are in different states.
            if bitset(vector, site, N) != bitset(vector, site + 1, N): 
                # Add calculated coefficient to array.
                Heta_arr[flipbits(vector, site, N)] += 4*coefficient
            
            # If they are in the same state.
            else:
                # Add calculated coefficient to array.
                Heta_arr[flipbits(vector, site, N)] += 4*gamma*coefficient
            
    ### Loop for closed chain terms, if necessary.
    # Apply closed chain terms to |eta⟩.
    if closed:
        # Apply the (closed chain terms) Hamiltonian to each basis vector, individually, and store results in Heta_arr.
        for vector, coefficient in enumerate(eta_arr):
            # Check if sites are in different states.
            if bitset(vector, N-1, N) != bitset(vector, 0, N):
                # Flip bits and multiply coefficient by 4.                
                # Add calculated coefficient to array.
                Heta_arr[flipendbits(vector, N)] += 4*coefficient
            else:
                # Flip bits and multiply coefficient by 4*gamma.                
                # Add calculated coefficient to array.
                Heta_arr[flipendbits(vector, N)] += 4*gamma*coefficient
                
    return Heta_arr

def H_xy_f(eta_arr, closed = True, gamma = 0.5, field = 0):
    """ 
    This function implements H|eta⟩, for the XY Hamiltonian with an external transverse magnetic
    field, without the -J/8 factor.
    
    Parameters
    ----------
    eta_arr : array_like
        A 1-D NumPy ndarray for |eta⟩. Indices represent basis vectors, elements are their coefficients.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    gamma :  float, optional
        Degree of anisotropy. Must be in the interval [0, 1]. Default is 0.5.
    field : float, optional
        Strength of the external magnetic field relative to the exchange interaction (h/J). Default is 0.
    
    Returns
    -------
    Heta_arr : array_like
        A 1-D NumPy ndarray for H|eta⟩. Indices represent basis vectors, elements are their coefficient.
        (!)Does not include the -J/8 factor from the Hamiltonian.
    """
    # Get number of spins in chain.
    N = len(eta_arr).bit_length() - 1
    
    # Set array for H|eta⟩ with zeros.
    Heta_arr = np.zeros(len(eta_arr)).astype(complex)
    
    ### Loop for open chain terms and field terms.
    # Apply the (open chain and field terms) Hamiltonian to each basis vector, individually, and store results 
    # in Heta_arr.
    for vector, coefficient in enumerate(eta_arr):        
        ### Open chain terms.
        # This for loop goes from site = 0 to site = N-2. For the expression of H_{xyf}, i = site + 1.
        for site in range(0, N-1):            
            # Application of terms in the Hamiltonian to basis vectors will never be null. Moreover, given the
            # expression for the XY Hamiltonian, for each vector, and for each site, one, and only one, of the
            # sigma combinations will act.
            # The result will always be to multiply the coefficient by 4 and flip neighbouring bits if neighbouring
            # sites are in different states, and by 4*gamma if they are in the same state.
            
            # If neighbouring sites are in different states.
            if bitset(vector, site, N) != bitset(vector, site + 1, N): 
                # Add calculated coefficient to array.
                Heta_arr[flipbits(vector, site, N)] += 4*coefficient
            
            # If they are in the same state.
            else:
                # Add calculated coefficient to array.
                Heta_arr[flipbits(vector, site, N)] += 4*gamma*coefficient
            
        ### Field terms.
        # The overall effect of sum_{j} sigma_{j}^{z} on a product state is to multiply it by (n_up - n_down), where
        # n_up is the number of sites in a spin-up state, and n_down is the number of sites in a spin-down state.
        # sigma_{i}^(z) |state⟩ = |state⟩  if |state⟩  ~ |0⟩_{i}
        # sigma_{i}^(z) |state⟩ = -|state⟩ if |state⟩  ~ |1⟩_{i}
        # Note that n_down is N - n_up. We can count the number of set bits (number of sites in a spin-down[!] state)
        # in the binary representation of a number using the count method for strings (which seems faster than Kernighan's
        # algorithm for small numbers).
        Heta_arr[vector] += 4*field*(N - 2*bin(vector).count("1"))*coefficient
        
            
            
    ### Loop for closed chain terms, if necessary.
    # Apply closed chain terms to |eta⟩.
    if closed:
        # Apply the (closed chain terms) Hamiltonian to each basis vector, individually, and store results in Heta_arr.
        for vector, coefficient in enumerate(eta_arr):
            # Check if sites are in different states.
            if bitset(vector, N-1, N) != bitset(vector, 0, N):
                # Flip bits and multiply coefficient by 4.                
                # Add calculated coefficient to array.
                Heta_arr[flipendbits(vector, N)] += 4*coefficient
            else:
                # Flip bits and multiply coefficient by 4*gamma.                
                # Add calculated coefficient to array.
                Heta_arr[flipendbits(vector, N)] += 4*gamma*coefficient
                
    return Heta_arr

def H_fne_ising_c(eta_arr, closed = True, coordination = 2, alpha = 2, field = 0):
    """ 
    This function implements H|eta⟩ for the few-neighbor extended Ising Hamiltonian, without the -J/8 
    factor, for periodic boundary conditions.
    
    WARNING: open chain (`closed` = False) not implemented in this function. This was done in another 
    function (`H_fne_ising_o`) to avoid unnecessary checks when handling boundary terms. Argument is 
    still present to allow expectation value functions to work properly.
    
    Parameters
    ----------
    eta_arr : array_like
        A 1-D NumPy ndarray for |eta⟩. Indices represent basis vectors, elements are their coefficients.
    closed : bool, optional
        Must be True(!). The spin chain has periodic boundary conditions.
    coordination : int, optional
        The coordination number, which should satisfy 1 < `coordination` < N-1, where N is the length of 
        the chain. It's the number of interacting neighbors to the right. This number should be chosen with
        extra care if the chain doesn't have periodic boundary conditions.
    alpha : float, optional
        The tunning parameter, which dictates the interaction strength between neighbors. The non local
        regime is for `alpha` < 1, the quasilocal regime is when 1 < `alpha` < 2. Else, local regime.
    field : float, optional
        Strength of the external magnetic field relative to the exchange interaction (h/J). Default is 0.
        Due to the way the constant A is defined, there's a critical point at 2.
        
    Returns
    -------
    Heta_arr : array_like
        A 1-D NumPy ndarray for H|eta⟩. Indices represent basis vectors, elements are their coefficient.
        (!)Does not include the -J/8 factor from the Hamiltonian.
    """
    ### Set up.
    # Store terms 1/r^{alpha} so as to calculate them only once.
    inverse_arr = np.array([1/((r+1)**alpha) for r in range(coordination)])
    
    # Multiply 1/r^{alpha} terms by 2/A.
    inverse_arr *= (2/inverse_arr.sum())
    
    # Get number of spins in chain.
    N = len(eta_arr).bit_length() - 1
    
    # Set array for H|eta⟩ with zeros.
    Heta_arr = np.zeros(len(eta_arr)).astype(complex)
    
    ### Loop for open chain terms.
    # Apply the (open chain terms) Hamiltonian to each basis vector, individually, and store results in Heta_arr.
    for vector, coefficient in enumerate(eta_arr):
        ### Field terms.
        # The overall effect of sum_{j} sigma_{j}^{z} on a product state is to multiply it by (n_up - n_down), where
        # n_up is the number of sites in a spin-up state, and n_down is the number of sites in a spin-down state.
        # sigma_{i}^(z) |state⟩ = |state⟩  if |state⟩  ~ |0⟩_{i}
        # sigma_{i}^(z) |state⟩ = -|state⟩ if |state⟩  ~ |1⟩_{i}
        # Note that n_down is N - n_up. We can count the number of set bits (number of sites in a spin-down[!] state)
        # in the binary representation of a number using the count method for strings (which seems faster than Kernighan's
        # algorithm for small numbers).
        Heta_arr[vector] += 4*field*(N - 2*bin(vector).count("1"))*coefficient
        
        ### Chain Terms
        for site in range(0, N):            
            # Again, for the specific combinations of ladder operators present in the Hamiltonian, their application
            # to basis vectors will never be null. Moreover, one, and only one, of the those combinationsn will act.
            # The result will always be to multiply the coefficient by 4 and flip corresponding bits.

            # Add calculated coefficient to array for r = 1.
            Heta_arr[flipanybits(vector, site, site+1, N)] += 4*coefficient*inverse_arr[0]
            
            # Loop for other r values (coordination number is at least 2).
            for r_code in range(1, coordination):
                # First, the application of the prod_{l} sigma_{l}^{z} term on a product state is to multiply it by
                # (n_up - n_down)_{l}, where the subindex l indicates that we only count sites strictly between i and i+r, 
                # with r = r_code + 1.
                # Get the number of number of sites in a spin-down state strictly between i (site) and i+r.
                n_down = 0
                for distance in range(1, r_code+1):
                    # We use that bool is a subclass of int, where True = 1 and False = 0.
                    n_down += bitset(vector, site+distance, N)
                
                # Apply the ladder operators on appropriate sites and add calculated coefficient to array.
                # Since the total number of sites between i and i+r is given by r_code, we can calculate 
                # (n_up - n_down)_{l} as r_code - 2*n_down.
                Heta_arr[flipanybits(vector, site, site+r_code+1, N)] += 4*coefficient*inverse_arr[r_code]*(r_code - 2*n_down)
                
    return Heta_arr

def H_fne_ising_o(eta_arr, closed = False, coordination = 2, alpha = 2, field = 0):
    """ 
    This function implements H|eta⟩ for the few-neighbor extended Ising Hamiltonian, without the -J/8 
    factor, for free ends.
    
    WARNING: closed chain (`closed` = True) not implemented in this function. This was done in another 
    function (`H_fne_ising_c`) to avoid unnecessary checks when handling boundary terms. Argument is 
    still present to allow expectation value functions to work properly.
    
    Parameters
    ----------
    eta_arr : array_like
        A 1-D NumPy ndarray for |eta⟩. Indices represent basis vectors, elements are their coefficients.
    closed : bool, optional
        Must be False(!). The spin chain has free ends.
    coordination : int, optional
        The coordination number, which should satisfy 1 < `coordination` < N-1, where N is the length of 
        the chain. It's the number of interacting neighbors to the right. This number should be chosen with
        extra care if the chain doesn't have periodic boundary conditions.
    alpha : float, optional
        The tunning parameter, which dictates the interaction strength between neighbors. The non local
        regime is for `alpha` < 1, the quasilocal regime is when 1 < `alpha` < 2. Else, local regime.
    field : float, optional
        Strength of the external magnetic field relative to the exchange interaction (h/J). Default is 0.
        Due to the way the constant A is defined, there's a critical point at 2.
        
    Returns
    -------
    Heta_arr : array_like
        A 1-D NumPy ndarray for H|eta⟩. Indices represent basis vectors, elements are their coefficient.
        (!)Does not include the -J/8 factor from the Hamiltonian.
    """
    ### Set up.
    # Store terms 1/r^{alpha} so as to calculate them only once.
    inverse_arr = np.array([1/((r+1)**alpha) for r in range(coordination)])
    
    # Multiply 1/r^{alpha} terms by 2/A.
    inverse_arr *= (2/inverse_arr.sum())
    
    # Get number of spins in chain.
    N = len(eta_arr).bit_length() - 1
    
    # Set array for H|eta⟩ with zeros.
    Heta_arr = np.zeros(len(eta_arr)).astype(complex)
    
    ### Loop for open chain terms.
    # Apply the (open chain terms) Hamiltonian to each basis vector, individually, and store results in Heta_arr.
    for vector, coefficient in enumerate(eta_arr):
        ### Field terms.
        # The overall effect of sum_{j} sigma_{j}^{z} on a product state is to multiply it by (n_up - n_down), where
        # n_up is the number of sites in a spin-up state, and n_down is the number of sites in a spin-down state.
        # sigma_{i}^(z) |state⟩ = |state⟩  if |state⟩  ~ |0⟩_{i}
        # sigma_{i}^(z) |state⟩ = -|state⟩ if |state⟩  ~ |1⟩_{i}
        # Note that n_down is N - n_up. We can count the number of set bits (number of sites in a spin-down[!] state)
        # in the binary representation of a number using the count method for strings (which seems faster than Kernighan's
        # algorithm for small numbers).
        Heta_arr[vector] += 4*field*(N - 2*bin(vector).count("1"))*coefficient
        
        ### Chain Terms
        # Loop for r=1.
        for site in range(0, N-1):            
            # Again, for the specific combinations of ladder operators present in the Hamiltonian, their application
            # to basis vectors will never be null. Moreover, one, and only one, of the those combinationsn will act.
            # The result will always be to multiply the coefficient by 4 and flip corresponding bits.

            # Add calculated coefficient to array for r = 1.
            Heta_arr[flipanybits(vector, site, site+1, N)] += 4*coefficient*inverse_arr[0]
            
        # Loop for other r values (coordination number is at least 2).
        for r_code in range(1, coordination):
            for site in range(0, N-1-r_code):
                # First, the application of the prod_{l} sigma_{l}^{z} term on a product state is to multiply it by
                # (n_up - n_down)_{l}, where the subindex l indicates that we only count sites strictly between i and i+r, 
                # with r = r_code + 1.
                # Get the number of number of sites in a spin-down state strictly between i (site) and i+r.
                n_down = 0
                for distance in range(1, r_code+1):
                    # We use that bool is a subclass of int, where True = 1 and False = 0.
                    n_down += bitset(vector, site+distance, N)

                # Apply the ladder operators on appropriate sites and add calculated coefficient to array.
                # Since the total number of sites between i and i+r is given by r_code, we can calculate 
                # (n_up - n_down)_{l} as r_code - 2*n_down.
                Heta_arr[flipanybits(vector, site, site+r_code+1, N)] += 4*coefficient*inverse_arr[r_code]*(r_code - 2*n_down)
                
    return Heta_arr

def H_ssh(eta_arr, closed = True, relative_J = 1):
    """ 
    This function implements H|eta⟩, for the SSH Hamiltonian, without the -J_w/8 factor.
    
    Parameters
    ----------
    eta_arr : array_like
        A 1-D NumPy ndarray for |eta⟩. Indices represent basis vectors, elements are their coefficients.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    relative_J : float, optional
        Strong exchange interaction relative to weak exchange ingeraction (J_s/J_w). Default is 1.
    
    
    Returns
    -------
    Heta_arr : array_like
        A 1-D NumPy ndarray for H|eta⟩. Indices represent basis vectors, elements are their coefficient.
        (!)Does not include the -J/8 factor from the Hamiltonian.
    """
    # Get number of spins in chain.
    N = len(eta_arr).bit_length() - 1
    
    # Set arr for H|eta⟩ with zeros.
    Heta_arr = np.zeros(len(eta_arr)).astype(complex)
    
    ### Loop for open chain terms.
    # Apply the (open chain terms) Hamiltonian to each basis vector, individually, and store results in Heta_arr.
    for vector, coefficient in enumerate(eta_arr):        
        
        # This for loop goes from site = 0 to site = N-2. For the expression of H_{SSH}, i = site + 1.
        for site in range(0, N-1):            
            # Application of terms in the Hamiltonian to basis vectors will be null, unless neighbouring sites
            # are in different states.
            if bitset(vector, site, N) != bitset(vector, site + 1, N):                
                # sigma_{i}^(+) sigma_{i+1}^(-) |state⟩ is null, unless |state⟩ ~ |1⟩_{i} |0⟩_{i+1}. In that case:
                # sigma_{i}^(+) sigma_{i+1}^(-) |1⟩_{i} |0⟩_{i+1} = 4 |0⟩_{i} |1⟩_{i+1}.
                    
                # sigma_{i}^(-) sigma_{i+1}^(+) |state⟩ is null, unless |state⟩ ~ |0⟩_{i} |1⟩_{i+1}. In that case:
                # sigma_{i}^(-) sigma_{i+1}^(+) |0⟩_{i} |1⟩_{i+1} = 4 |1⟩_{i} |0⟩_{i+1}.
                
                # Since the flipbits function flips bits no matter what, and both of these operators give the same
                # result in terms of the coefficient, we don't need to differentiate between them.
                # We just get the resulting vector and multiply the coefficient by 4
                
                # The factor (1 + ((site+1) % 2)*(relative_J-1)) is 1 if site is odd (i even)
                # and relative_J if site is even (i odd)
                
                # Add calculated coefficient to array.
                Heta_arr[flipbits(vector, site, N)] += 4*coefficient*(1 + ((site+1) % 2)*(relative_J-1))
            
    ### Loop for closed chain terms, if necessary.
    # Apply sigma_{N}^(+) sigma_{1}^(-) and sigma_{N}^(-) sigma_{1}^(+) to |eta⟩.
    if closed:
        # Apply the (closed chain terms) Hamiltonian to each basis vector, individually, and store results in Heta_arr.
        for vector, coefficient in enumerate(eta_arr):
            # Check if sites are in different states.
            if bitset(vector, N-1, N) != bitset(vector, 0, N):
                # Flip bits and multiply coefficient by 4.                
                # Add calculated coefficient to array.
                Heta_arr[flipendbits(vector, N)] += 4*coefficient*(1 + ((N) % 2)*(relative_J-1))
                
    return Heta_arr

#######################################################################################################
#######################################################################################################
####################################### Expectation Value #############################################
#######################################################################################################
#######################################################################################################

def exp_val_nsym(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    without making any assumptions regarding symmetries.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = int(N*(N-1)/2), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1,2}, varphi_{1,3}, ..., varphi_{1,N}, varphi_{2,3}, ..., varphi_{N-1, N}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformation_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + n_deform:]
   
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and phase gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformation_arr[site-1, 2*superpos_idx] + deformation_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, no symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            counter, offset = 1, 0
            for step in range(pair[0] - 1):
                offset += (N - counter)
                counter += 1

            phase_idx = offset + (pair[1] - pair[0] - 1)

            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)
            
            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)

    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)
    
    # Return expectation value, adding the -J/8 factor
    return J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))
    
    ##########################################################################################
    # Note on taking Re(⟨eta|H|eta⟩)/Re(⟨eta|eta⟩) instead of Re(⟨eta|H|eta⟩ / ⟨eta|eta⟩).      #
    # While it's clear that for complex numbers Re(a/b) != Re(a)/Re(b), both ⟨eta|H|eta⟩      #
    # and ⟨eta|eta⟩ are supposed to be real numbers but since ⟨eta|eta⟩ and ⟨eta|H|eta⟩ could  #
    # not be pure real numbers due to precision errors, we take their real part individually.#
    ##########################################################################################

def exp_val_nsym_ad(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    without making any assumptions regarding symmetries using an alternate deformation approach.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = int(N*(N-1)/2), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1,2}, varphi_{1,3}, ..., varphi_{1,N}, varphi_{2,3}, ..., varphi_{N-1, N}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformationd_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    deformationf_arr = params[n_phases + n_unit + n_deform: n_phases + n_unit + 2*n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + 2*n_deform:]
   
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        # Do the same for |0⟩.
        zeroes = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '0']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationd_arr[site-1, 2*superpos_idx] + deformationd_arr[site-1, 2*superpos_idx+1]*1j
            
            # Do the same for |0⟩.
            for site in zeroes:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationf_arr[site-1, 2*superpos_idx] + deformationf_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            counter, offset = 1, 0
            for step in range(pair[0] - 1):
                offset += (N - counter)
                counter += 1

            phase_idx = offset + (pair[1] - pair[0] - 1)

            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)
            
            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)

    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)
    
    # Return expectation value, adding the -J/8 factor
    return J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))

def exp_val_nsym_ad_cocr(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, const = 1, pairs = [(1, 2)], *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    without making any assumptions regarding symmetries using an alternate deformation approach 
    with cross correlations as a penalty function.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    const : float, optional
        Constant for the penalty function.
    pairs : list, optional
        List of pairs of sites as tuples for cross correlations.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = int(N*(N-1)/2), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1,2}, varphi_{1,3}, ..., varphi_{1,N}, varphi_{2,3}, ..., varphi_{N-1, N}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformationd_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    deformationf_arr = params[n_phases + n_unit + n_deform: n_phases + n_unit + 2*n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + 2*n_deform:]
   
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        # Do the same for |0⟩.
        zeroes = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '0']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationd_arr[site-1, 2*superpos_idx] + deformationd_arr[site-1, 2*superpos_idx+1]*1j
            
            # Do the same for |0⟩.
            for site in zeroes:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationf_arr[site-1, 2*superpos_idx] + deformationf_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            counter, offset = 1, 0
            for step in range(pair[0] - 1):
                offset += (N - counter)
                counter += 1

            phase_idx = offset + (pair[1] - pair[0] - 1)

            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)
            
            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)

    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)

    ### Calculate expectation value, adding the -J/8 factor.
    ev = J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))
    
    ### Correlation functions.
    cross_corr = 0
    for pair in pairs:
        # Set array to store sigma_{spin_1}^(x) sigma_{spin_2}^(y) |eta⟩ 
        temp_eta =  np.zeros(len(eta_arr)).astype(complex)
        for vector, coefficient in enumerate(eta_arr): 
            # For every pair of sites, one, and only one, sigma combination
            # will act for S^{x} and S^{y}.

            # If 2nd spin is down.
            if bitset(vector, pair[1]-1, N): 
                # Add calculated coefficient to array. 
                temp_eta[flipanybits(vector, pair[0]-1, pair[1]-1, N)] += coefficient

            # If 2nd spin is up.
            else:
                # Add calculated coefficient to array.
                temp_eta[flipanybits(vector, pair[0]-1, pair[1]-1, N)] -= coefficient


        # Calculate correlation and add it.
        corr = np.real(1j*np.vdot(eta_arr, temp_eta)) / np.real(-4*np.vdot(eta_arr, eta_arr))
        cross_corr += corr*corr
        
    # Return expectation value plus penalty
    return ev + const*cross_corr

def exp_val_psym(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with partial phase symmetrization.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = int(N-1), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformation_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformation_arr[site-1, 2*superpos_idx] + deformation_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            phase_idx = pair[1] - pair[0] - 1

            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)
        
    # Return expectation value, adding the -J/8 factor
    return J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))

def exp_val_psym_c(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with partial phase symmetrization. Unitaries are parametrized by a Cayley transform.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = int(N-1), 4*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformation_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformation_arr[site-1, 2*superpos_idx] + deformation_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            phase_idx = pair[1] - pair[0] - 1

            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get a, b, c and d parameters.
            a = unitaries_arr[4*site]
            b = unitaries_arr[4*site + 1]
            c = unitaries_arr[4*site + 2]
            d = unitaries_arr[4*site + 3]

            # Calculate factor.
            factor = a*b - 1 - c*c - d*d + a*1j + b*1j
            factor = factor / ((a*b - 1 - c*c - d*d)**2 + (a+b)**2)

            # Set matrix elements.
            u11 = -a*b - 1 + c*c + d*d + a*1j - b*1j
            u12 = -2*d + 2*c*1j
            u21 = 2*d + 2*c*1j
            u22 = -a*b - 1 + c*c + d*d - a*1j + b*1j

            # Add matrix to list.
            unitaries_list.append(factor*np.array([[u11, u12], [u21, u22]]))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)
        
    # Return expectation value, adding the -J/8 factor
    return J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))

def exp_val_psym_coxy(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, const = 1, pairs = [(1, 2)], *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with partial phase symmetrization with the equality of correlations on the plane being used
    as a penalty function.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    const : float, optional
        Constant for the penalty function.
    pairs : list, optional
        List of pairs of sites as tuples for correlation functions.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = int(N-1), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformation_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformation_arr[site-1, 2*superpos_idx] + deformation_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            phase_idx = pair[1] - pair[0] - 1

            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)
    
    ### Calculate expectation value, adding the -J/8 factor.
    ev = J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))
    
    ### Correlation functions.
    corr_dif = 0
    for pair in pairs:
        # Set arrays to store sigma_{spin_1}^(l) sigma_{spin_2}^(l) |eta⟩ 
        temp_eta_x = np.zeros(len(eta_arr)).astype(complex)
        temp_eta_y = np.zeros(len(eta_arr)).astype(complex)
        for vector, coefficient in enumerate(eta_arr): 
            # For every pair of sites, one, and only one, sigma combination
            # will act for S^{x} and S^{y}.

            # If neighbouring sites are in different states.
            if bitset(vector, pair[0]-1, N) != bitset(vector, pair[1]-1, N): 
                # Add calculated coefficient to array. 
                temp_eta_x[flipanybits(vector, pair[0]-1, pair[1]-1, N)] += coefficient
                temp_eta_y[flipanybits(vector, pair[0]-1, pair[1]-1, N)] -= coefficient

            # If they are in the same state.
            else:
                # Add calculated coefficient to array.
                temp_eta_x[flipanybits(vector, pair[0]-1, pair[1]-1, N)] += coefficient
                temp_eta_y[flipanybits(vector, pair[0]-1, pair[1]-1, N)] += coefficient


        # Calculate correlation and add it.
        modulus = np.real(np.vdot(eta_arr, eta_arr))
        corr_x = np.real(np.vdot(eta_arr, temp_eta_x)) / (4*modulus)
        corr_y = np.real(np.vdot(eta_arr, temp_eta_y)) / (-4*modulus)
        corr_dif += (corr_x - corr_y)*(corr_x - corr_y)
        
    # Return expectation value plus penalty
    return ev + const*corr_dif

def exp_val_psym_cocr(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, const = 1, pairs = [(1, 2)], *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with partial phase symmetrization with cross correlations as a penalty function.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    const : float, optional
        Constant for the penalty function.
    pairs : list, optional
        List of pairs of sites as tuples for cross correlations.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = int(N-1), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformation_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformation_arr[site-1, 2*superpos_idx] + deformation_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            phase_idx = pair[1] - pair[0] - 1

            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)
    
    ### Calculate expectation value, adding the -J/8 factor.
    ev = J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))
    
    ### Correlation functions.
    cross_corr = 0
    for pair in pairs:
        # Set array to store sigma_{spin_1}^(x) sigma_{spin_2}^(y) |eta⟩ 
        temp_eta =  np.zeros(len(eta_arr)).astype(complex)
        for vector, coefficient in enumerate(eta_arr): 
            # For every pair of sites, one, and only one, sigma combination
            # will act for S^{x} and S^{y}.

            # If 2nd spin is down.
            if bitset(vector, pair[1]-1, N): 
                # Add calculated coefficient to array. 
                temp_eta[flipanybits(vector, pair[0]-1, pair[1]-1, N)] += coefficient

            # If 2nd spin is up.
            else:
                # Add calculated coefficient to array.
                temp_eta[flipanybits(vector, pair[0]-1, pair[1]-1, N)] -= coefficient


        # Calculate correlation and add it.
        corr = np.real(1j*np.vdot(eta_arr, temp_eta)) / np.real(-4*np.vdot(eta_arr, eta_arr))
        cross_corr += corr*corr
        
    # Return expectation value plus penalty
    return ev + const*cross_corr

def exp_val_psym_cocrxy(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, c1 = 1, c2 = 1, pairs = [(1, 2)], *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with partial phase symmetrization with the equality of correlations on the plane and cross
    correlations being used as a penalty function.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    c1 : float, optional
        Constant for the correlation penalty function.
    c2 : float, optional
        Constant for the cross correlation penalty function.
    pairs : list, optional
        List of pairs of sites as tuples for correlations.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = int(N-1), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformation_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformation_arr[site-1, 2*superpos_idx] + deformation_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            phase_idx = pair[1] - pair[0] - 1

            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)
    
    ### Calculate expectation value, adding the -J/8 factor.
    ev = J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))
    
    ### Correlation functions.
    corr = 0
    for pair in pairs:
        # Set arrays to store sigma_{spin_1}^(l) sigma_{spin_2}^(l) |eta⟩ 
        temp_eta_x = np.zeros(len(eta_arr)).astype(complex)
        temp_eta_y = np.zeros(len(eta_arr)).astype(complex)
        temp_eta =  np.zeros(len(eta_arr)).astype(complex)
        for vector, coefficient in enumerate(eta_arr): 
            # For every pair of sites, one, and only one, sigma combination
            # will act for S^{x} and S^{y}.

            # If neighbouring sites are in different states.
            if bitset(vector, pair[0]-1, N) != bitset(vector, pair[1]-1, N): 
                # Add calculated coefficient to array. 
                temp_eta_x[flipanybits(vector, pair[0]-1, pair[1]-1, N)] += coefficient
                temp_eta_y[flipanybits(vector, pair[0]-1, pair[1]-1, N)] -= coefficient

            # If they are in the same state.
            else:
                # Add calculated coefficient to array.
                temp_eta_x[flipanybits(vector, pair[0]-1, pair[1]-1, N)] += coefficient
                temp_eta_y[flipanybits(vector, pair[0]-1, pair[1]-1, N)] += coefficient

            # If 2nd spin is down.
            if bitset(vector, pair[1]-1, N): 
                # Add calculated coefficient to array. 
                temp_eta[flipanybits(vector, pair[0]-1, pair[1]-1, N)] += coefficient

            # If 2nd spin is up.
            else:
                # Add calculated coefficient to array.
                temp_eta[flipanybits(vector, pair[0]-1, pair[1]-1, N)] -= coefficient
        
        # Calculate correlation and add it.
        modulus = 4*np.real(np.vdot(eta_arr, eta_arr))
        corr_x = np.real(np.vdot(eta_arr, temp_eta_x)) / (modulus)
        corr_y = np.real(np.vdot(eta_arr, temp_eta_y)) / (-modulus)
        cross_corr = np.real(1j*np.vdot(eta_arr, temp_eta)) / (-modulus)
        corr += c1*(corr_x - corr_y)*(corr_x - corr_y) + c2*cross_corr*cross_corr
        
    # Return expectation value plus penalty
    return ev + corr

def exp_val_psym_cozz(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, const = 1, pairs = [(1, 2)], *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with partial phase symmetrization with the equality of correlations perpendicular to the plane 
    being used as a penalty function.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    const : float, optional
        Constant for the penalty function.
    pairs : list, optional
        List of pairs of sites as tuples for correlation functions.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = int(N-1), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformation_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformation_arr[site-1, 2*superpos_idx] + deformation_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            phase_idx = pair[1] - pair[0] - 1

            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)
    
    ### Calculate expectation value, adding the -J/8 factor.
    ev = J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))
    
    ### Correlation functions.
    corr_dif = 0
    for pair in pairs:
        # Set arrays to store sigma_{spin_1}^(l) sigma_{spin_2}^(l) |eta⟩ 
        temp_eta_1 = np.zeros(len(eta_arr)).astype(complex)
        temp_eta_2 = np.zeros(len(eta_arr)).astype(complex)
        for vector, coefficient in enumerate(eta_arr): 
            # For every pair of sites, one, and only one, sigma combination
            # will act for S^{x} and S^{y}.
            
            ### First pair.
            # If neighbouring sites are in different states.
            if bitset(vector, pair[0]-1, N) != bitset(vector, pair[1]-1, N): 
                # Add calculated coefficient to array. 
                temp_eta_1[vector] -= coefficient

            # If they are in the same state.
            else:
                # Add calculated coefficient to array.
                temp_eta_1[vector] += coefficient

            ### Second pair.
            # If neighbouring sites are in different states.
            if bitset(vector, pair[0], N) != bitset(vector, pair[1], N): 
                # Add calculated coefficient to array. 
                temp_eta_2[vector] -= coefficient

            # If they are in the same state.
            else:
                # Add calculated coefficient to array.
                temp_eta_2[vector] += coefficient

        # Calculate correlation and add it.
        modulus = np.real(np.vdot(eta_arr, eta_arr))
        corr_1 = np.real(np.vdot(eta_arr, temp_eta_1)) / (4*modulus)
        corr_2 = np.real(np.vdot(eta_arr, temp_eta_2)) / (4*modulus)
        corr_dif += (corr_1 - corr_2)*(corr_1 - corr_2)
    
    # Return expectation value plus penalty
    return ev + const*corr_dif

def exp_val_psym_ad(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with partial phase symmetrization using an alternate deformation approach.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = int(N-1), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformationd_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    deformationf_arr = params[n_phases + n_unit + n_deform: n_phases + n_unit + 2*n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + 2*n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        # Do the same for |0⟩.
        zeroes = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '0']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationd_arr[site-1, 2*superpos_idx] + deformationd_arr[site-1, 2*superpos_idx+1]*1j
            
            # Do the same for |0⟩.
            for site in zeroes:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationf_arr[site-1, 2*superpos_idx] + deformationf_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            phase_idx = pair[1] - pair[0] - 1

            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)
        
    # Return expectation value, adding the -J/8 factor
    return J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))

def exp_val_psym_ad_cocr(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, const = 1, pairs = [(1, 2)], *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with partial phase symmetrization using an alternate deformation approach with cross correlations 
    as a penalty function.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    const : float, optional
        Constant for the penalty function.
    pairs : list, optional
        List of pairs of sites as tuples for cross correlations.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = int(N-1), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformationd_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    deformationf_arr = params[n_phases + n_unit + n_deform: n_phases + n_unit + 2*n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + 2*n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        # Do the same for |0⟩.
        zeroes = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '0']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationd_arr[site-1, 2*superpos_idx] + deformationd_arr[site-1, 2*superpos_idx+1]*1j
            
            # Do the same for |0⟩.
            for site in zeroes:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationf_arr[site-1, 2*superpos_idx] + deformationf_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            phase_idx = pair[1] - pair[0] - 1

            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)

    ### Calculate expectation value, adding the -J/8 factor.
    ev = J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))
    
    ### Correlation functions.
    cross_corr = 0
    for pair in pairs:
        # Set array to store sigma_{spin_1}^(x) sigma_{spin_2}^(y) |eta⟩ 
        temp_eta =  np.zeros(len(eta_arr)).astype(complex)
        for vector, coefficient in enumerate(eta_arr): 
            # For every pair of sites, one, and only one, sigma combination
            # will act for S^{x} and S^{y}.

            # If 2nd spin is down.
            if bitset(vector, pair[1]-1, N): 
                # Add calculated coefficient to array. 
                temp_eta[flipanybits(vector, pair[0]-1, pair[1]-1, N)] += coefficient

            # If 2nd spin is up.
            else:
                # Add calculated coefficient to array.
                temp_eta[flipanybits(vector, pair[0]-1, pair[1]-1, N)] -= coefficient


        # Calculate correlation and add it.
        corr = np.real(1j*np.vdot(eta_arr, temp_eta)) / np.real(-4*np.vdot(eta_arr, eta_arr))
        cross_corr += corr*corr
        
    # Return expectation value plus penalty
    return ev + const*cross_corr

def exp_val_fsym(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with full phase symmetrization.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = math.floor(N/2), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformation_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformation_arr[site-1, 2*superpos_idx] + deformation_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, full symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            # For varphi_{i, j}, it will be the minimum of j-i-1 and (i+N)-j-1
            if pair[1] - pair[0] < pair[0] + N - pair[1]:
                phase_idx = pair[1] - pair[0] - 1
            else:
                phase_idx = pair[0] + N - pair[1] - 1
                
            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)
        
    # Return expectation value, adding the -J/8 factor
    return J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))

def exp_val_fsym_ad(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with full phase symmetrization using an alternate deformation approach.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = math.floor(N/2), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformationd_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    deformationf_arr = params[n_phases + n_unit + n_deform: n_phases + n_unit + 2*n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + 2*n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        # Do the same for |0⟩.
        zeroes = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '0']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationd_arr[site-1, 2*superpos_idx] + deformationd_arr[site-1, 2*superpos_idx+1]*1j
            
            # Do the same for |0⟩.
            for site in zeroes:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationf_arr[site-1, 2*superpos_idx] + deformationf_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            # For varphi_{i, j}, it will be the minimum of j-i-1 and (i+N)-j-1
            if pair[1] - pair[0] < pair[0] + N - pair[1]:
                phase_idx = pair[1] - pair[0] - 1
            else:
                phase_idx = pair[0] + N - pair[1] - 1
                
            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)
        
    # Return expectation value, adding the -J/8 factor
    return J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))

def exp_val_fsym_ad_cocr(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, const = 1, pairs = [(1, 2)], *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with full phase symmetrization using an alternate deformation approach with cross correlations 
    as a penalty function.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = math.floor(N/2), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformationd_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    deformationf_arr = params[n_phases + n_unit + n_deform: n_phases + n_unit + 2*n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + 2*n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        # Do the same for |0⟩.
        zeroes = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '0']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationd_arr[site-1, 2*superpos_idx] + deformationd_arr[site-1, 2*superpos_idx+1]*1j
            
            # Do the same for |0⟩.
            for site in zeroes:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationf_arr[site-1, 2*superpos_idx] + deformationf_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase.
            # For varphi_{i, j}, it will be the minimum of j-i-1 and (i+N)-j-1
            if pair[1] - pair[0] < pair[0] + N - pair[1]:
                phase_idx = pair[1] - pair[0] - 1
            else:
                phase_idx = pair[0] + N - pair[1] - 1
                
            # Add phase to the phase variable.
            phase += phases_arr[phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)

    ### Calculate expectation value, adding the -J/8 factor.
    ev = J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))
    
    ### Correlation functions.
    cross_corr = 0
    for pair in pairs:
        # Set array to store sigma_{spin_1}^(x) sigma_{spin_2}^(y) |eta⟩ 
        temp_eta =  np.zeros(len(eta_arr)).astype(complex)
        for vector, coefficient in enumerate(eta_arr): 
            # For every pair of sites, one, and only one, sigma combination
            # will act for S^{x} and S^{y}.

            # If 2nd spin is down.
            if bitset(vector, pair[1]-1, N): 
                # Add calculated coefficient to array. 
                temp_eta[flipanybits(vector, pair[0]-1, pair[1]-1, N)] += coefficient

            # If 2nd spin is up.
            else:
                # Add calculated coefficient to array.
                temp_eta[flipanybits(vector, pair[0]-1, pair[1]-1, N)] -= coefficient


        # Calculate correlation and add it.
        corr = np.real(1j*np.vdot(eta_arr, temp_eta)) / np.real(-4*np.vdot(eta_arr, eta_arr))
        cross_corr += corr*corr
        
    # Return expectation value plus penalty
    return ev + const*cross_corr

def exp_val_sshsym(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with SSH phase symmetrization.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = 2*math.floor(N/2), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # phi_{1}, phi_{2}, ..., phi_{N-1},
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1},
    phases_arr = params[:n_phases].reshape(2, math.floor(N/2))
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformation_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformation_arr[site-1, 2*superpos_idx] + deformation_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, full symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase, as with full symmetryzation.
            # For varphi_{i, j}, it will be the minimum of j-i-1 and (i+N)-j-1
            # This will also decide which of the two phases to use.
            # If j-i is the smaller, then i is the "first site". If that's not
            # the case, then j is the "first site".
            if pair[1] - pair[0] < pair[0] + N - pair[1]:
                phase_idx = pair[1] - pair[0] - 1
                
                # Add phase to the phase variable.
                phase += phases_arr[pair[0]%2][phase_idx]
                
            else:
                phase_idx = pair[0] + N - pair[1] - 1
                
                # Add phase to the phase variable.
                phase += phases_arr[pair[1]%2][phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)
        
    # Return expectation value, adding the -J/8 factor
    return J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))

def exp_val_sshsym_ad(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with SSH phase symmetrization using an alternate deformation approach.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = 2*math.floor(N/2), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # phi_{1}, phi_{2}, ..., phi_{N-1},
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1},
    phases_arr = params[:n_phases].reshape(2, math.floor(N/2))
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformationd_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    deformationf_arr = params[n_phases + n_unit + n_deform: n_phases + n_unit + 2*n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + 2*n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        # Do the same for |0⟩.
        zeroes = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '0']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationd_arr[site-1, 2*superpos_idx] + deformationd_arr[site-1, 2*superpos_idx+1]*1j
            
            # Do the same for |0⟩.
            for site in zeroes:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationf_arr[site-1, 2*superpos_idx] + deformationf_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase, as with full symmetryzation.
            # For varphi_{i, j}, it will be the minimum of j-i-1 and (i+N)-j-1
            # This will also decide which of the two phases to use.
            # If j-i is the smaller, then i is the "first site". If that's not
            # the case, then j is the "first site".
            if pair[1] - pair[0] < pair[0] + N - pair[1]:
                phase_idx = pair[1] - pair[0] - 1
                
                # Add phase to the phase variable.
                phase += phases_arr[pair[0]%2][phase_idx]
                
            else:
                phase_idx = pair[0] + N - pair[1] - 1
                
                # Add phase to the phase variable.
                phase += phases_arr[pair[1]%2][phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)
        
    # Return expectation value, adding the -J/8 factor
    return J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))

def exp_val_sshsym_ad_cocr(params, N, m, hamiltonian_function, J_sign = 1, unitaries = False, closed = True, const = 1, pairs = [(1, 2)], *args):
    """
    This function sets up the variational state and implements the expectation value calculation 
    with SSH phase symmetrization using an alternate deformation approach with cross correlations 
    as a penalty function.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray with the values for the parameters of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    hamiltonian_function : callable 
        This function should accept at least two arguments, an array of length 2^N for the 
        coefficients of the variational |eta⟩ state and a bool to specify the topology of the 
        chain, and return an array of length 2^N for the coefficients of H|eta⟩ for the desired 
        Hamiltonian. Should be of the form `hamiltonian_function(eta_arr, closed, *args)`. 
        See `args` for additional arguments.
    J_sign : int, optional
        Either 1 or -1, for ferromagnetic and antiferromagnetic chain, respectively. Default is 1.
    unitaries : bool, optional
        If True, unitaries are applied when constructing the variational state. Default is False.
    closed : bool, optional
        If True then the spin chain has periodic boundary conditions (closed chain). Default is True.
    const : float, optional
        Constant for the penalty function.
    pairs : list, optional
        List of pairs of sites as tuples for cross correlations.
    args : optional
        Any additional arguments needed for `hamiltonian_function`.
    
    Returns
    -------
    exp_val : float
        Expectation value of the Hamiltonian.
    """
    
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = 2*math.floor(N/2), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # phi_{1}, phi_{2}, ..., phi_{N-1},
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1},
    phases_arr = params[:n_phases].reshape(2, math.floor(N/2))
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformationd_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    deformationf_arr = params[n_phases + n_unit + n_deform: n_phases + n_unit + 2*n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + 2*n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        # Do the same for |0⟩.
        zeroes = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '0']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationd_arr[site-1, 2*superpos_idx] + deformationd_arr[site-1, 2*superpos_idx+1]*1j
            
            # Do the same for |0⟩.
            for site in zeroes:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationf_arr[site-1, 2*superpos_idx] + deformationf_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates, partial symmetrization.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase, as with full symmetryzation.
            # For varphi_{i, j}, it will be the minimum of j-i-1 and (i+N)-j-1
            # This will also decide which of the two phases to use.
            # If j-i is the smaller, then i is the "first site". If that's not
            # the case, then j is the "first site".
            if pair[1] - pair[0] < pair[0] + N - pair[1]:
                phase_idx = pair[1] - pair[0] - 1
                
                # Add phase to the phase variable.
                phase += phases_arr[pair[0]%2][phase_idx]
                
            else:
                phase_idx = pair[0] + N - pair[1] - 1
                
                # Add phase to the phase variable.
                phase += phases_arr[pair[1]%2][phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    ### H|eta⟩ array.
    Heta_arr = hamiltonian_function(eta_arr, closed, *args)

    ### Calculate expectation value, adding the -J/8 factor.
    ev = J_sign*np.real(np.vdot(eta_arr, Heta_arr)) / np.real(-8*np.vdot(eta_arr, eta_arr))
    
    ### Correlation functions.
    cross_corr = 0
    for pair in pairs:
        # Set array to store sigma_{spin_1}^(x) sigma_{spin_2}^(y) |eta⟩ 
        temp_eta =  np.zeros(len(eta_arr)).astype(complex)
        for vector, coefficient in enumerate(eta_arr): 
            # For every pair of sites, one, and only one, sigma combination
            # will act for S^{x} and S^{y}.

            # If 2nd spin is down.
            if bitset(vector, pair[1]-1, N): 
                # Add calculated coefficient to array. 
                temp_eta[flipanybits(vector, pair[0]-1, pair[1]-1, N)] += coefficient

            # If 2nd spin is up.
            else:
                # Add calculated coefficient to array.
                temp_eta[flipanybits(vector, pair[0]-1, pair[1]-1, N)] -= coefficient


        # Calculate correlation and add it.
        corr = np.real(1j*np.vdot(eta_arr, temp_eta)) / np.real(-4*np.vdot(eta_arr, eta_arr))
        cross_corr += corr*corr
        
    # Return expectation value plus penalty
    return ev + const*cross_corr

#######################################################################################################
#######################################################################################################
########################################## Correlation ################################################
#######################################################################################################
#######################################################################################################

def correlation(params, N, m, unitaries, symmetrization, cayley = False):
    """
    This function calculates the spin-spin correlation functions and average magnetization
    per spin for the ground state.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray that describes the ground state with the values for the parameters 
        of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    unitaries : bool
        If True, unitaries were applied when constructing the variational state.
    symmetrization : int
        Used to specify phase symmetrization.
            0: No symmetrization.
            1: Partial symmetrization.
            2: Full symmetrization.
    
    cayley : bool, optional
        If True, unitaries are parametrized by a Cayley transform.
    
    Returns
    -------
    corr_matrix_x : (N, N) ndarray 
        Matrix with elements S_{jk}^{x}.
    corr_matrix_y : (N, N) ndarray 
        Matrix with elements S_{jk}^{y}.
    corr_matrix_z : (N, N) ndarray 
        Matrix with elements S_{jk}^{z}.
    magnetization_arr : (N,) ndarray
        1-D NumPy ndarray with elements M_{j}^{z}.
    """
    
    ### Parameter handling.
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    if symmetrization == 0:
        n_phases, n_unit, n_deform, n_superpos = int(N*(N-1)/2), 3*N*unitaries, 2*m*N, 2*m
    elif symmetrization == 1:
        n_phases, n_unit, n_deform, n_superpos = int(N-1), 3*N*unitaries, 2*m*N, 2*m
    else:
        n_phases, n_unit, n_deform, n_superpos = math.floor(N/2), 3*N*unitaries, 2*m*N, 2*m
        
    if cayley:
        n_unit += N*unitaries
        if symmetrization != 1:
            return "Only partial symmetrization implemented"
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1,2}, varphi_{1,3}, ..., varphi_{1,N}, varphi_{2,3}, ..., varphi_{N-1, N}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformation_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformation_arr[site-1, 2*superpos_idx] + deformation_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        if symmetrization == 0:
            for pair in site_combinations:
                # Get index for corresponding phase.
                counter, offset = 1, 0
                for step in range(pair[0] - 1):
                    offset += (N - counter)
                    counter += 1

                phase_idx = offset + (pair[1] - pair[0] - 1)
                
                # Add phase to the phase variable.
                phase += phases_arr[phase_idx]
        
        elif symmetrization == 1:
            for pair in site_combinations:
                # Get index for corresponding phase.
                phase_idx = pair[1] - pair[0] - 1

                # Add phase to the phase variable.
                phase += phases_arr[phase_idx]
            
        else:
            for pair in site_combinations:
                # Get index for corresponding phase.
                # For varphi_{i, j}, it will be the minimum of j-i-1 and (i+N)-j-1
                if pair[1] - pair[0] < pair[0] + N - pair[1]:
                    phase_idx = pair[1] - pair[0] - 1
                else:
                    phase_idx = pair[0] + N - pair[1] - 1

                # Add phase to the phase variable.
                phase += phases_arr[phase_idx]
       

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []
        
        if cayley:
            # Set matrices and store them.
            for site in range(N):
                # Get a, b, c and d parameters.
                a = unitaries_arr[4*site]
                b = unitaries_arr[4*site + 1]
                c = unitaries_arr[4*site + 2]
                d = unitaries_arr[4*site + 3]

                # Calculate factor.
                factor = a*b - 1 - c*c - d*d + a*1j + b*1j
                factor = factor / ((a*b - 1 - c*c - d*d)**2 + (a+b)**2)

                # Set matrix elements.
                u11 = -a*b - 1 + c*c + d*d + a*1j - b*1j
                u12 = -2*d + 2*c*1j
                u21 = 2*d + 2*c*1j
                u22 = -a*b - 1 + c*c + d*d - a*1j + b*1j

                # Add matrix to list.
                unitaries_list.append(factor*np.array([[u11, u12], [u21, u22]]))
                
        else:
            # Set matrices and store them.
            for site in range(N):
                # Get beta, gamma and delta parameters.
                beta = unitaries_arr[3*site]
                gamma = unitaries_arr[3*site + 1]
                delta = unitaries_arr[3*site + 2]

                # Set matrix elements.          
                u11 = np.exp(beta*1j) * np.cos(delta)
                u12 = -np.exp(-gamma*1j) * np.sin(delta)
                u21 = -np.conjugate(u12)
                u22 = np.conjugate(u11)

                # Add matrix to list.
                unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    
    ### Magnetization
    magnetization_arr = np.zeros(N)
    
    for spin_1 in range(N):
        # Set array to store sigma_{spin_1}^(z) |eta⟩ 
        temp_eta =  np.zeros(len(eta_arr)).astype(complex)
        
        # Loop through product states.
        for vector, coefficient in enumerate(eta_arr): 
            # sigma_{i}^(z) |state⟩ = |state⟩  if |state⟩  ~ |0⟩_{i}
            # sigma_{i}^(z) |state⟩ = -|state⟩ if |state⟩  ~ |1⟩_{i}
            if bitset(vector, spin_1, N):
                temp_eta[vector] -= coefficient
            else:
                temp_eta[vector] += coefficient
            
        # Store expectation value in magnetization array. Add 1/2 factor
        magnetization_arr[spin_1] = np.real(np.vdot(eta_arr, temp_eta)) / np.real(2*np.vdot(eta_arr, eta_arr))
    
    ### Correlation
    corr_matrix_x = np.zeros((N, N))
    corr_matrix_y = np.zeros((N, N))
    corr_matrix_z = np.zeros((N, N))
    
    for spin_1 in range(N):
        for spin_2 in range(spin_1, N):
            # Set arrays to store sigma_{spin_1}^(l) sigma_{spin_2}^(l) |eta⟩ 
            temp_eta =  np.zeros(len(eta_arr)).astype(complex)
            temp_eta_x = np.zeros(len(eta_arr)).astype(complex)
            temp_eta_y = np.zeros(len(eta_arr)).astype(complex)
            for vector, coefficient in enumerate(eta_arr): 
                # For every pair of sites, one, and only one, sigma combination
                # will act for S^{x} and S^{y}. For S^{z}, it always acts.
                     
                # If neighbouring sites are in different states.
                if bitset(vector, spin_1, N) != bitset(vector, spin_2, N): 
                    # Add calculated coefficient to array. 
                    # Instead of multiplying by 4 (due to the laddter operators)
                    # we divide by 4 (instead of by 16) when taking the expectation values.
                    temp_eta_x[flipanybits(vector, spin_1, spin_2, N)] += coefficient
                    temp_eta_y[flipanybits(vector, spin_1, spin_2, N)] -= coefficient
                    temp_eta[vector] -= coefficient
                # If they are in the same state.
                else:
                    # Add calculated coefficient to array.
                    temp_eta_x[flipanybits(vector, spin_1, spin_2, N)] += coefficient
                    temp_eta_y[flipanybits(vector, spin_1, spin_2, N)] += coefficient
                    temp_eta[vector] += coefficient
                    
            # Store expectation values in matrix with appropriate factors.
            corr_matrix_x[spin_1, spin_2] = np.real(np.vdot(eta_arr, temp_eta_x)) / np.real(4*np.vdot(eta_arr, eta_arr))
            corr_matrix_x[spin_2, spin_1] = corr_matrix_x[spin_1, spin_2]
            corr_matrix_y[spin_1, spin_2] = np.real(np.vdot(eta_arr, temp_eta_y)) / np.real(-4*np.vdot(eta_arr, eta_arr))
            corr_matrix_y[spin_2, spin_1] = corr_matrix_y[spin_1, spin_2]
            corr_matrix_z[spin_1, spin_2] = np.real(np.vdot(eta_arr, temp_eta)) / np.real(4*np.vdot(eta_arr, eta_arr))
            corr_matrix_z[spin_2, spin_1] = corr_matrix_z[spin_1, spin_2]
            
    # Return arrays
    return corr_matrix_x, corr_matrix_y, corr_matrix_z, magnetization_arr

def xy_cross_correlation(params, N, m, unitaries, symmetrization, cayley = False):
    """
    This function calculates the xy cross correlations for the ground state.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray that describes the ground state with the values for the parameters 
        of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    unitaries : bool
        If True, unitaries were applied when constructing the variational state.
    symmetrization : int
        Used to specify phase symmetrization.
            0: No symmetrization.
            1: Partial symmetrization.
            2: Full symmetrization.
    
    cayley : bool, optional
        If True, unitaries are parametrized by a Cayley transform.
    
    Returns
    -------
    corr_matrix_xy : (N, N) ndarray 
        Matrix with elements Im(S_{jk}^{xy}).
    """
    
    ### Parameter handling.
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    if symmetrization == 0:
        n_phases, n_unit, n_deform, n_superpos = int(N*(N-1)/2), 3*N*unitaries, 2*m*N, 2*m
    elif symmetrization == 1:
        n_phases, n_unit, n_deform, n_superpos = int(N-1), 3*N*unitaries, 2*m*N, 2*m
    else:
        n_phases, n_unit, n_deform, n_superpos = math.floor(N/2), 3*N*unitaries, 2*m*N, 2*m
        
    if cayley:
        n_unit += N*unitaries
        if symmetrization != 1:
            return "Only partial symmetrization implemented"
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1,2}, varphi_{1,3}, ..., varphi_{1,N}, varphi_{2,3}, ..., varphi_{N-1, N}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformation_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformation_arr[site-1, 2*superpos_idx] + deformation_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        if symmetrization == 0:
            for pair in site_combinations:
                # Get index for corresponding phase.
                counter, offset = 1, 0
                for step in range(pair[0] - 1):
                    offset += (N - counter)
                    counter += 1

                phase_idx = offset + (pair[1] - pair[0] - 1)
                
                # Add phase to the phase variable.
                phase += phases_arr[phase_idx]
        
        elif symmetrization == 1:
            for pair in site_combinations:
                # Get index for corresponding phase.
                phase_idx = pair[1] - pair[0] - 1

                # Add phase to the phase variable.
                phase += phases_arr[phase_idx]
            
        else:
            for pair in site_combinations:
                # Get index for corresponding phase.
                # For varphi_{i, j}, it will be the minimum of j-i-1 and (i+N)-j-1
                if pair[1] - pair[0] < pair[0] + N - pair[1]:
                    phase_idx = pair[1] - pair[0] - 1
                else:
                    phase_idx = pair[0] + N - pair[1] - 1

                # Add phase to the phase variable.
                phase += phases_arr[phase_idx]
       

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []
        
        if cayley:
            # Set matrices and store them.
            for site in range(N):
                # Get a, b, c and d parameters.
                a = unitaries_arr[4*site]
                b = unitaries_arr[4*site + 1]
                c = unitaries_arr[4*site + 2]
                d = unitaries_arr[4*site + 3]

                # Calculate factor.
                factor = a*b - 1 - c*c - d*d + a*1j + b*1j
                factor = factor / ((a*b - 1 - c*c - d*d)**2 + (a+b)**2)

                # Set matrix elements.
                u11 = -a*b - 1 + c*c + d*d + a*1j - b*1j
                u12 = -2*d + 2*c*1j
                u21 = 2*d + 2*c*1j
                u22 = -a*b - 1 + c*c + d*d - a*1j + b*1j

                # Add matrix to list.
                unitaries_list.append(factor*np.array([[u11, u12], [u21, u22]]))
                
        else:
            # Set matrices and store them.
            for site in range(N):
                # Get beta, gamma and delta parameters.
                beta = unitaries_arr[3*site]
                gamma = unitaries_arr[3*site + 1]
                delta = unitaries_arr[3*site + 2]

                # Set matrix elements.          
                u11 = np.exp(beta*1j) * np.cos(delta)
                u12 = -np.exp(-gamma*1j) * np.sin(delta)
                u21 = -np.conjugate(u12)
                u22 = np.conjugate(u11)

                # Add matrix to list.
                unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    
    ### Correlation
    corr_matrix_xy = np.zeros((N, N))
    
    for spin_1 in range(N):
        for spin_2 in range(spin_1, N):
            # Set array to store sigma_{spin_1}^(x) sigma_{spin_2}^(y) |eta⟩ 
            temp_eta =  np.zeros(len(eta_arr)).astype(complex)
            for vector, coefficient in enumerate(eta_arr): 
                # For every pair of sites, one, and only one, sigma combination
                # will act for S^{x} and S^{y}.
                     
                # If 2nd spin is down.
                if bitset(vector, spin_2, N): 
                    # Add calculated coefficient to array. 
                    # Instead of multiplying by 4 (due to the laddter operators)
                    # we divide by 4 (instead of by 16) when taking the expectation values.
                    temp_eta[flipanybits(vector, spin_1, spin_2, N)] += coefficient

                # If 2nd spin is up.
                else:
                    # Add calculated coefficient to array.
                    temp_eta[flipanybits(vector, spin_1, spin_2, N)] -= coefficient
                    
            # Store expectation values in matrix with appropriate factors.
            corr_matrix_xy[spin_1, spin_2] = np.real(1j*np.vdot(eta_arr, temp_eta)) / np.real(-4*np.vdot(eta_arr, eta_arr))
            corr_matrix_xy[spin_2, spin_1] = corr_matrix_xy[spin_1, spin_2]
            
    # Return array
    return corr_matrix_xy

def correlation_ad(params, N, m, unitaries, symmetrization, cayley = False):
    """
    This function calculates the spin-spin correlation functions and average magnetization
    per spin for the ground state using an alternate deformation approach.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray that describes the ground state with the values for the parameters 
        of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    unitaries : bool
        If True, unitaries were applied when constructing the variational state.
    symmetrization : int
        Used to specify phase symmetrization.
            0: No symmetrization.
            1: Partial symmetrization.
            2: Full symmetrization.
    
    cayley : bool, optional
        If True, unitaries are parametrized by a Cayley transform.
    
    Returns
    -------
    corr_matrix_x : (N, N) ndarray 
        Matrix with elements S_{jk}^{x}.
    corr_matrix_y : (N, N) ndarray 
        Matrix with elements S_{jk}^{y}.
    corr_matrix_z : (N, N) ndarray 
        Matrix with elements S_{jk}^{z}.
    magnetization_arr : (N,) ndarray
        1-D NumPy ndarray with elements M_{j}^{z}.
    """
    
    ### Parameter handling.
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    if symmetrization == 0:
        n_phases, n_unit, n_deform, n_superpos = int(N*(N-1)/2), 3*N*unitaries, 2*m*N, 2*m
    elif symmetrization == 1:
        n_phases, n_unit, n_deform, n_superpos = int(N-1), 3*N*unitaries, 2*m*N, 2*m
    else:
        n_phases, n_unit, n_deform, n_superpos = math.floor(N/2), 3*N*unitaries, 2*m*N, 2*m
        
    if cayley:
        n_unit += N*unitaries
        if symmetrization != 1:
            return "Only partial symmetrization implemented"
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1,2}, varphi_{1,3}, ..., varphi_{1,N}, varphi_{2,3}, ..., varphi_{N-1, N}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformationd_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    deformationf_arr = params[n_phases + n_unit + n_deform: n_phases + n_unit + 2*n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + 2*n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        # Do the same for |0⟩.
        zeroes = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '0']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationd_arr[site-1, 2*superpos_idx] + deformationd_arr[site-1, 2*superpos_idx+1]*1j
            
            # Do the same for |0⟩.
            for site in zeroes:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationf_arr[site-1, 2*superpos_idx] + deformationf_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        if symmetrization == 0:
            for pair in site_combinations:
                # Get index for corresponding phase.
                counter, offset = 1, 0
                for step in range(pair[0] - 1):
                    offset += (N - counter)
                    counter += 1

                phase_idx = offset + (pair[1] - pair[0] - 1)
                
                # Add phase to the phase variable.
                phase += phases_arr[phase_idx]
        
        elif symmetrization == 1:
            for pair in site_combinations:
                # Get index for corresponding phase.
                phase_idx = pair[1] - pair[0] - 1

                # Add phase to the phase variable.
                phase += phases_arr[phase_idx]
            
        else:
            for pair in site_combinations:
                # Get index for corresponding phase.
                # For varphi_{i, j}, it will be the minimum of j-i-1 and (i+N)-j-1
                if pair[1] - pair[0] < pair[0] + N - pair[1]:
                    phase_idx = pair[1] - pair[0] - 1
                else:
                    phase_idx = pair[0] + N - pair[1] - 1

                # Add phase to the phase variable.
                phase += phases_arr[phase_idx]
       

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []
        
        if cayley:
            # Set matrices and store them.
            for site in range(N):
                # Get a, b, c and d parameters.
                a = unitaries_arr[4*site]
                b = unitaries_arr[4*site + 1]
                c = unitaries_arr[4*site + 2]
                d = unitaries_arr[4*site + 3]

                # Calculate factor.
                factor = a*b - 1 - c*c - d*d + a*1j + b*1j
                factor = factor / ((a*b - 1 - c*c - d*d)**2 + (a+b)**2)

                # Set matrix elements.
                u11 = -a*b - 1 + c*c + d*d + a*1j - b*1j
                u12 = -2*d + 2*c*1j
                u21 = 2*d + 2*c*1j
                u22 = -a*b - 1 + c*c + d*d - a*1j + b*1j

                # Add matrix to list.
                unitaries_list.append(factor*np.array([[u11, u12], [u21, u22]]))
                
        else:
            # Set matrices and store them.
            for site in range(N):
                # Get beta, gamma and delta parameters.
                beta = unitaries_arr[3*site]
                gamma = unitaries_arr[3*site + 1]
                delta = unitaries_arr[3*site + 2]

                # Set matrix elements.          
                u11 = np.exp(beta*1j) * np.cos(delta)
                u12 = -np.exp(-gamma*1j) * np.sin(delta)
                u21 = -np.conjugate(u12)
                u22 = np.conjugate(u11)

                # Add matrix to list.
                unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    
    ### Magnetization
    magnetization_arr = np.zeros(N)
    
    for spin_1 in range(N):
        # Set array to store sigma_{spin_1}^(z) |eta⟩ 
        temp_eta =  np.zeros(len(eta_arr)).astype(complex)
        
        # Loop through product states.
        for vector, coefficient in enumerate(eta_arr): 
            # sigma_{i}^(z) |state⟩ = |state⟩  if |state⟩  ~ |0⟩_{i}
            # sigma_{i}^(z) |state⟩ = -|state⟩ if |state⟩  ~ |1⟩_{i}
            if bitset(vector, spin_1, N):
                temp_eta[vector] -= coefficient
            else:
                temp_eta[vector] += coefficient
            
        # Store expectation value in magnetization array. Add 1/2 factor
        magnetization_arr[spin_1] = np.real(np.vdot(eta_arr, temp_eta)) / np.real(2*np.vdot(eta_arr, eta_arr))
    
    ### Correlation
    corr_matrix_x = np.zeros((N, N))
    corr_matrix_y = np.zeros((N, N))
    corr_matrix_z = np.zeros((N, N))
    
    for spin_1 in range(N):
        for spin_2 in range(spin_1, N):
            # Set arrays to store sigma_{spin_1}^(l) sigma_{spin_2}^(l) |eta⟩ 
            temp_eta =  np.zeros(len(eta_arr)).astype(complex)
            temp_eta_x = np.zeros(len(eta_arr)).astype(complex)
            temp_eta_y = np.zeros(len(eta_arr)).astype(complex)
            for vector, coefficient in enumerate(eta_arr): 
                # For every pair of sites, one, and only one, sigma combination
                # will act for S^{x} and S^{y}. For S^{z}, it always acts.
                     
                # If neighbouring sites are in different states.
                if bitset(vector, spin_1, N) != bitset(vector, spin_2, N): 
                    # Add calculated coefficient to array. 
                    # Instead of multiplying by 4 (due to the ladder operators)
                    # we divide by 4 (instead of by 16) when taking the expectation values.
                    temp_eta_x[flipanybits(vector, spin_1, spin_2, N)] += coefficient
                    temp_eta_y[flipanybits(vector, spin_1, spin_2, N)] -= coefficient
                    temp_eta[vector] -= coefficient
                # If they are in the same state.
                else:
                    # Add calculated coefficient to array.
                    temp_eta_x[flipanybits(vector, spin_1, spin_2, N)] += coefficient
                    temp_eta_y[flipanybits(vector, spin_1, spin_2, N)] += coefficient
                    temp_eta[vector] += coefficient
                    
            # Store expectation values in matrix with appropriate factors.
            corr_matrix_x[spin_1, spin_2] = np.real(np.vdot(eta_arr, temp_eta_x)) / np.real(4*np.vdot(eta_arr, eta_arr))
            corr_matrix_x[spin_2, spin_1] = corr_matrix_x[spin_1, spin_2]
            corr_matrix_y[spin_1, spin_2] = np.real(np.vdot(eta_arr, temp_eta_y)) / np.real(-4*np.vdot(eta_arr, eta_arr))
            corr_matrix_y[spin_2, spin_1] = corr_matrix_y[spin_1, spin_2]
            corr_matrix_z[spin_1, spin_2] = np.real(np.vdot(eta_arr, temp_eta)) / np.real(4*np.vdot(eta_arr, eta_arr))
            corr_matrix_z[spin_2, spin_1] = corr_matrix_z[spin_1, spin_2]
            
    # Return arrays
    return corr_matrix_x, corr_matrix_y, corr_matrix_z, magnetization_arr

def xy_cross_correlation_ad(params, N, m, unitaries, symmetrization, cayley = False):
    """
    This function calculates the xy cross correlations for the ground state using an 
    alternate deformation approach.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray that describes the ground state with the values for the parameters 
        of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    unitaries : bool
        If True, unitaries were applied when constructing the variational state.
    symmetrization : int
        Used to specify phase symmetrization.
            0: No symmetrization.
            1: Partial symmetrization.
            2: Full symmetrization.
    
    cayley : bool, optional
        If True, unitaries are parametrized by a Cayley transform.
    
    Returns
    -------
    corr_matrix_xy : (N, N) ndarray 
        Matrix with elements Im(S_{jk}^{xy}).
    """
    
    ### Parameter handling.
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    if symmetrization == 0:
        n_phases, n_unit, n_deform, n_superpos = int(N*(N-1)/2), 3*N*unitaries, 2*m*N, 2*m
    elif symmetrization == 1:
        n_phases, n_unit, n_deform, n_superpos = int(N-1), 3*N*unitaries, 2*m*N, 2*m
    else:
        n_phases, n_unit, n_deform, n_superpos = math.floor(N/2), 3*N*unitaries, 2*m*N, 2*m
        
    if cayley:
        n_unit += N*unitaries
        if symmetrization != 1:
            return "Only partial symmetrization implemented"
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # varphi_{1,2}, varphi_{1,3}, ..., varphi_{1,N}, varphi_{2,3}, ..., varphi_{N-1, N}
    phases_arr = params[:n_phases]
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformationd_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    deformationf_arr = params[n_phases + n_unit + n_deform: n_phases + n_unit + 2*n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + 2*n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        # Do the same for |0⟩.
        zeroes = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '0']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationd_arr[site-1, 2*superpos_idx] + deformationd_arr[site-1, 2*superpos_idx+1]*1j
            
            # Do the same for |0⟩.
            for site in zeroes:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationf_arr[site-1, 2*superpos_idx] + deformationf_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        if symmetrization == 0:
            for pair in site_combinations:
                # Get index for corresponding phase.
                counter, offset = 1, 0
                for step in range(pair[0] - 1):
                    offset += (N - counter)
                    counter += 1

                phase_idx = offset + (pair[1] - pair[0] - 1)
                
                # Add phase to the phase variable.
                phase += phases_arr[phase_idx]
        
        elif symmetrization == 1:
            for pair in site_combinations:
                # Get index for corresponding phase.
                phase_idx = pair[1] - pair[0] - 1

                # Add phase to the phase variable.
                phase += phases_arr[phase_idx]
            
        else:
            for pair in site_combinations:
                # Get index for corresponding phase.
                # For varphi_{i, j}, it will be the minimum of j-i-1 and (i+N)-j-1
                if pair[1] - pair[0] < pair[0] + N - pair[1]:
                    phase_idx = pair[1] - pair[0] - 1
                else:
                    phase_idx = pair[0] + N - pair[1] - 1

                # Add phase to the phase variable.
                phase += phases_arr[phase_idx]
       

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []
        
        if cayley:
            # Set matrices and store them.
            for site in range(N):
                # Get a, b, c and d parameters.
                a = unitaries_arr[4*site]
                b = unitaries_arr[4*site + 1]
                c = unitaries_arr[4*site + 2]
                d = unitaries_arr[4*site + 3]

                # Calculate factor.
                factor = a*b - 1 - c*c - d*d + a*1j + b*1j
                factor = factor / ((a*b - 1 - c*c - d*d)**2 + (a+b)**2)

                # Set matrix elements.
                u11 = -a*b - 1 + c*c + d*d + a*1j - b*1j
                u12 = -2*d + 2*c*1j
                u21 = 2*d + 2*c*1j
                u22 = -a*b - 1 + c*c + d*d - a*1j + b*1j

                # Add matrix to list.
                unitaries_list.append(factor*np.array([[u11, u12], [u21, u22]]))
                
        else:
            # Set matrices and store them.
            for site in range(N):
                # Get beta, gamma and delta parameters.
                beta = unitaries_arr[3*site]
                gamma = unitaries_arr[3*site + 1]
                delta = unitaries_arr[3*site + 2]

                # Set matrix elements.          
                u11 = np.exp(beta*1j) * np.cos(delta)
                u12 = -np.exp(-gamma*1j) * np.sin(delta)
                u21 = -np.conjugate(u12)
                u22 = np.conjugate(u11)

                # Add matrix to list.
                unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    
    ### Correlation
    corr_matrix_xy = np.zeros((N, N))
    
    for spin_1 in range(N):
        for spin_2 in range(spin_1, N):
            # Set array to store sigma_{spin_1}^(x) sigma_{spin_2}^(y) |eta⟩ 
            temp_eta =  np.zeros(len(eta_arr)).astype(complex)
            for vector, coefficient in enumerate(eta_arr): 
                # For every pair of sites, one, and only one, sigma combination
                # will act for S^{x} and S^{y}.
                     
                # If 2nd spin is down.
                if bitset(vector, spin_2, N): 
                    # Add calculated coefficient to array. 
                    # Instead of multiplying by 4 (due to the ladder operators)
                    # we divide by 4 (instead of by 16) when taking the expectation values.
                    temp_eta[flipanybits(vector, spin_1, spin_2, N)] += coefficient

                # If 2nd spin is up.
                else:
                    # Add calculated coefficient to array.
                    temp_eta[flipanybits(vector, spin_1, spin_2, N)] -= coefficient
                    
            # Store expectation values in matrix with appropriate factors.
            corr_matrix_xy[spin_1, spin_2] = np.real(1j*np.vdot(eta_arr, temp_eta)) / np.real(-4*np.vdot(eta_arr, eta_arr))
            corr_matrix_xy[spin_2, spin_1] = corr_matrix_xy[spin_1, spin_2]
            
    # Return array
    return corr_matrix_xy

def correlation_ad_ssh(params, N, m, unitaries):
    """
    This function calculates the spin-spin correlation functions and average magnetization
    per spin for the ground state using SSH symmetrization and alternate deformation approach.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray that describes the ground state with the values for the parameters 
        of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    unitaries : bool
        If True, unitaries were applied when constructing the variational state.
    
    Returns
    -------
    corr_matrix_x : (N, N) ndarray 
        Matrix with elements S_{jk}^{x}.
    corr_matrix_y : (N, N) ndarray 
        Matrix with elements S_{jk}^{y}.
    corr_matrix_z : (N, N) ndarray 
        Matrix with elements S_{jk}^{z}.
    magnetization_arr : (N,) ndarray
        1-D NumPy ndarray with elements M_{j}^{z}.
    """
    
    ### Parameter handling.
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = 2*math.floor(N/2), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # phi_{1}, phi_{2}, ..., phi_{N-1},
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1},
    phases_arr = params[:n_phases].reshape(2, math.floor(N/2))
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformationd_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    deformationf_arr = params[n_phases + n_unit + n_deform: n_phases + n_unit + 2*n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + 2*n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        # Do the same for |0⟩.
        zeroes = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '0']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationd_arr[site-1, 2*superpos_idx] + deformationd_arr[site-1, 2*superpos_idx+1]*1j
            
            # Do the same for |0⟩.
            for site in zeroes:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationf_arr[site-1, 2*superpos_idx] + deformationf_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase, as with full symmetryzation.
            # For varphi_{i, j}, it will be the minimum of j-i-1 and (i+N)-j-1
            # This will also decide which of the two phases to use.
            # If j-i is the smaller, then i is the "first site". If that's not
            # the case, then j is the "first site".
            if pair[1] - pair[0] < pair[0] + N - pair[1]:
                phase_idx = pair[1] - pair[0] - 1
                
                # Add phase to the phase variable.
                phase += phases_arr[pair[0]%2][phase_idx]
                
            else:
                phase_idx = pair[0] + N - pair[1] - 1
                
                # Add phase to the phase variable.
                phase += phases_arr[pair[1]%2][phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    
    ### Magnetization
    magnetization_arr = np.zeros(N)
    
    for spin_1 in range(N):
        # Set array to store sigma_{spin_1}^(z) |eta⟩ 
        temp_eta =  np.zeros(len(eta_arr)).astype(complex)
        
        # Loop through product states.
        for vector, coefficient in enumerate(eta_arr): 
            # sigma_{i}^(z) |state⟩ = |state⟩  if |state⟩  ~ |0⟩_{i}
            # sigma_{i}^(z) |state⟩ = -|state⟩ if |state⟩  ~ |1⟩_{i}
            if bitset(vector, spin_1, N):
                temp_eta[vector] -= coefficient
            else:
                temp_eta[vector] += coefficient
            
        # Store expectation value in magnetization array. Add 1/2 factor
        magnetization_arr[spin_1] = np.real(np.vdot(eta_arr, temp_eta)) / np.real(2*np.vdot(eta_arr, eta_arr))
    
    ### Correlation
    corr_matrix_x = np.zeros((N, N))
    corr_matrix_y = np.zeros((N, N))
    corr_matrix_z = np.zeros((N, N))
    
    for spin_1 in range(N):
        for spin_2 in range(spin_1, N):
            # Set arrays to store sigma_{spin_1}^(l) sigma_{spin_2}^(l) |eta⟩ 
            temp_eta =  np.zeros(len(eta_arr)).astype(complex)
            temp_eta_x = np.zeros(len(eta_arr)).astype(complex)
            temp_eta_y = np.zeros(len(eta_arr)).astype(complex)
            for vector, coefficient in enumerate(eta_arr): 
                # For every pair of sites, one, and only one, sigma combination
                # will act for S^{x} and S^{y}. For S^{z}, it always acts.
                     
                # If neighbouring sites are in different states.
                if bitset(vector, spin_1, N) != bitset(vector, spin_2, N): 
                    # Add calculated coefficient to array. 
                    # Instead of multiplying by 4 (due to the ladder operators)
                    # we divide by 4 (instead of by 16) when taking the expectation values.
                    temp_eta_x[flipanybits(vector, spin_1, spin_2, N)] += coefficient
                    temp_eta_y[flipanybits(vector, spin_1, spin_2, N)] -= coefficient
                    temp_eta[vector] -= coefficient
                # If they are in the same state.
                else:
                    # Add calculated coefficient to array.
                    temp_eta_x[flipanybits(vector, spin_1, spin_2, N)] += coefficient
                    temp_eta_y[flipanybits(vector, spin_1, spin_2, N)] += coefficient
                    temp_eta[vector] += coefficient
                    
            # Store expectation values in matrix with appropriate factors.
            corr_matrix_x[spin_1, spin_2] = np.real(np.vdot(eta_arr, temp_eta_x)) / np.real(4*np.vdot(eta_arr, eta_arr))
            corr_matrix_x[spin_2, spin_1] = corr_matrix_x[spin_1, spin_2]
            corr_matrix_y[spin_1, spin_2] = np.real(np.vdot(eta_arr, temp_eta_y)) / np.real(-4*np.vdot(eta_arr, eta_arr))
            corr_matrix_y[spin_2, spin_1] = corr_matrix_y[spin_1, spin_2]
            corr_matrix_z[spin_1, spin_2] = np.real(np.vdot(eta_arr, temp_eta)) / np.real(4*np.vdot(eta_arr, eta_arr))
            corr_matrix_z[spin_2, spin_1] = corr_matrix_z[spin_1, spin_2]
            
    # Return arrays
    return corr_matrix_x, corr_matrix_y, corr_matrix_z, magnetization_arr

def xy_cross_correlation_ad_ssh(params, N, m, unitaries):
    """
    This function calculates the xy cross correlations for the ground state using 
    SSH symmetrization and alternate deformation approach.
    
    Parameters
    ----------
    params : array_like
        A 1-D NumPy ndarray that describes the ground state with the values for the parameters 
        of the variational state |eta⟩.
    N : int
        The number of spins in the chain.
    m : int
        The number of superposition states.
    unitaries : bool
        If True, unitaries were applied when constructing the variational state.
    
    Returns
    -------
    corr_matrix_xy : (N, N) ndarray 
        Matrix with elements Im(S_{jk}^{xy}).
    """
    
    ### Parameter handling.
    # Set number of parameters. Note on n_unit: booleans are a subclass of int; True is 1, False is 0
    n_phases, n_unit, n_deform, n_superpos = 2*math.floor(N/2), 3*N*unitaries, 2*m*N, 2*m
    
    ### Slice parameter array to get specific types of parameters. 
    # Phases and unitaries' parameters are real numbers. Deformation and superposition
    # coefficients are complex numbers, and the parameters are their real and imaginary parts.
    
    # Phases. Order is as follows: 
    # phi_{1}, phi_{2}, ..., phi_{N-1},
    # varphi_{1}, varphi_{2}, ..., varphi_{N-1},
    phases_arr = params[:n_phases].reshape(2, math.floor(N/2))
    
    # Unitaries. Order is as follows: 
    # beta_{1}, gamma_{1}, delta_{1}, beta_{2}, gamma_{2}, delta_{2}, ..., delta_{N}
    unitaries_arr = params[n_phases: n_phases + n_unit]
    
    # Deformation. Order is as follows:
    # Re(d_{1}^{(1)}), Im(d_{1}^{(1)}), Re(d_{1}^{(2)}), Im(d_{1}^{(2)}), ..., Re(d_{1}^{(m)}), Im(d_{1}^{(m)}),
    # Re(d_{2}^{(1)}), Im(d_{2}^{(1)}), Re(d_{2}^{(2)}), Im(d_{2}^{(2)}), ..., Re(d_{2}^{(m)}), Im(d_{2}^{(m)}), 
    # ...,
    # Re(d_{N}^{(1)}), Im(d_{N}^{(1)}), Re(d_{N}^{(2)}), Im(d_{N}^{(2)}), ..., Re(d_{N}^{(m)}), Im(d_{N}^{(m)})
    deformationd_arr = params[n_phases + n_unit: n_phases + n_unit + n_deform].reshape(N, 2*m)
    deformationf_arr = params[n_phases + n_unit + n_deform: n_phases + n_unit + 2*n_deform].reshape(N, 2*m)
    
    # Superposition. Order is as follows:
    # Re(alpha^{(1)}), Im(alpha^{(1)}), Re(alpha^{(2)}), Im(alpha^{(2)}), ..., Re(alpha^{(m)}), Im(alpha^{(m)})
    superposition_arr = params[n_phases + n_unit + 2*n_deform:]
    
    ### |eta⟩ array.
    # Set coefficients of |eta⟩ to 0.
    eta_arr = np.zeros(2**N).astype(complex)
    
    # Set superposition, deformation and gates.
    for vector in range(2**N):
        # Get sites for which spin is in state |1⟩. Ex: for |010110⟩ the ones array will be [2, 4, 5]
        # Note on format; f'0{N}b' means to format the number (vector) as a binary string of length N
        # and use 0's for padding if necessary.
        ones = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '1']
        # Do the same for |0⟩.
        zeroes = [site for site, digit in enumerate(format(vector, f'0{N}b'), start=1) if digit == '0']
        
        ### Superposition and deformation.
        # We want to set sum_j alpha^{j} prod_ones d_{ones}^{j}
        for superpos_idx in range(m):
            # Set superposition coefficient.
            temp_coeff = superposition_arr[2*superpos_idx] + superposition_arr[2*superpos_idx + 1]*1j 
            
            # For each spin in state |1⟩, multiply by deformation coefficient.
            for site in ones:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationd_arr[site-1, 2*superpos_idx] + deformationd_arr[site-1, 2*superpos_idx+1]*1j
            
            # Do the same for |0⟩.
            for site in zeroes:
                # Multiply by deformation coefficients for site and current superposition index.
                temp_coeff *= deformationf_arr[site-1, 2*superpos_idx] + deformationf_arr[site-1, 2*superpos_idx+1]*1j
            
            # Add to vector coefficient
            eta_arr[vector] += temp_coeff
            
        ### Phase gates.
        # Get suitable pairs of sites. Ex: [2, 4, 5] gives [(2, 4), (2, 5), (4, 5)]
        site_combinations = list(combinations(ones, 2))

        # Set phase to be zero.
        phase = 0

        # For every pair, add corresponding phase to the phase variable.
        for pair in site_combinations:
            # Get index for corresponding phase, as with full symmetryzation.
            # For varphi_{i, j}, it will be the minimum of j-i-1 and (i+N)-j-1
            # This will also decide which of the two phases to use.
            # If j-i is the smaller, then i is the "first site". If that's not
            # the case, then j is the "first site".
            if pair[1] - pair[0] < pair[0] + N - pair[1]:
                phase_idx = pair[1] - pair[0] - 1
                
                # Add phase to the phase variable.
                phase += phases_arr[pair[0]%2][phase_idx]
                
            else:
                phase_idx = pair[0] + N - pair[1] - 1
                
                # Add phase to the phase variable.
                phase += phases_arr[pair[1]%2][phase_idx]
            

        # Multiply vector coefficient by exp[i(sum of phases)].
        eta_arr[vector] *= np.exp(phase*1j)
            
    
    ### Unitaries.
    if unitaries:
        # Set list to store matrices.
        unitaries_list = []

        # Set matrices and store them.
        for site in range(N):
            # Get beta, gamma and delta parameters.
            beta = unitaries_arr[3*site]
            gamma = unitaries_arr[3*site + 1]
            delta = unitaries_arr[3*site + 2]

            # Set matrix elements.          
            u11 = np.exp(beta*1j) * np.cos(delta)
            u12 = -np.exp(-gamma*1j) * np.sin(delta)
            u21 = -np.conjugate(u12)
            u22 = np.conjugate(u11)

            # Add matrix to list.
            unitaries_list.append(np.array([u11, u12, u21, u22]).reshape(2, 2))

        # Calculate tensor product times |eta⟩.
        eta_arr = kron_vec_prod(unitaries_list, eta_arr)
    
    
    ### Correlation
    corr_matrix_xy = np.zeros((N, N))
    
    for spin_1 in range(N):
        for spin_2 in range(spin_1, N):
            # Set array to store sigma_{spin_1}^(x) sigma_{spin_2}^(y) |eta⟩ 
            temp_eta =  np.zeros(len(eta_arr)).astype(complex)
            for vector, coefficient in enumerate(eta_arr): 
                # For every pair of sites, one, and only one, sigma combination
                # will act for S^{x} and S^{y}.
                     
                # If 2nd spin is down.
                if bitset(vector, spin_2, N): 
                    # Add calculated coefficient to array. 
                    # Instead of multiplying by 4 (due to the ladder operators)
                    # we divide by 4 (instead of by 16) when taking the expectation values.
                    temp_eta[flipanybits(vector, spin_1, spin_2, N)] += coefficient

                # If 2nd spin is up.
                else:
                    # Add calculated coefficient to array.
                    temp_eta[flipanybits(vector, spin_1, spin_2, N)] -= coefficient
                    
            # Store expectation values in matrix with appropriate factors.
            corr_matrix_xy[spin_1, spin_2] = np.real(1j*np.vdot(eta_arr, temp_eta)) / np.real(-4*np.vdot(eta_arr, eta_arr))
            corr_matrix_xy[spin_2, spin_1] = corr_matrix_xy[spin_1, spin_2]
            
    # Return array
    return corr_matrix_xy