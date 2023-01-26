import numpy as np                  # Linear algebra and mathematical functions
import scipy.constants as cn        # Physical constants
import scipy.integrate as si        # Numerical integration methods (Romberg)
# from multiprocessing import Pool  # Performing operations in parallel


# ====== General Maths Functions

factorial_lookup = np.array([
    np.math.factorial(i) for i in range(50)
], dtype=np.float64)


def factorial(n):
    return factorial_lookup[n]


def rising_fact(z: complex, n: int) -> complex:
    '''
    Calculates the rising factorial of z, often written
    (z)^n = z * (z+1) * (z+2) * (z+3) ... (z+n-1)
    although (z)_n is also used; although that is also sometimes
    used for the falling factorial, so check the definitions.
    This notation is referred to as Pochammer's Symbol, see
    Abramowitz & Stegun, 6.1.22
    '''

    # If compiled with numba and z specified as an integer, then the
    # integer representation can overflow; although pure python would not.
    # To overcome this, total could be specified as a float so that larger
    # numbers can be represented. See tests for verified range of inputs.
    total = 1.0

    for k in range(n):
        total *= (z + k)

    return total


def Lna(n: int, alpha: float, z: complex) -> complex:
    '''
    Computes the generalised Laguerre polynomial, L_{n}^{(\alpha)}(z),
    with the condition that n, a are real, and z is in general complex.
    Whilst Laguerre polynomials are only orthogonal for a > -1, they
    can be calculated for the case where a <= -1.

    This definition comes from https://doi.org/10.4134/CKMS.c200208
    Specifically equation 15, with p=1, as specified after Equation 6.
    '''

    # An alternative implementation seems to work better for large n and alpha,
    # but is slower. It can be made to be fast using lru_cache, but I haven't
    # gotten this to work with numba.
    # def Lna(n:int, m:float, x:complex) -> complex:
    #     '''
    #     Computes the generalised Laguerre polynomial, L_{n}^{(\alpha)}(z),
    #     with a recurrence relation.
    #     '''

    #     if n == 0:
    #         return np.ones(len(x))
    #     elif n == 1:
    #         return 1 + m - x
    #     else:
    #         return ( (2*(n-1) + 1 + m - x) * Lna(n-1, m, x) - ( (n-1) + m) * Lna(n-2, m, x) )/n

    total = np.zeros(len(z), dtype=np.float64)
    mult = np.ones(len(z), dtype=np.float64)
    sign = +1

    for k in range(n + 1):
        total += (sign / factorial(k) / factorial(n - k)
                  * rising_fact(k + alpha + 1, n - k) * mult)
        mult = mult * z
        sign = -sign

    return total


# =========== Functions for Unperturbed Distribution ===========

def Ilnk(w: float, a: float, l: int, n: int, k: int, beta: float) -> float:
    '''
    Not used in practice to avoid re-computing constants. Left for research.

    w = (w' - w_{xi})
    a = constant used in Laguerre series expansion of unperturbed distribution
    l = azimuthal mode number
    n = radial mode number
    k = sum index
    '''
    absl = np.abs(l)
    absw = np.abs(w)
    sigma = 0

    return ((np.sign(w * l))**absl
            * (-1)**(n + k)
            * (absw / beta / cn.c / 2 / a)**absl
            * np.exp(-absw**2 / beta**2 / cn.c**2 / 4 / a)
            * Lna(n, sigma - n + k, absw**2 / beta**2 / cn.c**2 / 4 / a)
            * Lna(k, absl - sigma + n - k,
                  absw**2 / beta**2 / cn.c**2 / 4 / a))


def generateGksumDict(Gk_array, w, a, max_l, max_n, beta):
    '''
    Computes
    Sum_{k=0}^{len(Gk_array)} G_{k} * I _{lnk}(w, a, l, n, k)

    for all possible combinations of l and n and returns a
    dictionary of values. This is usually used to avoid having to recompute
    Gksum for every element of the matrix (size (max_n*(2*max_l + 1))**2)
    when there are only (max_n*(2*max_l + 1)) possible combinations of l and n.

    max_l = positive integer (l = -max_l, ..., -1, 0, 1, ..., max_l)
    max_n = positive integer (n = 0, 1, 2, ..., n-1, n)

    '''
    Gksum_dict = {}

    # pre-calculate any variables which don't depend on l, n or k
    # outside of the loops.
    absw = np.abs(w)
    signw = np.sign(w)
    const_1 = absw / beta / cn.c / 2 / a
    lna_argument = absw**2 / beta**2 / cn.c**2 / 4 / a
    coef = np.exp(-lna_argument)
    sigma = 0

    # preallocate some memory
    n_array = np.zeros(shape=(max_n + 1, len(w)), dtype=np.float64)
    for l in range(max_l + 1):
        # Calculate everything that depends only on l (not n or k)
        # outside of the nested loops
        coef_l = signw**l * (const_1)**l

        for n in range(max_n + 1):
            # Now we loop over k, calculate the result of the integral,
            # multiply by the Laguerre series coefficient and add it.
            # total is the result of this summation.
            total = np.zeros(len(w))
            for k, Gk in enumerate(Gk_array):
                total += (Gk * (-1)**(n + k)
                          * Lna(n, sigma - n + k, lna_argument)
                          * Lna(k, l - sigma + n - k, lna_argument))

            n_array[n] = total

        Gksum_dict[l] = coef_l * coef * n_array
        Gksum_dict[-l] = (-1)**l * Gksum_dict[l]

    return Gksum_dict


def g0Hat(r, a, Gk):
    total = np.zeros(len(r), dtype=np.float64)
    for k, i in enumerate(Gk):
        total += i * Lna(k, 0, a * r**2)

    return np.exp(-a * r**2) * total

# =========== Functions for Perturbed Distribution ===========


def Qln(w: float, a: float, l: int, n: int, beta: float) -> float:
    '''
    Not used in practice to avoid re-computing constants. Left for research.

    Returns the result Q_{l'n'} which is the coefficient of alpha_{l'n'}
    in the Vlasov equation.
    w = (w' - w_{xi})
    a = constant used in Laguerre series expansion of unperturbed distribution
    l = azimuthal mode number
    n = radial mode number
    '''
    absl = abs(l)
    sqrta = np.sqrt(a)

    return (np.sign(l)**absl
            * a**(absl / 2 - 1) * 1 / (2 * factorial(n))
            * (1 / 2 / sqrta * w / beta / cn.c)**(2 * n + absl)
            * np.exp(-1 / 4 / a * w**2 / beta**2 / cn.c**2))


def generateQlnDict(w: float, a: float, max_l: int, max_n: int, beta: float) -> float:
    '''
    w = (w' - w_{xi})
    a = constant used in Laguerre series expansion of unperturbed distribution
    max_l = maximum azimuthal mode number
    max_n = maximum radial mode number

    Calculates Q_{ln}(w' - w_{xi}) for different combinations of
    l and n and returns the result in a dictionary. This is to avoid
    having to recompute Q_{ln} for same combinations of l and n more than once.
    '''
    Qln_dict = {}

    # Any parts of the formula that don't depend on n or l don't
    # need to be performed inside the loop. Doing them outside
    # speeds up these calculations.
    sqrta = np.sqrt(a)
    coef = np.exp(-1 / 4 / a * w**2 / beta**2 / cn.c**2)
    const_1 = 1 / 2 / sqrta * w / beta / cn.c

    # This factorial only depends on n, not l.
    # If it is computed inside the loop then np.math.factorial will
    # run len(ll) * len(nn) times, but it only needs to be performed
    # len(nn) times. Calculate all the factorials first, then index
    # them inside the loop.
    facn = factorial(np.arange(int(max_n) + 1))

    # n starts from zero, so can select n from the index of the array
    # rather than using a dictionary.
    n_array = np.zeros(shape=(max_n + 1, len(w)))

    # only positive values of l, and symmetry will be used to
    # find the value for the corresponding negative values
    for l in range(max_l + 1):
        for n in range(max_n + 1):
            n_array[n] = 1 / (2 * facn[n]) * (const_1)**(2 * n + l)

        # multiply the coefficients now, not inside n loop
        Qln_dict[l] = coef * a**(l / 2 - 1) * n_array
        Qln_dict[-l] = (-1)**l * Qln_dict[l]

    return Qln_dict


def calculateg1Point(a, eigenvectors, r, phi, max_l, max_n,
                     beta, chromaticity, w_b, eta):
    '''
    r - float, the radial coordinate to evaluate g_1(r, phi)
    phi - float, the angular coordinate to evaluate g_1(r, phi)
    eigenvectors - list, a list containing all eigenvectors
    eigenvalue_index - integer, the eigenvalue for which to compute g_1(r, phi)

    This function evaluates g_1(r, phi) for a particular eigenvalue.
    It takes the eigenvectors associated with a particular eigenvalue
    and then performs the summations required to produce a distribution.
    '''
    l_list = np.arange(-int(max_l), int(max_l) + 1)

    ar2 = np.exp(-a * r**2)  # calculate this outside the loop for speed

    # The eigenvectors are in the order of
    # alpha_{l=-10,n=0}
    # alpha_{l=-9, n=0}
    # ...
    # alpha_{l=+10,n=0}
    # alpha_{l=-10,n=1}
    # alpha_{l=-9, n=1}
    # ...
    #
    # So for each value of n we want to sum over all the different values
    # of l. There are (2*max_l + 1) values of l (e.g. -3, -2, -1, 0, 1, 2, 3).
    # So list of eigenvectors corresponding to n have the starting
    # index n*(2*max_l+1). So for max_l=3, the starting indexes would
    # be 0, 7, 14 etc.
    # index, l
    # 0 , -3
    # 1 , -2
    # 2 , -1
    # 3 , 0
    # 4 , 1
    # 5 , 2
    # 6 , 3
    # 7 , -3
    #
    # The end limit is then just n*(2*max_l+1) + 2*max_l+1 = 2*(max_l+1)*(n+1)
    # To verify this, a quick bit of code is
    # test_max_l = 3
    # test_max_n = 2
    # for n in range(0, test_max_n+1):
    #     for l in test[n*(2*test_max_l+1):(2*test_max_l+1)*(n+1)]:
    #         print(f'{n}{l}')
    total = np.zeros(1, dtype=np.complex128)
    for n in range(0, max_n + 1):
        for l, alpha_ln in zip(l_list, eigenvectors[n * (2 * max_l + 1):(2 * max_l + 1) * (n + 1)]):
            total += a**np.abs(l) * r**np.abs(l) * ar2 * np.exp(1j * l * phi) * alpha_ln * Lna(n, np.abs(l), np.array([a * r**2]))

    # Need z to include the head-tail phase factor
    z = r * np.cos(phi)
    return total * np.exp(1j * chromaticity * w_b * z / eta / beta / cn.c)


def g1(grid_size, a, eigenvectors, max_r, max_l, max_n, beta, chromaticity, w_b, eta):
    zz = np.linspace(-max_r, max_r, grid_size)
    dd = np.linspace(-max_r, max_r, grid_size)

    mat = np.zeros(shape=(grid_size, grid_size), dtype=np.complex128)

    for n, i in enumerate(zz):
        for m, j in enumerate(dd):
            phi = np.mod(np.arctan2(j, i) + 2 * np.pi, 2 * np.pi)  # to get 0 - 2pi
            r = np.sqrt(i**2 + j**2)
            mat[-m, n] = calculateg1Point(a, eigenvectors, r, phi, max_l, max_n, beta, chromaticity, w_b, eta)[0]

    return mat


# =========== Helper Functions                  ===========

def generateFactCoefficient(max_l: int, max_n: int) -> float:
    '''
    This function is for calculating the coefficient
    n! / Gamma(|l| + n + 1) for every value of n to avoid
    having to recalculate it for every element.

    Since |l| and n are integers it is possible to calculate
    this as
    n! / (|l| + n)!
    and the np.math.factorial functions are faster than using
    the Gamma function.

    I haven't experienced overflow errors from these functions
    but note that if they occur then this is an ideal situation for
    the gammaln() function which calculates the log of the gamma
    function which avoids overflows. Afterwards calculate the
    difference log(Gamma(n + 1)) - log(Gamma(|l| + n + 1)) = log((n+1)/(|l| + n + 1))
    then take the exp. This is not the default because I have
    found it to be slower than using the factorial functions.
    '''

    gamma_coef = {}
    nfact = factorial(np.arange(int(max_n) + 1))
    for l in range(max_l + 1):
        # Since n has indexes 0 - n, the key of a list is the index of n
        n_array = nfact / factorial(np.arange(l, l + max_n + 1))

        gamma_coef[l] = n_array

    return gamma_coef

# =========== Functions for Generating Matrices ===========


def generateLMatrix(max_l, max_n):
    matrix = np.zeros(((2 * max_l + 1) * (max_n + 1), (2 * max_l + 1) * (max_n + 1)), np.complex128)
    np.fill_diagonal(matrix, [i for i in range(-max_l, max_l + 1)] * (max_n + 1))
    return matrix


def generateSimpleBaseMatrix(max_l: int, max_n: int, a: float, Gk: np.ndarray,
                             sample_frequencies: np.ndarray, sampled_impedance: np.ndarray,
                             f0: float, num_bunches: float, beta: float, w_xi: float) -> np.ndarray:
    '''
    max_l = max number of azimuthal modes
    max_n = max number of radial modes
    a     = constant from unperturbed distribution fit
    Gk = Laguerre series coefficients for the longitudinal distributions
    sample_frequencies = frequencies at which the impedance was sampled (f, not omega)
    sampled_impedance = dipolar impedance (Ohm/m) sampled at the sample frequencies
    f0    = Accelerator revolution frequencies
    num_bunches = number of bunches (bunches assumed identical and equidistant)
    beta  = relativistic beta (v/c)
    w_xi  = head-tail phase shift angle
    '''
    fp = sample_frequencies
    wp = 2 * np.pi * fp
    w = wp - w_xi

    # Make a list of all the values of l that will be calculated, this gets repeated a few times
    l = np.arange(-int(max_l), int(max_l) + 1)

    # Preliminarily work out all of the sum of Gk * Integrals
    Gksums = generateGksumDict(Gk, w, a, max_l, max_n, beta)

    # Preliminarily work out all values of Q_{l'n'}
    Qln_vals = generateQlnDict(w, a, max_l, max_n, beta)

    # Generate coefficients which depend on n
    gamma_coef = generateFactCoefficient(max_l, max_n)

    # Allocate memory for the matrix
    matrix = np.zeros(shape=((2 * max_l + 1) * (max_n + 1),
                             (2 * max_l + 1) * (max_n + 1)),
                      dtype=np.complex128)

    # populate the matrix
    # If max_l=2, and max_n=1 the values of (l, l') are
    #                  n' = 0                                     n' = 1
    #         (-2,-2) (-2,-1) (-2,+0) (-2,+1) (-2,+2)    (-2,-2) (-2,-1) (-2,+0) (-2,+1) (-2,+2)
    #         (-1,-2) (-1,-1) (-1,+0) (-1,+1) (-1,+2)    (-1,-2) (-1,-1) (-1,+0) (-1,+1) (-1,+2)
    # n = 0   (+0,-2) (+0,-1) (+0,+0) (+0,+1) (+0,+2)    (+0,-2) (+0,-1) (+0,+0) (+0,+1) (+0,+2)
    #         (+1,-2) (+1,-1) (+1,+0) (+1,+1) (+1,+2)    (+1,-2) (+1,-1) (+1,+0) (+1,+1) (+1,+2)
    #         (+2,-2) (+2,-1) (+2,+0) (+2,+1) (+2,+2)    (+2,-2) (+2,-1) (+2,+0) (+2,+1) (+2,+2)
    #
    #         (-2,-2) (-2,-1) (-2,+0) (-2,+1) (-2,+2)    (-2,-2) (-2,-1) (-2,+0) (-2,+1) (-2,+2)
    #         (-1,-2) (-1,-1) (-1,+0) (-1,+1) (-1,+2)    (-1,-2) (-1,-1) (-1,+0) (-1,+1) (-1,+2)
    # n = 1   (+0,-2) (+0,-1) (+0,+0) (+0,+1) (+0,+2)    (+0,-2) (+0,-1) (+0,+0) (+0,+1) (+0,+2)
    #         (+1,-2) (+1,-1) (+1,+0) (+1,+1) (+1,+2)    (+1,-2) (+1,-1) (+1,+0) (+1,+1) (+1,+2)
    #         (+2,-2) (+2,-1) (+2,+0) (+2,+1) (+2,+2)    (+2,-2) (+2,-1) (+2,+0) (+2,+1) (+2,+2)
    #
    lenl = len(l)
    temp = np.zeros(len(fp), dtype=np.complex128)
    for nn in range(max_n + 1):  # rows, n
        for ii, i in enumerate(range(-max_l, 1)):  # rows, l
            temp = Gksums[i][nn] * sampled_impedance  # calculate this outside the next loop for speed
            for nnp in range(max_n + 1):  # cols, n'
                for jj, j in enumerate(range(-max_l, 1)):  # cols, l'
                    matrix[nn * lenl + ii, nnp * lenl + jj] = (1j)**(i - j) * gamma_coef[np.abs(i)][nn] * np.sum(Qln_vals[j][nnp] * temp)

                    # The next three lines are utilising symmetry between (l, -l) and (l', -l')
                    #
                    # The factor i^{l - l'} has to be updated for the sign of l.
                    # i^{l - l'} = (-1)^{l} * i^{-l - l'}
                    # i^{l - -l'} = (-1)^{l'} * i^{l - +l'}
                    # i^{l - l'} = (-1)^{l + l'} * i^{-l - -l'}
                    #
                    # The quantities Q_{l' n'} has to be updated for the new sign of l'
                    # If we stick with only the negative signs of l', then Q_{l',n} = (-1)^{l'} Q_{-l', n}
                    # If -l' -> l' then there is a new factor if (-1)^{l'} * (-1)^{-l} = 1
                    #
                    # The quantities I_{lnk} have to be updated for the new value of l
                    # If we stick with only the negative signs of l, then I_{l,n,k} = (-1)^{l} I_{-l, n, k}
                    # If -l -> l, then this has a new factor of (-1)^l * (-1)^l = (-1)^(2l) = 1
                    #
                    # In all cases the sign is ultimately unchanged, so these matrix elements are the same.
                    #
                    # To stop utilising this symmetry, comment out the next three lines and replace
                    # the (ii, i) and (jj, j) loops to loop over enumerate(l) rather than enumerate(range(-max_l, 1))
                    matrix[nn * lenl + (2 * max_l + 1) - ii - 1, nnp * lenl + jj] = matrix[nn * lenl + ii, nnp * lenl + jj]
                    matrix[nn * lenl + (2 * max_l + 1) - ii - 1, nnp * lenl + (2 * max_l + 1) - jj - 1] = matrix[nn * lenl + ii, nnp * lenl + jj]
                    matrix[nn * lenl + ii, nnp * lenl + (2 * max_l + 1) - jj - 1] = matrix[nn * lenl + ii, nnp * lenl + jj]

    return matrix


def generateFullBaseMatrix(max_l: int, max_n: int, a: float, Gk: np.ndarray,
                           sample_frequencies: np.ndarray, sampled_impedance: np.ndarray,
                           f0: float, num_bunches: float, beta: float, w_xi: float) -> np.ndarray:
    '''
    max_l = max number of azimuthal modes
    max_n = max number of radial modes
    a     = constant from unperturbed distribution fit
    Gk = Laguerre series coefficients for the longitudinal distributions
    sample_frequencies = frequencies at which the impedance was sampled (f, not omega)
    sampled_impedance = dipolar impedance (Ohm/m) sampled at the sample frequencies
    f0    = Accelerator revolution frequencies
    num_bunches = number of bunches (bunches assumed identical and equidistant)
    beta  = relativistic beta (v/c)
    w_xi  = head-tail phase shift angle
    '''
    fp = sample_frequencies
    wp = 2 * np.pi * fp
    w = wp - w_xi

    # Make a list of all the values of l that will be calculated, this gets repeated a few times
    l = np.arange(-int(max_l), int(max_l) + 1)

    # Preliminarily work out all of the sum of Gk * Integrals
    # Since the index of w is l, this can just be done in a loop over w
    Gksums = []
    for i in w:
        Gksums.append(generateGksumDict(Gk, i, a, max_l, max_n, beta))

    # Preliminarily work out all values of Q_{l'n'}
    # Since the index of w is l, this can just be done in a loop over w
    Qln_vals = []
    for i in w:
        Qln_vals.append(generateQlnDict(i, a, max_l, max_n, beta))

    # Generate coefficients which depend on n
    gamma_coef = generateFactCoefficient(max_l, max_n)

    # Allocate memory for the matrix
    matrix = np.zeros(shape=((2 * max_l + 1) * (max_n + 1),
                             (2 * max_l + 1) * (max_n + 1)),
                      dtype=np.complex128)

    # populate the matrix
    lenl = len(l)
    temp = np.zeros(len(fp), dtype=np.complex128)
    for nn in range(max_n + 1):  # rows, n
        for ii, i in enumerate(l):  # rows, l
            temp = Gksums[i][i][nn] * sampled_impedance[i]  # calculate this outside the next loop for speed
            for nnp in range(max_n + 1):  # cols, n'
                for jj, j in enumerate(l):  # cols, l'
                    matrix[nn * lenl + ii, nnp * lenl + jj] = (1j)**(i - j) * gamma_coef[np.abs(i)][nn] * np.sum(Qln_vals[i][j][nnp] * temp)

    return matrix


def generateInteractionMatrix(base_matrix, l_matrix, protons_per_bunch, num_bunches, mass, beta, f0, w_b, w_s):
    '''
    base_matrix = a matrix whose elements are given by
                         M[i, j] = SUM(J_i(w') * J_j(w') * Z_(perp)(w'))
                         where the sum is over harmonics.
    '''
    gamma = 1 / np.sqrt(1 - beta**2)
    r0 = cn.e**2 / mass / cn.c**2

    # The elements in zperp_j_sum_matrix correspond to the summation terms which
    # are in every element. They are missing a coefficient, which depends on the
    # number of particles per bunch. This coefficient is the same for all elements,
    # so multiply the whole matrix by it.
    kl_coef = -1j * protons_per_bunch * r0 * cn.c * f0**2 / (2 * gamma * w_b * w_s * beta) * num_bunches
    matrix = kl_coef * base_matrix

    # There is an additional term in the matrix l * delta_{l l'} * delta_{n n'}.
    # These terms appear along the main diagonal, and do not have the same coefficient
    # as the other term in the matrix. Generate a diagonal matrix with the values of
    # l, repeating for each n. Add this on.
    matrix = matrix + l_matrix

    # The matrix has now been entirely generated.
    return matrix


# ============== Compile Code Above
# If the optional dependency numba is available then the code below
# will automatically decorate all functions already defined with a
# function that creates a compiled, optimised version of the same
# function. Types are inferred from the first call to the function,
# then subsequent calls use the compiled version.
# If numba is not installed then the functions will produce the same
# output, but will take more time to complete.
try:
    from numba import jit_module
    jit_module(nopython=True,
               cache=True,
               nogil=True, fastmath=True)
    print("Using Numba optimised Laguerre methods.")

except ModuleNotFoundError:
    print("Using native Python methods for Laguerre Methods.")
    print("Consider installing numba for compiled and parallelised methods.")


# ================= Uncompiled Functions below ========================
# For numba to compile functions it restricts what numpy functions
# are used. In situations where functions are only used once and
# aren't very demanding, there isn't much to gain by rewriting code
# to use the limits selection of numpy functions. Such functions
# are placed below this line.


# =========== Numerical Methods
# Functions for numerically integrating functions
# These are used to ensure the proper normalisation numerically.

def midpoint(func, a, b, n):
    '''
    Computes the midpoint Riemann sum of the function func between the
    limits a and b. Spacing between samples is h = 1/2**n.
    '''
    total = 0
    tttpon = 2**n
    for i in range(int(2**(n - 1))):
        total += func((2 * i + 1) * (a + b) / tttpon)

    return (b - a) / 2**(n - 1) * total


def midpointrichardson(func, a, b, divmax=10, tol=1.0e-8, show=False):
    '''
    Performs Richardson extrapolation on the midpoint Riemann integral of
    the function func between the limits a and b.
    '''
    R = np.zeros((divmax, divmax))
    for i in range(divmax):
        R[i, 0] = midpoint(func, a, b, i + 1)

        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + 1 / (4**(j) - 1) * (R[i, j - 1] - R[i - 1, j - 1])

        if i > 1:
            # No need to do further calculations if we've already converged
            if abs(R[i, i] - R[i - 1][i - 1]) < tol:
                break

    if abs(R[-1][-1] - R[-2][-2]) > tol:
        print("WARNING: Did not converge to required tolerance")

    if show is True:
        print(R)

    return R[i][i]


def normalisation_coefficient(func, split_position):
    '''
    Finds the normalisation coefficient for the function func.
    split_position is required for the numerical integration and
    should be set to a point somewhere near the "middle" of the
    distribution. For example with a gaussian it might be set to
    1*sigma; although the exact value should not affect the result.
    '''
    i1 = si.romberg(lambda x: x * func(x), 0, split_position)
    i2 = midpointrichardson(lambda t: 1 / t**3 * func(1 / t), 0, 1 / split_position)
    normalisation_coefficient = 1 / (i1 + i2)

    return normalisation_coefficient


def normalise(func, normalisation_coefficient):
    '''
    This function redefines an input function, func, by multiplying it
    by the supplied normalisation coefficient.
    '''
    def normalised_func(r):
        return normalisation_coefficient * func(r)

    return normalised_func


# =========== Unperturbed Distribution
def calculateGk(g0hat, a, k, method='midpoint', max_radius=1):
    '''
    Calculate G_k by performing the integral
    G_k = \int_0^\infty ĝ_0(r) L_k^{(0)}(ar^2) d(ar^2)

    max_radius is only significant for trapz and midpoint methods.

    A few methods are provided but all of them should do the same thing.

    Midpoint       - Divides the integral into two. Uses standard romberg integration
                     for the region 0 to max_radius/10, and then the midpoint for
                     the region max_radius/10 to infinity. Max_radius is used to split
                     the integral up, but its value is not hugely significant.

    Laguerre-Gauss - This is effectively fitting a high order polynomial and
                     calculating the integral of that. This is faster, but not
                     necessarily the most accurate, especially for discontinuous
                     functions.

    trapz          - This is the most basic method. It generates an array of 5000
                     points between 0 and max_radius and uses the trapezium rule.
                     max_radius is important with this method. If it is too small
                     then the region beyond it won't be included in the integral.
                     If it is too large the steps between samples will be too wide.

    '''
    if method.lower() == 'laguerre-gauss':
        # Note that since g0hat does not necessarily have the required weight
        # function, the inverse of the weight function np.exp(x) is explicitly
        # multiplied by g0hat.
        g, w = np.polynomial.laguerre.laggauss(100)
        return np.sum(w * np.exp(g) * g0hat(np.sqrt(g / a)) * Lna(k, 0, g))

    elif method.lower() == 'midpoint':
        Gkint_1 = si.romberg(lambda u: g0hat(np.sqrt(u / a)) * Lna(k, 0, np.array([u])), 0, max_radius / 10, divmax=15)[0]
        Gkint_2 = midpointrichardson(lambda t: 1 / t**2 * g0hat(np.sqrt(1 / t / a)) * Lna(k, 0, np.array([1 / t])), 0, 10 / max_radius, divmax=15)
        return Gkint_1 + Gkint_2

    elif method.lower() == 'trapz':
        u = np.linspace(0, max_radius**2 * a, 5000)
        y = g0hat(np.sqrt(u / a)) * Lna(k, 0, u)
        return np.trapz(y, u)
