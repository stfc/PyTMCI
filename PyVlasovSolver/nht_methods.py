import numpy as np                   # Linear algebra and mathematical functions
import scipy.constants as cn         # Physical constants
import scipy.integrate as si         # Numerical integration methods (Romberg)
import scipy.optimize as so          # Root finding
from . import airbag_methods as am


def generateSimpleBaseMatrix(max_l: int, ring_radii,
                       sampled_frequencies: np.ndarray, sampled_impedance: np.ndarray,
                       f0:float, num_bunches:float, beta:float, w_b:float, w_xi:float) -> np.ndarray:
    num_rings = len(ring_radii)
    fp = sampled_frequencies
        
    wp = 2*np.pi*fp
    w = wp-w_xi

    # Make a list of all the values of l that will be calculated, this gets repeated a few times
    l = np.arange(-int(max_l), int(max_l)+1)

    # Preliminarily work out all of the required bessel functions. This avoids
    # calling them inside the nested loop since we only need to calculate J for
    # each value of l.
    #
    # By making this a dictionary rather than an array we can still use i and j
    # (e.g. if we set matrixSize=6 i=(-3, -2, -1, 0, 1, 2, 3) etc)
    # to index, whereas if we used an array we'd have to work out what that l=1 
    # was index 4.
    jdict = []
    for i in ring_radii:
        jdict.append(am.generateBesselJDict1D(max_l, w*i/beta/cn.c))

    # Allocate memory for the matrix
    matrix = np.zeros(shape=((2*max_l+1)*num_rings,
                             (2*max_l+1)*num_rings), dtype=np.complex128)
    
    # populate the matrix
    # Note that since this is the sum j_i * j_j, that it doesn't matter the
    # order of the orders(there is no difference between j_i*j_j and j_j*j_i).
    # As such we only need an upper or a lower triangular matrix and then the
    # result will be mirrored. For example if max_l=2, the order of the
    # bessel function orders in the matrix elements are
    # (-2,-2) (-2,-1) (-2,+0) (-2,+1) (-2,+2)
    # (-1,-2) (-1,-1) (-1,+0) (-1,+1) (-1,+2)
    # (+0,-2) (+0,-1) (+0,+0) (+0,+1) (+0,+2)
    # (+1,-2) (+1,-1) (+1,+0) (+1,+1) (+1,+2)
    # (+2,-2) (+2,-1) (+2,+0) (+2,+1) (+2,+2)
    #
    # If (-2,-1) = (-1,-2) then this matrix is reflected around the main diagonal.
    #
    # The top loop is looping over rows and we evaluate all values of l for rows.
    # The inner loop is then looping over columns and only needs to go to the
    # main diagonal. Values off the main diagonal are then reflected.
    lenl = len(l)
    for ring1 in range(num_rings):  # rows, n
        for ring2 in range(num_rings):  # cols, n'
            for ii, i in enumerate(range(-max_l, 1)):  # rows, l

                # Index the sampled impedance with 0*w_s in the frequency to make the 
                temp = jdict[ring1][i] * sampled_impedance
                for jj, j in enumerate(range(-max_l, 1)):  # cols, l'
                    matrix[ring1*lenl + ii, ring2*lenl + jj] = 1/num_rings * (1j)**(i - j) * np.sum(temp * jdict[ring2][j])
                    
                    matrix[ring1*lenl + lenl - ii -1, ring2*lenl + jj] = matrix[ring1*lenl + ii, ring2*lenl + jj]
                    matrix[ring1*lenl + ii, ring2*lenl + lenl - jj -1] = matrix[ring1*lenl + ii, ring2*lenl + jj]
                    matrix[ring1*lenl + lenl - ii -1, ring2*lenl + lenl - jj -1] = matrix[ring1*lenl + ii, ring2*lenl + jj]

    return matrix

def generateFullBaseMatrix(max_l: int, ring_radii,
                           sampled_frequencies: np.ndarray, sampled_impedance: np.ndarray,
                           f0: float, num_bunches: float, beta: float, w_b: float, w_xi: float) -> np.ndarray:
    num_rings = len(ring_radii)
    fp = sampled_frequencies
    wp = 2 * np.pi * fp
    w = wp - w_xi

    # Make a list of all the values of l that will be calculated, this gets repeated a few times
    l = np.arange(-int(max_l), int(max_l) + 1)

    # Preliminarily work out all of the required bessel functions. This avoids
    # calling them inside the nested loop since we only need to calculate J for
    # each value of l.
    # By looping over w, whose index is l this will come out with an index of l too
    jdict = []
    for i in w:
        temp = []
        for j in ring_radii:
            temp.append(am.generateBesselJDict1D(max_l, i * j / beta / cn.c))

        jdict.append(temp)

    # Allocate memory for the matrix
    matrix = np.zeros(shape=((2 * max_l + 1) * num_rings,
                             (2 * max_l + 1) * num_rings), dtype=np.complex128)

    # populate the matrix
    lenl = len(l)
    for ring1 in range(num_rings):  # rows, n
        for ring2 in range(num_rings):  # cols, n'
            for ii, i in enumerate(l):  # rows, l
                temp = jdict[i][ring1][i] * sampled_impedance[i]
                for jj, j in enumerate(l):  # cols, l'
                    matrix[ring1 * lenl + ii, ring2 * lenl + jj] = 1 / num_rings * (1j)**(i - j) * np.sum(temp * jdict[i][ring2][j])

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
               nogil=True, fastmath=True)
    print("Using Numba optimised NHT methods.")

except ModuleNotFoundError:
    print("Using native Python methods for NHT Methods.")
    print("Consider installing numba for compiled and parallelised methods.")


def calculateRingRadii(g0hat, num_rings, search_limit):
    '''
    The NHT splits space into num_rings regions with the outermost
    region extending to infinity and each ring containing the same
    number of particles N' = N/num_rings. This function returns one
    ring radius per region, where half the charge in a region is
    inside the ring and half is outside.

    g0hat - function for unperturbed distribution, with normalisation applied.
    num_rings - number of rings to represent g0hat with
    search_limit - radial position to limit search for ring radii.
                   Typically this is a bit larger than distribution's "radius",
                   for example, 6-10 standard deviations might be typical for a
                   gaussian distribution. If this is set too high, then the
                   algorithm will struggle to converge.
    '''
    assert type(num_rings) == int, "num_rings must be an integer."
    assert num_rings > 0, "num_rings must be a positive integer."
    assert search_limit > 0, "Search limit must be positive."

    target = 1 / num_rings
    region_midpoints = [0]

    def wrapper(upper_limit, lower_limit, target):
        result = si.romberg(lambda r: r * g0hat(r),
                            lower_limit, upper_limit)[0] - target
        return result

    for i in range(num_rings):
        try:
            midpoint = so.root_scalar(wrapper,
                                      args=(0, (i + 0.5) * target),
                                      method='bisect',
                                      bracket=[region_midpoints[i],
                                               search_limit],
                                      # between the last ring and the limit
                                      ).root

            region_midpoints.append(midpoint)

        except:
            raise Exception(
                "Root finder or integral failed to converge.\n"
                + "This is probably because search_limit is too high and the number of " +
                + "loops of the root finder has been exceeded. It could also be because " +
                + "search_limit is too small, so that the root finder can't identify a suitable radius.")

    ring_radii = np.array(region_midpoints[1:])
    return ring_radii