import numpy as np                  # Linear algebra and mathematical functions
import scipy.constants as cn        # Physical constants
import scipy.special as sp
import scipy.integrate as si


def generateSimpleBaseMatrix(max_l: int, zhat,
                             sampled_frequencies: np.ndarray,
                             sampled_impedance: np.ndarray,
                             f0: float, num_bunches: float,
                             beta: float, w_xi: float) -> np.ndarray:
    '''
    Provided l*ws is neglected in the frequency sampling, the bessel functions
    and impedance are sampled at the same frequency for every matrix element.
    '''
    fp = sampled_frequencies
    wp = 2 * np.pi * fp
    w = wp - w_xi

    # Preliminarily work out all of the required bessel functions. This avoids
    # calling them inside the nested loop since we only need to calculate J for
    # each value of l.
    #
    # By making this a dictionary rather than an array we can still use i and j
    # (e.g. if we set matrixSize=6 i=(-3, -2, -1, 0, 1, 2, 3) etc)
    # to index, whereas if we used an array we'd have to work out what that l=1
    # was index 4.
    #
    # We index 0 here, so that we're sampling the frequency with 0 * w_s in the
    # sample frequency.
    jdict = generateBesselJDict1D(max_l, w * zhat / beta / cn.c)

    # Allocate memory for the matrix
    matrix = np.zeros(shape=(2 * max_l + 1, 2 * max_l + 1),
                      dtype=np.complex128)

    # populate the matrix
    # Note that since this is the sum J_{i} * J_{j}, and the frequencies are
    # same for both bessel functions, it doesn't matter the
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
    # If (-2,-1) = (-1,-2) then this matrix is reflected around main diagonal.
    #
    # Top loop is looping over rows and we evaluate all values of l for rows.
    # The inner loop is then looping over columns and only needs to go to the
    # main diagonal. Values off the main diagonal are then reflected.
    for ii, i in enumerate(range(-max_l, 1)):
        temp = sampled_impedance * jdict[i]
        for jj, j in enumerate(range(-max_l, 1)):
            # # For checking value of l'
            # matrix[ii, jj] = j
            # matrix[ii, (2*max_l+1)-1-jj] = -j  # l->l, l'->-l'
            # matrix[(2*max_l+1)-1-ii, jj] = j  #l->-l, l'->l'
            # matrix[(2*max_l+1)-1-ii, (2*max_l+1)-1-jj] = -j # l->-l, l'->-l'

            matrix[ii, jj] = 1j**(i - j) * np.sum(temp * jdict[j])

            # Since Omega ~ pw_0 + wb is assumed
            # It is the case that the value of the sampled impedance is the
            # same for all terms, whilst the bessel function inside the
            # summation can change. The relationship between signs of order
            # of bessel functions let us only evaluate for one sign of l and l'
            # then calculate the rest. We then only evaluate quarter of the
            # overall matrix.
            # i^{l - l'} = (-1)^{l} * i^{-l - l'}
            # i^{l - -l'} = (-1)^{l'} * i^{l - +l'}
            # i^{l - l'} = (-1)^{l + l'} * i^{-l - -l'}
            # and in both cases J_{-l} = (-1)^l * J_{-l}, so the two factors
            # multiply
            # (-1)^l * (-1)^l = (-1)^(2l) = +1,
            # and make the matrix elements the same. To stop using this
            # symmetry comment out the next three lines and replace
            # the loops for enumerate(range(-max_l, 1)) with enumerate(l)
            #
            # Note that there is symmetry not being exploited here, namely,
            # the fact that the airbag is symmetric with l and l'.
            matrix[ii, (2 * max_l + 1) - 1 - jj] = matrix[ii, jj]  # l>l,l'>-l'
            matrix[(2 * max_l + 1) - 1 - ii, jj] = matrix[ii, jj]
            matrix[(2 * max_l + 1) - 1 - ii, (2 * max_l + 1) - 1 - jj] = matrix[ii, jj]

    return matrix


def generateFullBaseMatrix(max_l: int, zhat,
                           sampled_frequencies: np.ndarray,
                           sampled_impedance: np.ndarray,
                           f0: float, num_bunches: float,
                           beta: float, w_xi: float) -> np.ndarray:
    '''

    '''
    fp = sampled_frequencies
    wp = 2 * np.pi * fp
    w = wp - w_xi

    # Make a list of all values of l to be used, this gets repeated
    l = np.arange(-int(max_l), int(max_l) + 1)

    # Preliminarily work out all of the required bessel functions. Unlike
    # with the perturbed-simple method, this doesn't actually save
    # time, because sp.jn() has to be run for every value of l and l'.
    # This was kept as-is so that the airbag and NHT solvers remain similar.
    jdict = generateBesselJDict2D(max_l, w * zhat / beta / cn.c)

    # Allocate memory for the matrix
    matrix = np.zeros(shape=(2 * max_l + 1, 2 * max_l + 1),
                      dtype=np.complex128)

    # populate the matrix
    # Top loop is looping over rows and we evaluate all values of l for rows.
    # The inner loop is then looping over columns which are for values of l'.
    # Symmetry here is limited because l changes the sample frequencies.
    for ii, i in enumerate(l):
        temp = sampled_impedance[i] * jdict[i][i]
        for jj, j in enumerate(range(-max_l, 1)):
            matrix[ii, jj] = 1j**(i - j) * np.sum(temp * jdict[i][j])
            matrix[ii, -jj - 1] = matrix[ii, jj]

    return matrix


def generateBesselJDict2D(max_order, x):
    jdict = []
    for i in x:
        jdict.append(generateBesselJDict1D(max_order, i))

    return jdict


def generateBesselJDict1D(max_order, x):
    '''
    n is an integer with the values
    -max_order, -max_order+1, -max_order+2, ..., 0, 1, 2, ..., max_order

    This function returns a dictionary of bessel functions of order n
    evaluated at x. The index of the dictionary is the order of the
    bessel function.

    For example
    jdict = besselJ_dict(3, x)
    jdict[-3] = J_(-3)(x)
    jdict[-2] = J_(-2)(x)
    ...
    jdict[2] = J_2(x)
    etc
    '''
    jdict = np.zeros((2 * max_order + 1, len(x)))
    jdict[0] = np.array([sp.j0(i) for i in x])
    jdict[1] = np.array([sp.j1(i) for i in x])
    jdict[-1] = -jdict[1]

    l = range(2, max_order + 1)

    for ll in l:
        jdict[ll] = np.array([sp.jn(float(ll), i) for i in x])
        jdict[-ll] = (-1)**(ll) * jdict[ll]  # faster than using jv again

    return jdict


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
    print("Using Numba optimised Airbag methods.")

except ModuleNotFoundError:
    print("Using native Python methods for Airbag Methods.")
    print("Consider installing numba for compiled and parallelised methods.")


def generateBroadbandBaseMatrix(max_l: int, zhat,
                                upper_limit: np.ndarray,
                                zperp,
                                f0: float, num_bunches: int,
                                beta: float, w_xi: float) -> np.ndarray:
    '''
    Prior to Chao's Equation 6.184, the value of Ω in the definition of
    ω' remains unknown; note that Equation 6.180 is listed in the errata.
    The perturbation technique that resulted in Equation 6.184 is used
    extensively to obtain subsequent results, as it defines the frequencies
    at which the impedance and other functions are sampled. This
    technique is the basis of PyVlasovSolver's base_matrix_perturbed methods,
    and makes the assumption that frequency shifts are small compared to
    the zero-intensity frequencies.

    An alternative approach is first introduced in Chao's Equation 6.213
    and used in Equation 6.222, which is to change the summation to an integral.
    This assumes the impedance (and other functions) change very slowly on the
    scale between harmonics, so that effectively the summation is similar to
    a Riemann sum, which is an approximation of an integral. Chao describes
    it as a "broadband impedance" model or a "single-turn wake" model.

    '''

    # Some simplified impedance definitions don't evaluate well at 0, so
    # add a very small inequality between the magnitude of the upper and
    # lower limits, making 0 not exactly in the middle.
    lower_limit = -0.9999 * upper_limit

    def airbag_integrand(wp, l, lp):
        integrand = (sp.jn(l, (wp - w_xi) * zhat / beta / cn.c)
                     * sp.jn(lp, (wp - w_xi) * zhat / beta / cn.c)
                     * zperp(wp / 2 / np.pi))
        return integrand

    # Allocate memory for the matrix
    matrix = np.zeros(shape=(2 * max_l + 1, 2 * max_l + 1),
                      dtype=np.complex128)

    for ii, i in enumerate(range(-max_l, 1)):
        for jj, j in enumerate(range(-max_l, 1)):
            re = si.quad(lambda w: np.real(airbag_integrand(w, i, j)), lower_limit, upper_limit)[0]
            im = si.quad(lambda w: np.imag(airbag_integrand(w, i, j)), lower_limit, upper_limit)[0]

            matrix[ii, jj] = 1j**(i - j) * 1 / (2 * np.pi * f0 * num_bunches) * (re + 1j * im)

            matrix[ii, (2 * max_l + 1) - 1 - jj] = matrix[ii, jj]  # l>l,l'>-l'
            matrix[(2 * max_l + 1) - 1 - ii, jj] = matrix[ii, jj]
            matrix[(2 * max_l + 1) - 1 - ii, (2 * max_l + 1) - 1 - jj] = matrix[ii, jj]

    return matrix
