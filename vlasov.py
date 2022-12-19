import numpy as np               # Linear algebra and mathematical functions
import scipy.constants as cn     # Physical constants
import matplotlib.pyplot as plt
import laguerre_methods as lm
import airbag_methods as am
import nht_methods as nht


class accelerator:
    def __init__(self, circumference, mass, charge, beta, Q_b, Q_s, alpha,
                 chromaticity, num_bunches):
        self.circumference = circumference
        self.mass = mass
        self.charge = charge
        self.beta = beta
        self.Q_b = Q_b
        self.Q_s = Q_s
        self.alpha = alpha
        self.chromaticity = chromaticity
        self.num_bunches = num_bunches

        self.gamma = 1 / np.sqrt(1 - beta**2)
        self.f0 = 1 / (circumference / beta / cn.c)
        self.T0 = 1 / self.f0
        self.w0 = 2 * np.pi * self.f0
        self.w_b = 2 * np.pi * Q_b * self.f0
        self.w_s = 2 * np.pi * Q_s * self.f0

        self.eta = alpha - 1 / self.gamma**2
        self.w_xi = self.w_b * self.chromaticity / self.eta


class vlasovSolver():
    def __init__(self, accelerator, max_l, max_n, *args, **kwargs):
        self.max_l = max_l
        self.max_n = max_n
        self.accelerator = accelerator

        self.l_matrix = np.zeros(shape=((2 * max_l + 1) * (max_n + 1),
                                        (2 * max_l + 1) * (max_n + 1)),
                                 dtype=np.complex128)
        self.l_matrix = self.generateLMatrix()

        self.base_matrix = np.zeros(shape=((2 * max_l + 1) * (max_n + 1),
                                           (2 * max_l + 1) * (max_n + 1)),
                                    dtype=np.complex128)

    def _impedanceSampleFrequencies(self, max_harmonics, multi_bunch_mode):
        '''
        Returns the quantities
        $$$
        f' = \frac{\omega'}{2\pi} \approx (p\frac{\omega_0}{2\pi} + \frac{\omega_\beta}{2\pi}),
        $$$
        where $\omega'$ is as defined by Chao's equation 6.237, with the
        approximation that $\Omega \approx \omega_\beta$.

        These can be used to sample an impedance function, or pass to methods
        like generating the base matrix.
        '''
        if multi_bunch_mode >= self.accelerator.num_bunches:
            raise ValueError("Multi-bunch mode cannot be larger than "
                             + "the number of bunches.")

        p = np.arange(-max_harmonics, max_harmonics + 1, dtype=np.float64)

        ll = range(-self.max_l, self.max_l + 1)

        fp = np.zeros((len(ll), len(p)))
        for i in ll:
            fp[i] = ((p * self.accelerator.num_bunches
                      + multi_bunch_mode) * self.accelerator.f0
                     + self.accelerator.w_b / 2 / np.pi
                     + i * self.accelerator.w_s / 2 / np.pi)

        return fp

    def calculateGk(self):
        print("Only for arbitraryLongitudinal distributions.")

    def g0Hat(self):
        pass

    def generateBaseMatrix(self):
        '''
        Generate the base interaction matrix given the impedance sample
        frequencies and an array of transverse dipole impedances sampled at
        these frequencies.

        The method parameter specifies what assumptions are made when
        producing the base matrix, which can reduce computational time
        with no impact on results if used properly, or give totally inaccurate
        results if used improperly.

        For all methods that are 'perturbed-*', use impedanceSampleFrequencies
        helper function to generate the frequencies.

        Possible methods are:
        'perturbed-full' -   Assumes Ω ≈ ⍵_{β} + l * ⍵_{s} from Chao's
                             Equation 6.184. Assumes that deviation from
                             the wake-free or low intensity (N=0) case is
                             small. See Chao's equation for full description.
                             Available symmetry is limited, so this can be
                             resource intensive.

        'perturbed-simple' - Assumes Ω ≈ ⍵_{β}, neglecting (l * ⍵_{s})
                             from Chao's Equation 6.184.
                             Makes same assumptions as perturbed-full, but
                             is only reasonable provided ⍵_{β} >> ⍵_{s}, the
                             impedance is smooth with a resolution of ⍵_{s}
                             and l is not too large.
                             May not be a good assumption if the impedance
                             has a resonance with width on the order of ⍵_{s}.
                             By making this assumption, all functions are
                             evaluated at the same frequencies, significantly
                             reducing computation time by exploiting symmetry.

        '''
        pass

    def generateLMatrix(self):
        '''
        Uses a function specified in laguerre_methods, as that is the
        most general definition available.
        '''
        return lm.generateLMatrix(self.max_l, self.max_n)

    def generateInteractionMatrix(self, N):
        '''
        Uses the supplied particles per bunch N, and 
        the base matrix to calculate an interaction matrix of the
        kind in Chao's Equation 6.223; but not identical to that matrix.

        To understand this method, note that
        in Chao's Equation 6.223 the matrix elements are given by Eq. 6.222,
        which assumes a broadband impedance, χ=0 and an airbag longitudinal
        distribution; neglecting radial modes. This function makes none
        of these assumptions but does follow the general form of
        Equation 6.222 whereby the matrix elements are

        M_{ll'nn'} = l * δ_{ll'} * δ_{nn'} + k * m_{ll'nn'}

        where δ_{ll'} is a Kronecker delta, m_{ll'nn'} is a matrix which
        contains all remaining l, l', n and n' dependence and
        is here referred to as the 'base matrix' and k is a coefficient

        k = -i N r_{0} c / (2 β γ ⍵_{β} ⍵_{s} T_{0}^2)

        which is equivalent to the coefficient in Chao's Equation 6.222
        although it is not identical for several reasons.

        The final interaction matrix is formed by multiplying the
        base matrix by k and adding the Kronecker delta terms; referred
        to here as the l-matrix. This method allows vectorised methods
        to be applied and means that solving for numerous values
        of N can be done without having to recompute the base
        matrix; only k changes.
        '''
        assert self.base_matrix[0, 0] != 0.0, "generateBaseMatrix() must be run before generateInteractionMatrix()." 

        interactionMatrix = lm.generateInteractionMatrix(self.base_matrix,
                                                         self.l_matrix,
                                                         N,
                                                         self.accelerator.num_bunches,
                                                         self.accelerator.mass,
                                                         self.accelerator.beta,
                                                         self.accelerator.f0,
                                                         self.accelerator.w_b,
                                                         self.accelerator.w_s)

        return interactionMatrix

    def g1(self, eigenvectors, r, phi):
        '''
        '''
        print("Not Implemented for this distribution.")

    def Omega(self, eigenvalue):
        return eigenvalue * self.accelerator.w_s + self.accelerator.w_b


class airbag(vlasovSolver):
    def __init__(self, accelerator, max_l, zhat, *args, **kwargs):
        super().__init__(accelerator, max_l, 0)
        self.zhat = zhat

    def generateBaseMatrix(self, method, zperp, max_harmonics, multi_bunch_mode):
        if method == "perturbed-simple":
            sample_frequencies = self._impedanceSampleFrequencies(max_harmonics, multi_bunch_mode)[0]
            sampled_dipole_impedance = zperp(sample_frequencies)
            self.base_matrix = am.generateSimpleBaseMatrix(
                self.max_l, self.zhat,
                sample_frequencies, sampled_dipole_impedance,
                self.accelerator.f0, self.accelerator.num_bunches,
                self.accelerator.beta,
                self.accelerator.w_b, self.accelerator.w_xi)

        elif method == "perturbed-full":
            sample_frequencies = self._impedanceSampleFrequencies(max_harmonics, multi_bunch_mode)
            sampled_dipole_impedance = zperp(sample_frequencies)
            self.base_matrix = am.generateFullBaseMatrix(
                self.max_l, self.zhat,
                sample_frequencies, sampled_dipole_impedance,
                self.accelerator.f0, self.accelerator.num_bunches,
                self.accelerator.beta,
                self.accelerator.w_b, self.accelerator.w_xi)
        else:
            raise ValueError("Invalid method.")

        return self.base_matrix


class NHT(vlasovSolver):
    def __init__(self, accelerator, max_l, num_rings, *args, **kwards):
        super().__init__(accelerator, max_l, num_rings - 1)

        self.ring_radii = np.zeros([])
        self.num_rings = num_rings

    def calculateRingRadii(self, g0hat, search_limit):
        self.ring_radii = nht.calculateRingRadii(g0hat, self.num_rings,
                                                 search_limit)
        return self.ring_radii

    def generateBaseMatrix(self, method, zperp, max_harmonics, multi_bunch_mode):
        if len(self.ring_radii) != self.num_rings:
            raise Exception("Calculate ring radii first.")
            return self.base_matrix

        if method == "perturbed-simple":
            sample_frequencies = self._impedanceSampleFrequencies(max_harmonics, multi_bunch_mode)[0]
            sampled_dipole_impedance = zperp(sample_frequencies)

            self.base_matrix = nht.generateSimpleBaseMatrix(
                self.max_l, self.ring_radii,
                sample_frequencies, sampled_dipole_impedance,
                self.accelerator.f0, self.accelerator.num_bunches,
                self.accelerator.beta,
                self.accelerator.w_b, self.accelerator.w_xi)
        else:
            raise ValueError("Invalid method.")

        return self.base_matrix


class arbitraryLongitudinal(vlasovSolver):
    def __init__(self, accelerator, max_l, max_n, *args, **kwargs):
        super().__init__(accelerator, max_l, max_n)

        self.a = 0
        self.Gk = np.array([])

    def calculateGk(self, g0hat, a, terms, max_radius=1,
                    numerical_normalise=False, *args, **kwargs):
        Gk = np.zeros(terms, dtype=np.float64)
        for k in range(terms):
            Gk[k] = lm.calculateGk(g0hat, a, k, **kwargs)

        self.a = a
        self.Gk = Gk

        return (a, self.Gk)

    def g0Hat(self, r):
        return lm.g0Hat(r, self.a, self.Gk)

    def generateBaseMatrix(self, method, zperp, max_harmonics, multi_bunch_mode):
        if len(self.Gk) == 0:
            raise Exception("Unperturbed longitudinal distribution has not"
                            + "been fit. \n Run calculateGk() before"
                            + "calculating a base matrix.")
            return self.base_matrix

        if method == "perturbed-simple":
            sample_frequencies = self._impedanceSampleFrequencies(max_harmonics, multi_bunch_mode)[0]
            sampled_dipole_impedance = zperp(sample_frequencies)

            self.base_matrix = lm.generateSimpleBaseMatrix(
                self.max_l, self.max_n, self.a, self.Gk,
                sample_frequencies, sampled_dipole_impedance,
                self.accelerator.f0, self.accelerator.num_bunches,
                self.accelerator.beta,
                self.accelerator.w_b, self.accelerator.w_xi)
        else:
            raise ValueError("Invalid method.")

        return self.base_matrix

    def g1(self, eigenvectors, max_r, grid_size):
        return lm.g1(grid_size, self.a, eigenvectors, max_r,
                     self.max_l, self.max_n,
                     self.accelerator.beta, self.accelerator.chromaticity,
                     self.accelerator.w_b, self.accelerator.eta)


# Helper Functions
def find_most_unstable(eigenvalues):
    '''
    Find the mode unstable mode

    The mode the the largest positive imaginary part is the
    mode with the fastest growth rate. This can be of particular
    interest if one mode dominates since it can then predict
    growth times and the expected distribution.
    '''
    eigenvalue_index = np.argmax(np.imag(eigenvalues))
    return eigenvalue_index


if __name__ == '__main__':

    def transverse_resonator(f: float, fr: float, Rperp: float, Q:float, beta:float) -> complex:
        '''
        Ideal transverse resonator impedance as described by Chao's Equation 2.87, HOWEVER
        Rperp != Rs as defined by Chao. Rperp = v/omega_r * Rs. Rperp has units of Ohm/m
        whilst Rs as defined by Chao has units of Ohm/m^2.
        '''
        w = 2*np.pi*f
        wr = 2 * np.pi*fr
        Rs = Rperp * wr/cn.c/beta
        return cn.c*beta/w * Rs/(1+1j*Q*(wr/w - w/wr))

    def gaus(r, sigma=1):
        return np.atleast_1d(1/sigma**2 * np.exp(-r**2/2/sigma**2))

    def g0hat(r):
        return gaus(r, sigma=23e-2)

    
    ISIS = accelerator(163.3, cn.m_p, cn.e, 0.367, 3.83, 0.0196634, 5.034**(-2), -1.4, 1)
    SPS  = accelerator(1100*2*np.pi, cn.m_p, cn.e, 0.9993481, 20.18, 0.017, 18**(-2), 0, 1)
    test = arbitraryLongitudinal(SPS, 10, 5)

    # Perform fitting on unperturbed longitudinal distribution.
    # Plot results to check.
    test.calculateGk(g0hat, 0.25/(23e-2)**2, 5, method='laguerre-gauss')
    r = np.linspace(0, 1.5, 1000)
    plt.figure()
    plt.plot(r, g0hat(r), 'C0')
    plt.plot(r, test.g0Hat(r), 'C1--')
    plt.show()

    import time

    start = time.time()

    sample_frequencies = test.impedanceSampleFrequencies(25000)
    sampled_impedance = transverse_resonator(sample_frequencies,  # note we're skipping l*omega_s
                                             13e6, 1e7, 1, SPS.beta)
    
    # Generate the base matrix
    test.generateBaseMatrix(sample_frequencies, sampled_impedance)

    # Generate the interaction matrix using the base matrix
    c = test.generateInteractionMatrix(2e13)

    # Investigate results
    eigvals = np.linalg.eigvals(c)

    stop = time.time()
    print(f"Calculation took {stop-start:.2f} s")

    most_unstable_index = np.argmax(np.imag(eigvals))
    print(eigvals[most_unstable_index])
    dist_Omega = eigvals[most_unstable_index]*SPS.w_s + SPS.w_b
    print(f"Growth time = {1/np.imag(dist_Omega)*1e6:.1f} us")
