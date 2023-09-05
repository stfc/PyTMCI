import unittest
import numpy as np
import scipy.constants as cn
import PyTMCI as vs
from PyTMCI import airbag_methods as am


class TestNHT(unittest.TestCase):
    def test_generateSimpleBaseMatrix(self):
        import scipy.special as sp

        for num_bunches in range(1, 5):
            for multi_bunch_mode in range(num_bunches):
                for num_rings in [2, 3, 5, 10]:
                    rng = np.random.default_rng(seed=num_rings)

                    circumference = 523.456
                    mass = 1.5e-20
                    charge = 1.6e-19
                    beta = 0.5
                    Qb = 3.9
                    Qs = 0.34
                    alpha = 0.22
                    chromaticity = -1.5

                    acc1 = vs.accelerator(circumference,
                                          mass,
                                          charge,
                                          beta,
                                          Qb,
                                          Qs,
                                          alpha,
                                          chromaticity,
                                          num_bunches)
                    max_l = 3
                    max_n = num_rings - 1
                    ring_radii = rng.lognormal(size=num_rings, sigma=3)  # Just set some constants
                    ring_radii.sort()

                    # Calculate impedance sample frequencies
                    max_harmonics = 5

                    # Calculate slippage factor.
                    f0 = beta * cn.c / circumference
                    w_b = 2 * np.pi * Qb * f0
                    gamma = 1 / np.sqrt(1 - beta**2)
                    eta = alpha - 1 / gamma**2  # Chao Eq. 1.10, pp. 9
                    w_xi = chromaticity * w_b / eta  # Chao Eq. 6.181, pp. 338

                    # Verify that accelerator parameters have been worked out correctly
                    self.assertEqual(f0, acc1.f0)
                    self.assertEqual(w_b, acc1.w_b)
                    self.assertEqual(gamma, acc1.gamma)
                    self.assertEqual(eta, acc1.eta)
                    self.assertEqual(w_xi, acc1.w_xi)

                    p = np.arange(-max_harmonics, max_harmonics + 1, dtype=np.float64)
                    # Often impedance is given as a function of f=w/2/pi, rather than w so
                    # find this first
                    fp = ((p * num_bunches + multi_bunch_mode) * f0 + Qb * f0)
                    wp = 2 * np.pi * fp
                    w = wp - w_xi

                    # for perturbed-simple w is a 1D array. All bessel functions and
                    # impedances are sampled at the same frequencies.
                    orders = list(range(0, max_l + 1)) + list(range(-max_l, 0))
                    jdict = [[sp.jn(i, w * zhat / beta / cn.c) for i in orders] for zhat in ring_radii]

                    def zperp(f):
                        return np.cos(f) + 1j * np.sin(f)

                    sampled_impedance = zperp(fp)

                    # Now create the base matrix
                    submatrix_size = 2 * max_l + 1
                    my_base_matrix = np.zeros((submatrix_size * num_rings,
                                               submatrix_size * num_rings), dtype=np.complex128)

                    for row in range((2 * max_l + 1) * num_rings):
                        l = row % submatrix_size - max_l
                        n = row // submatrix_size

                        temp = jdict[n][l] * sampled_impedance

                        for col in range((2 * max_l + 1) * num_rings):
                            lpp = col % submatrix_size - max_l
                            npp = col // submatrix_size

                            my_base_matrix[row, col] = 1 / num_rings * (1j)**(l - lpp) * np.sum(temp * jdict[npp][lpp])

                    test = vs.NHT(acc1, max_l, num_rings)
                    test.ring_radii = ring_radii
                    base_matrix = test.generateBaseMatrix('perturbed-simple', zperp, max_harmonics, multi_bunch_mode)

                    with self.subTest():
                        self.assertIsNone(np.testing.assert_array_almost_equal(base_matrix, my_base_matrix))

    def test_generateFullBaseMatrix(self):
        import scipy.special as sp

        for num_bunches in range(1, 5):
            for multi_bunch_mode in range(num_bunches):
                for num_rings in [2, 3, 5, 10]:
                    for max_harmonics in [10, 11, 100]:
                        rng = np.random.default_rng(seed=num_rings)

                        circumference = 523.456
                        mass = 1.5e-20
                        charge = 1.6e-19
                        beta = 0.5
                        Qb = 3.9
                        Qs = 0.34
                        alpha = 0.22
                        chromaticity = -1.5

                        acc1 = vs.accelerator(circumference,
                                              mass,
                                              charge,
                                              beta,
                                              Qb,
                                              Qs,
                                              alpha,
                                              chromaticity,
                                              num_bunches)
                        max_l = 3
                        max_n = num_rings - 1
                        ring_radii = rng.lognormal(size=num_rings, sigma=3)  # Just set some constants
                        ring_radii.sort()

                        # Calculate impedance sample frequencies
                        f0 = beta * cn.c / circumference
                        w_b = 2 * np.pi * Qb * f0
                        w_s = 2 * np.pi * Qs * f0
                        gamma = 1 / np.sqrt(1 - beta**2)
                        eta = alpha - 1 / gamma**2  # Chao Eq. 1.10, pp. 9
                        w_xi = chromaticity * w_b / eta  # Chao Eq. 6.181, pp. 338

                        # Verify that accelerator parameters have been worked out correctly
                        self.assertEqual(f0, acc1.f0)
                        self.assertEqual(w_b, acc1.w_b)
                        self.assertEqual(gamma, acc1.gamma)
                        self.assertEqual(eta, acc1.eta)
                        self.assertEqual(w_xi, acc1.w_xi)

                        p = np.arange(-max_harmonics, max_harmonics + 1, dtype=np.float64)
                        orders = list(range(0, max_l + 1)) + list(range(-max_l, 0))
                        # Often impedance is given as a function of f=w/2/pi, rather than w so
                        # find this first
                        fp = np.array([((p * num_bunches + multi_bunch_mode) * f0
                                       + Qb * f0
                                       + (l * w_s) / 2 / np.pi) for l in orders])
                        wp = 2 * np.pi * fp
                        w = wp - w_xi

                        # for perturbed-simple w is a 1D array. All bessel functions and
                        # impedances are sampled at the same frequencies.
                        jdict = [[[sp.jn(i, wl * zhat / beta / cn.c) for i in orders] for zhat in ring_radii] for wl in w]

                        def zperp(f):
                            return np.cos(f) + 1j * np.sin(f)

                        sampled_impedance = [zperp(fl) for fl in fp]

                        # Now create the base matrix
                        submatrix_size = 2 * max_l + 1
                        my_base_matrix = np.zeros((submatrix_size * num_rings,
                                                   submatrix_size * num_rings), dtype=np.complex128)

                        for row in range((2 * max_l + 1) * num_rings):
                            l = row % submatrix_size - max_l
                            n = row // submatrix_size

                            temp = jdict[l][n][l] * sampled_impedance[l]

                            for col in range((2 * max_l + 1) * num_rings):
                                lpp = col % submatrix_size - max_l
                                npp = col // submatrix_size

                                my_base_matrix[row, col] = 1 / num_rings * (1j)**(l - lpp) * np.sum(temp * jdict[l][npp][lpp])

                        test = vs.NHT(acc1, max_l, num_rings)
                        test.ring_radii = ring_radii
                        base_matrix = test.generateBaseMatrix('perturbed-full', zperp, max_harmonics, multi_bunch_mode)

                        with self.subTest():
                            self.assertIsNone(np.testing.assert_array_almost_equal(base_matrix, my_base_matrix))