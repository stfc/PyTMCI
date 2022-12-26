import unittest
import numpy as np
import scipy.constants as cn
import PyVlasovSolver as vs
from PyVlasovSolver import airbag_methods as am


class TestGeneral(unittest.TestCase):
    def test_impedanceSampleFrequencies_l00(self):
        """
        Test the impedance sample frequencies.
        """
        for max_harmonics in [5, 6]:
            for beta in [0.01, 0.1, 0.5, 0.9, 0.999]:
                for Qb in [0.1, 1.2, 3.9, 100]:
                    for Qs in [0.1, 0.4, 2.1]:
                        for num_bunches in range(1, 5):
                            for multi_bunch_mode in range(num_bunches):
                                circumference = 123.456
                                mass = 1.5e-30
                                charge = 1.6e-19
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
                                zhat = 5.0

                                test = vs.airbag(acc1, max_l, zhat)
                                result = test._impedanceSampleFrequencies(max_harmonics, multi_bunch_mode)

                                p = np.arange(-max_harmonics, max_harmonics + 1, dtype=np.float64)
                                f0 = cn.c * beta / circumference

                                orders = list(range(0, max_l + 1)) + list(range(-max_l, 0))
                                frequencies = []
                                for ll in orders:
                                    frequencies.append(((p * num_bunches + multi_bunch_mode) * f0
                                                        + Qb * f0
                                                        + ll * Qs * f0))

                                msg = (f"max_harmonics = {max_harmonics} "
                                       + f"beta = {beta} "
                                       + f"Qb = {Qb} "
                                       + f"Qs = {Qs} "
                                       + f"num_bunches = {num_bunches} "
                                       + f"multi_bunch_mode={multi_bunch_mode}")
                                with self.subTest(msg):
                                    self.assertEqual(np.shape(result), np.shape(frequencies))
                                    self.assertIsNone(np.testing.assert_array_almost_equal(result, frequencies))


class TestAirbag(unittest.TestCase):
    def test_BesselJDict1D(self):
        import scipy.special as sp

        for max_order in [5, 6, 100]:
            x = np.array([0.1, 1.0, 100.0, 1e-5, 1e9, -5.0])
            result = am.generateBesselJDict1D(max_order, x)

            # The resulting array should be indexed by the order, so these
            # won't go -5, -4, ..., 0, 1, 2 etc. The point is then that
            # res[n] = sp.jn(n, x)
            orders = list(range(0, max_order + 1)) + list(range(-max_order, 0))
            res = np.array([sp.jn(i, x) for i in orders])

            msg = f"max_order = {max_order}"
            with self.subTest(msg):
                self.assertIsNone(np.testing.assert_array_almost_equal(result, res))

    def test_BesselJDict2D(self):
        '''
        In this case x is a 2D matrix. This is an array of frequencies for different
        values of l.
        '''
        import scipy.special as sp

        for rng in [np.random.default_rng(seed=i) for i in [42, 43, 55]]:
            for num_harmonics in [5, 10, 1000]:
                for max_order in [2, 5, 20]:
                    # These are the frequencies for different values of l. They are indexed by l
                    # so the first row are the frequencies for 0*ω_s, the second row are the
                    # frequencies for 1*ω_s etc. The size of this array sets max_l, because the
                    # number of rows is 2*max_l + 1. lognormal used to get a large spread of numbers.
                    w = rng.lognormal(0, sigma=20, size=(2 * max_order + 1, num_harmonics))

                    # lognormal only makes positive numbers, also randomize the signs
                    sign = np.sign(rng.normal(0, size=(2 * max_order + 1, num_harmonics)))
                    w = sign * w

                    orders = list(range(0, max_order + 1)) + list(range(-max_order, 0))
                    res = [np.array([sp.jn(lp, w[i]) for lp in orders]) for i in orders]

                    result = am.generateBesselJDict2D(max_order, w)

                    msg = f"max_order = {max_order}"
                    with self.subTest(msg):
                        self.assertIsNone(np.testing.assert_array_almost_equal(result, res))

    def test_generateSimpleBaseMatrix(self):
        import scipy.special as sp

        for num_bunches in range(1, 5):
            for multi_bunch_mode in range(num_bunches):
                for max_l in range(3, 10):
                    circumference = 123.456
                    mass = 1.5e-30
                    charge = 1.6e-19
                    beta = 0.5
                    Qb = 3.9
                    Qs = 0.45
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
                    zhat = 5.0
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
                    fp = ((p * num_bunches + multi_bunch_mode) * f0
                          + Qb * f0
                          + 0 * Qs * f0)
                    wp = 2 * np.pi * fp
                    w = wp - w_xi

                    # for perturbed-simple w is a 1D array. All bessel functions and
                    # impedances are sampled at the same frequencies.
                    bessel_argument = w * zhat / beta / cn.c

                    orders = list(range(0, max_l + 1)) + list(range(-max_l, 0))
                    res = [sp.jn(i, bessel_argument) for i in orders]

                    def zperp(f):
                        return np.cos(f) + 1j * np.sin(f)

                    my_base_matrix = [
                        np.array([1j**(l - lp) * np.sum(res[lp] * res[l] * zperp(fp)) for lp in range(-max_l, max_l + 1)])
                        for l in range(-max_l, max_l + 1)
                    ]

                    test = vs.airbag(acc1, max_l, zhat)
                    base_matrix = test.generateBaseMatrix('perturbed-simple', zperp, max_harmonics, multi_bunch_mode)

                    msg = f"num_bunches = {num_bunches} "
                    with self.subTest(msg):
                        self.assertIsNone(np.testing.assert_array_almost_equal(base_matrix, my_base_matrix))

    def test_generateFullBaseMatrix(self):
        import scipy.special as sp

        for num_bunches in range(1, 5):
            for multi_bunch_mode in range(num_bunches):
                for Qs in [0, 0.45, 1.2]:
                    circumference = 523.456
                    mass = 1.5e-20
                    charge = 1.6e-19
                    beta = 0.5
                    Qb = 3.9
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
                    zhat = 5.0
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
                    orders = list(range(0, max_l + 1)) + list(range(-max_l, 0))

                    # Often impedance is given as a function of f=w/2/pi, rather than w so
                    # find this first
                    fp = np.array([((p * num_bunches + multi_bunch_mode) * f0
                                    + Qb * f0
                                    + i * Qs * f0) for i in orders])
                    wp = 2 * np.pi * fp
                    w = wp - w_xi

                    # for perturbed-simple w is a 1D array. All bessel functions and
                    # impedances are sampled at the same frequencies.
                    bessel_argument = [i * zhat / beta / cn.c for i in w]

                    res = [
                        [sp.jn(lp, bessel_argument[l]) for lp in orders]
                        for l in orders
                    ]

                    def zperp(f):
                        return np.cos(f) + 1j * np.sin(f)

                    my_base_matrix = np.array([
                        np.array([1j**(l - lp) * np.sum(res[l][lp] * res[l][l] * zperp(fp[l])) for lp in range(-max_l, max_l + 1)])
                        for l in range(-max_l, max_l + 1)
                    ])

                    test = vs.airbag(acc1, max_l, zhat)
                    base_matrix = test.generateBaseMatrix('perturbed-full', zperp, max_harmonics, multi_bunch_mode)

                    msg = f"num_bunches = {num_bunches} "
                    with self.subTest(msg):
                        self.assertIsNone(np.testing.assert_array_almost_equal(base_matrix, my_base_matrix))


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


if __name__ == '__main__':
    unittest.main()
