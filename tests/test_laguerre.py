import unittest
import numpy as np
import scipy.constants as cn
import PyTMCI as vs
from PyTMCI import airbag_methods as am
from PyTMCI import laguerre_methods as lm


# Max and min frequencies to check.
MAXw = 1e13
MINw = 1e-6

# Some quantities are limited due to things like overflow errors.
# The max and min values specified here
LAG_MAXL = 20
LAG_MAXN = 11
LAG_MAXK = LAG_MAXN


class testArbitrary(unittest.TestCase):
    def test_factorial(self):
        # Just checking the first 50 values.
        # The following array was computed with mpmath
        # then converted to an integer [int(mp.factorial(i)) for i in range(50)]
        res = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800,
                        39916800, 479001600, 6227020800, 87178291200, 1307674368000,
                        20922789888000, 355687428096000, 6402373705728000,
                        121645100408832000, 2432902008176640000, 51090942171709440000,
                        1124000727777607680000, 25852016738884978212864,
                        620448401733239409999872, 15511210043330986055303168,
                        403291461126605650322784256, 10888869450418351940239884288,
                        304888344611713871918902804480, 8841761993739701898620088352768,
                        265252859812191068217601719009280,
                        8222838654177922430198509928972288,
                        263130836933693517766352317727113216,
                        8683317618811885938715673895318323200,
                        295232799039604157334081533963162091520,
                        10333147966386145431134989962796349784064,
                        371993326789901254863672752494735387525120,
                        13763753091226345578872114833606270345281536,
                        523022617466601117141859892252474974331207680,
                        20397882081197444123129673397696887153724751872,
                        815915283247897683795548521301193790359984930816,
                        33452526613163807956284472299189486453163567349760,
                        1405006117752879954933135270705268945154855145570304,
                        60415263063373834074440829285578945930237590418489344,
                        2658271574788448869416579949034705352617757694297636864,
                        119622220865480188574992723157469373503186265858579103744,
                        5502622159812089153567237889924947737578015493424280502272,
                        258623241511168177673491006652997026552325199826237836492800,
                        12413915592536072528327568319343857274511609591659416151654400,
                        608281864034267522488601608116731623168777542102418391010639872],
                       dtype=object)
        result = lm.factorial(np.arange(50))
        self.assertIsNone(np.testing.assert_array_almost_equal(res, result))

    def test_rising_fact(self):
        '''
        Test rising factorial by percentage error. (threshold 0.005%)
        '''
        import mpmath as mp

        for x in range(-41, 42):
            for n in np.arange(0, 12):
                res = float(mp.rf(x, n))
                result = lm.rising_fact(x, n)

                # We test with fractional difference because the very large numbers
                # out of the rising factorial make comparing decimal places
                # unnecessary (6 dp on a number with 10^(40) is ~46 SF).
                # But rising_factorial can return 0, and calculating the
                # fractional difference from 0 is impossible. In that case check
                # decimal places.
                msg = f"(x, n) = ({x:.1e}, {n:.1e}), {res:.2e}, {result:.2e}"

                if abs(res) > 0.0:
                    fractional_difference = np.abs((res - result) / res) * 100

                    with self.subTest(msg):
                        self.assertLess(fractional_difference, 0.005)

                else:
                    with self.subTest(msg):
                        self.assertIsNone(np.testing.assert_array_almost_equal(res, result))

    def test_Lna(self):
        import mpmath as mp

        rng = np.random.default_rng(6)

        for a in range(-41, 42):
            for n in range(12):
                x = rng.lognormal(size=301, sigma=8)  # Just set some constants
                sign = np.sign(rng.normal(0, size=len(x)))
                x = sign * x

                res = np.array([float(mp.laguerre(n, a, i)) for i in x])
                result = lm.Lna(n, a, x)

                msg = f"(n, a) = ({n:.1e}, {a:.1e})"
                with self.subTest(msg):
                    self.assertIsNone(np.testing.assert_allclose(result, res, rtol=1e-6, atol=1e-12))

    def test_Ilnk(self):

        # At time of writing Ilnk is not used anymore, and has been replaced by
        # generateGksumDict, because Ilnk only appears in a summation, and by
        # computing the summation in one go then I avoid recomputing the same
        # thing many times. Nevertheless, Ilnk has been retained and will be
        # tested because it allows users to investigate individual terms and
        # see which ones are significant.

        print("")
        print("-----" * 5)
        print("Testing Ilnk. This takes some time (about 10mins on my laptop).")

        import mpmath as mp
        c = cn.c

        rng = np.random.default_rng(7)
        w = rng.lognormal(size=351, sigma=11)  # Just set some constants
        # lognormal can produce some huge and tiny numbers. Limit them to some reasonable range.
        w = w[w < MAXw]
        w = w[w > MINw]
        sign = np.sign(rng.normal(0, size=len(w)))
        w = sign * w

        laguerre = mp.memoize(mp.laguerre)

        for l in range(-LAG_MAXL, LAG_MAXL + 1):
            for n in range(LAG_MAXN + 1):
                for k in range(LAG_MAXK + 1):
                    for beta in [0.1, 0.5, 0.99]:
                        for a in [0.001, 0.01, 0.1, 1, 10]:
                            L1 = np.array([float(laguerre(n, k - n, np.abs(i)**2 / (4 * a * beta**2 * c**2), zeroprec=1000)) for i in w])
                            L2 = np.array([float(laguerre(k, np.abs(l) + n - k, np.abs(i)**2 / (4 * a * beta**2 * c**2), zeroprec=1000)) for i in w])

                            res = ((np.sign(w) * np.sign(l))**np.abs(l)
                                   * (-1)**(n + k)
                                   * (np.abs(w) / (2 * a * beta * c))**np.abs(l)
                                   * np.exp(-np.abs(w)**2 / (4 * a * beta**2 * c**2))
                                   * L1
                                   * L2)

                            result = lm.Ilnk(w, a, l, n, k, beta)

                            msg = f"a: {a:.2e}, l: {l}, n: {n}, k: {k}, beta: {beta:.2e}"
                            with self.subTest(msg):
                                self.assertIsNone(np.testing.assert_allclose(result, res, rtol=1e-6, atol=1e-12))

    def test_generateGksumDict(self):
        '''
        Test generateGksumDict using the already-tested Ilnk method.
        '''
        rng = np.random.default_rng(7)
        w = rng.lognormal(size=351, sigma=11)  # Just set some constants
        # lognormal can produce some huge and tiny numbers. Limit them to some reasonable range.
        w = w[w < MAXw]
        w = w[w > MINw]
        sign = np.sign(rng.normal(0, size=len(w)))
        w = sign * w

        Gk_array = rng.random(size=LAG_MAXK + 1)

        for beta in [0.1, 0.5, 0.99]:
            for a in [0.001, 0.01, 0.1, 1, 10]:

                result = lm.generateGksumDict(Gk_array, w, a, LAG_MAXL, LAG_MAXN, beta)

                for l in range(-LAG_MAXL, LAG_MAXL + 1):
                    for n in range(LAG_MAXN + 1):

                        total = 0
                        for k, Gk in enumerate(Gk_array):
                            total += Gk * lm.Ilnk(w, a, l, n, k, beta)

                        res = total

                        msg = f"a: {a:.2e}, l: {l}, n: {n}, beta: {beta:.2e}"
                        with self.subTest(msg):
                            self.assertIsNone(np.testing.assert_allclose(result[l][n], res, rtol=1e-6, atol=1e-12))

    def test_Qln(self):
        import mpmath as mp
        c = cn.c

        rng = np.random.default_rng(7)
        w = rng.lognormal(size=351, sigma=11)  # Just set some constants
        # lognormal can produce some huge and tiny numbers. Limit them to some reasonable range.
        w = w[w < MAXw]
        w = w[w > MINw]
        sign = np.sign(rng.normal(0, size=len(w)))
        w = sign * w

        for l in range(-LAG_MAXL, LAG_MAXL + 1):
            for n in range(LAG_MAXN + 1):
                for beta in [0.1, 0.5, 0.99]:
                    for a in [0.001, 0.01, 0.1, 1, 10]:
                        absl = np.abs(l)
                        res = (a**(absl / 2 - 1) * np.sign(l)**(absl)
                               * 1 / (2 * np.math.factorial(n))
                               * (w / (2 * np.sqrt(a) * beta * c))**(2 * n + absl)
                               * np.exp(-w**2 / 4 / a / beta**2 / c**2))

                        result = lm.Qln(w, a, l, n, beta)
                        msg = f"a: {a:.2e}, l: {l}, n: {n}, beta: {beta:.2e}"
                        with self.subTest(msg):
                            self.assertIsNone(np.testing.assert_allclose(result, res, rtol=1e-6, atol=1e-12))

    def test_generateFactCoefficient(self):
        import scipy.special as sp

        # This is a very simple calculation but does involve factorials, so
        # there's always a risk of overflow errors. To ensure I avoid these
        # in the test I'll use the gammaln
        result = lm.generateFactCoefficient(LAG_MAXL, LAG_MAXN)

        # only depends on |l| so just check the positive values
        for l in range(0, LAG_MAXL + 1):
            for n in range(0, LAG_MAXN + 1):
                res = np.exp(sp.gammaln(n + 1) - sp.gammaln(np.abs(l) + n + 1))

                msg = f"l: {l}, n: {n}"
                with self.subTest(msg):
                    self.assertAlmostEqual(result[l][n], res, places=5)

    def test_generateLMatrix(self):
        L = []
        L.append(np.array([
            [-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, +0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, +1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, +2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, +3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +3]
        ], dtype=np.complex128)
        )

        L.append(np.array([
            [-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, +0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, +1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, +2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, +0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, +1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, +2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +2],
        ], dtype=np.complex128)
        )

        for i, (max_l, max_n) in enumerate(zip([3, 2],
                                               [1, 2])):

            result = lm.generateLMatrix(max_l, max_n)

            msg = f"max_l = {max_l}, max_n = {max_n}"
            with self.subTest(msg):
                self.assertIsNone(np.testing.assert_array_equal(result, L[i]))

    def test_generateSimpleBaseMatrix(self):
        rng = np.random.default_rng(5)

        for num_bunches in range(3):
            for multi_bunch_mode in range(num_bunches):
                for max_l in [5, 10, 15, 20]:
                    for max_n in [3, 5, 10]:
                        for a in [0.01, 0.1, 1.0, 10]:

                            Gk_array = rng.lognormal(sigma=3, size=rng.integers(low=2, high=8))

                            circumference = 143.456
                            mass = 3.5e-30
                            charge = 2.6e-19
                            beta = 0.2
                            Qb = 2.9
                            Qs = 0.85
                            alpha = 0.12
                            chromaticity = -0.8
                            acc1 = vs.accelerator(circumference,
                                                  mass,
                                                  charge,
                                                  beta,
                                                  Qb,
                                                  Qs,
                                                  alpha,
                                                  chromaticity,
                                                  num_bunches)
                            max_harmonics = 100

                            # Calculate slippage factor.
                            f0 = beta * cn.c / circumference
                            w_b = 2 * np.pi * Qb * f0
                            gamma = 1 / np.sqrt(1 - beta**2)
                            eta = alpha - 1 / gamma**2  # Chao Eq. 1.10, pp. 9
                            w_xi = chromaticity * w_b / eta  # Chao Eq. 6.181, pp. 338

                            # Verify that accelerator parameters have been worked out correctly
                            self.assertAlmostEqual(f0, acc1.f0, places=6)
                            self.assertAlmostEqual(w_b, acc1.w_b, places=6)
                            self.assertAlmostEqual(gamma, acc1.gamma, places=6)
                            self.assertAlmostEqual(eta, acc1.eta, places=6)
                            self.assertAlmostEqual(w_xi, acc1.w_xi, places=6)

                            p = np.arange(-max_harmonics, max_harmonics + 1, dtype=np.float64)

                            # Often impedance is given as a function of f=w/2/pi, rather than w so
                            # find this first
                            fp = ((p * num_bunches + multi_bunch_mode) * f0
                                  + Qb * f0
                                  + 0 * Qs * f0)
                            wp = 2 * np.pi * fp
                            w = wp - w_xi

                            # for perturbed-simple all Ilnk and Qln's are sampled at the same frequency.
                            Ilnksums = {}
                            for l in range(-max_l, max_l + 1):
                                temp = {}
                                for n in range(max_n + 1):
                                    temp2 = 0
                                    for k, Gk in enumerate(Gk_array):
                                        temp2 += Gk * lm.Ilnk(w, a, l, n, k, beta)

                                    temp[n] = temp2

                                Ilnksums[l] = temp

                            Qln = {}
                            for l in range(-max_l, max_l + 1):
                                temp = {}
                                for n in range(max_n + 1):
                                    temp[n] = lm.Qln(w, a, l, n, beta)

                                Qln[l] = temp

                            gamma_coef = lm.generateFactCoefficient(max_l, max_n)

                            # Impedance
                            def zperp(f):
                                return f + 1j / f

                            sampled_impedance = zperp(fp)

                            # Allocate memory for the matrix
                            res = np.zeros(shape=((2 * max_l + 1) * (max_n + 1),
                                                  (2 * max_l + 1) * (max_n + 1)),
                                           dtype=np.complex128)

                            for nn in range(max_n + 1):
                                for nnp in range(max_n + 1):
                                    for ll, l in enumerate(range(-max_l, max_l + 1)):
                                        temp = sampled_impedance * Ilnksums[l][nn]
                                        for llp, lp in enumerate(range(-max_l, max_l + 1)):
                                            res[nn * (2 * max_l + 1) + ll, nnp * (2 * max_l + 1) + llp] = 1j**(l - lp) * gamma_coef[abs(l)][nn] * np.sum(temp * Qln[lp][nnp])

                            test = vs.arbitraryLongitudinal(acc1, max_l, max_n)
                            test.Gk = Gk_array
                            test.a = a
                            result = test.generateBaseMatrix('perturbed-simple', zperp, max_harmonics, multi_bunch_mode)

                            argmax = np.argmax(np.abs(res - result))
                            msg = f"x = {res.flatten()[argmax]:.5e}, y={result.flatten()[argmax]:.5e}"
                            with self.subTest(msg):
                                self.assertIsNone(np.testing.assert_allclose(res, result, rtol=1e-6, atol=1e-12))


    def test_generateFullBaseMatrix(self):
        rng = np.random.default_rng(5)

        for num_bunches in range(3):
            for multi_bunch_mode in range(num_bunches):
                for max_l in [5, 10, 15, 20]:
                    for max_n in [3, 5, 10]:
                        for a in [0.01, 0.1, 1.0, 10]:

                            Gk_array = rng.lognormal(sigma=3, size=rng.integers(low=2, high=8))

                            circumference = 143.456
                            mass = 3.5e-30
                            charge = 2.6e-19
                            beta = 0.2
                            Qb = 2.9
                            Qs = 0.85
                            alpha = 0.12
                            chromaticity = -0.8
                            acc1 = vs.accelerator(circumference,
                                                  mass,
                                                  charge,
                                                  beta,
                                                  Qb,
                                                  Qs,
                                                  alpha,
                                                  chromaticity,
                                                  num_bunches)
                            max_harmonics = 100

                            # Calculate slippage factor.
                            f0 = beta * cn.c / circumference
                            w_b = 2 * np.pi * Qb * f0
                            gamma = 1 / np.sqrt(1 - beta**2)
                            eta = alpha - 1 / gamma**2  # Chao Eq. 1.10, pp. 9
                            w_xi = chromaticity * w_b / eta  # Chao Eq. 6.181, pp. 338

                            # Verify that accelerator parameters have been worked out correctly
                            self.assertAlmostEqual(f0, acc1.f0, places=6)
                            self.assertAlmostEqual(w_b, acc1.w_b, places=6)
                            self.assertAlmostEqual(gamma, acc1.gamma, places=6)
                            self.assertAlmostEqual(eta, acc1.eta, places=6)
                            self.assertAlmostEqual(w_xi, acc1.w_xi, places=6)

                            p = np.arange(-max_harmonics, max_harmonics + 1, dtype=np.float64)
                            orders = list(range(0, max_l + 1)) + list(range(-max_l, 0))
                            # Often impedance is given as a function of f=w/2/pi, rather than w so
                            # find this first
                            fp = np.array([((p * num_bunches + multi_bunch_mode) * f0
                                            + Qb * f0
                                            + l * Qs * f0) for l in orders])
                            wp = 2 * np.pi * fp
                            w = wp - w_xi

                            # for perturbed-simple all Ilnk and Qln's are sampled at the same frequency.
                            Ilnksums = {}
                            for l in range(-max_l, max_l + 1):
                                temp = {}
                                for n in range(max_n + 1):
                                    temp2 = 0
                                    for k, Gk in enumerate(Gk_array):
                                        temp2 += Gk * lm.Ilnk(w[l], a, l, n, k, beta)

                                    temp[n] = temp2

                                Ilnksums[l] = temp

                            Qlns_l = []
                            for i in w:
                                Qln = {}
                                for lp in range(-max_l, max_l + 1):
                                    temp = {}
                                    for nnp in range(max_n + 1):
                                        temp[nnp] = lm.Qln(i, a, lp, nnp, beta)

                                    Qln[lp] = temp

                                Qlns_l.append(Qln)

                            gamma_coef = lm.generateFactCoefficient(max_l, max_n)

                            # Impedance
                            def zperp(f):
                                return f + 1j * f * 3

                            sampled_impedance = zperp(fp)

                            # Allocate memory for the matrix
                            res = np.zeros(shape=((2 * max_l + 1) * (max_n + 1),
                                                  (2 * max_l + 1) * (max_n + 1)),
                                           dtype=np.complex128)

                            for nn in range(max_n + 1):
                                for nnp in range(max_n + 1):
                                    for ll, l in enumerate(range(-max_l, max_l + 1)):
                                        temp = sampled_impedance[l] * Ilnksums[l][nn]
                                        for llp, lp in enumerate(range(-max_l, max_l + 1)):
                                            res[nn * (2 * max_l + 1) + ll, nnp * (2 * max_l + 1) + llp] = 1j**(l - lp) * gamma_coef[abs(l)][nn] * np.sum(temp * Qlns_l[l][lp][nnp])

                            test = vs.arbitraryLongitudinal(acc1, max_l, max_n)
                            test.Gk = Gk_array
                            test.a = a
                            result = test.generateBaseMatrix('perturbed-full', zperp, max_harmonics, multi_bunch_mode)

                            argmax = np.argmax(np.abs(res - result))
                            msg = f"x = {res.flatten()[argmax]:.5e}, y={result.flatten()[argmax]:.5e}"
                            with self.subTest(msg):
                                self.assertIsNone(np.testing.assert_allclose(res, result, rtol=1e-6, atol=1e-12))


if __name__ == '__main__':
    unittest.main()
