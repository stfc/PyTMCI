import unittest
import numpy as np
import scipy.constants as cn
import PyTMCI as vs
from PyTMCI import airbag_methods as am
from PyTMCI import laguerre_methods as lm


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