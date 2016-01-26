import numpy as np
import os
import unittest

from R1DL import r1dl

class TestR1D1(unittest.TestCase):

    def test_dataSet1(self):
        self._runTest("../testSet1")

    def test_dataSet2(self):
        self._runTest("../testSet2")

    def test_largeData(self):
        pass

    def _runTest(self, path, R = 0.2, M = 5, epsilon = 0.01, sigfigs = 6):
        S = np.loadtxt(os.path.join(path, "S.txt"))
        Ztrue = np.loadtxt(os.path.join(path, "z_groundtruth.txt"))
        Dtrue = np.loadtxt(os.path.join(path, "D_groundtruth.txt"))

        # Run the test.
        D, Z = r1dl(S, R, M, epsilon)


        np.savetxt("/Users/squinn/Desktop/Z.txt", Z)
        np.savetxt("/Users/squinn/Desktop/D.txt", D)

        # Compare the results.
        np.testing.assert_array_almost_equal(Z, Ztrue, decimal = sigfigs)
        np.testing.assert_array_almost_equal(D, Dtrue, decimal = sigfigs)


if __name__ == "__main__":
    unittest.main()
