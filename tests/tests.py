import unittest
import numpy as np
from util import rotate, embed_noise, map_to_coordinates


class NoiseTests(unittest.TestCase):

    def test_remains_within_valid_range(self):
        np.random.seed(1002)
        a = np.array([[225, 255, 250, 49]])
        res = embed_noise(a)

        self.assertTrue(np.all(res <= 255))

        a = np.array([[1, 2, 0, 0]])
        res = embed_noise(a)
        self.assertTrue(np.all(res >= 0))

    def test_with_2d_array(self):
        np.random.seed(3)

        a = np.array([[240, 0, 230]])

        res = embed_noise(a)
        self.assertEqual([[255, 21, 234]], res.tolist())


class RotationTests(unittest.TestCase):
    def test_mapping_to_coordinates(self):
        a = np.array([[1, 0, 6],
                      [15, 0, 2]])
        res = map_to_coordinates(a)
        expected = np.array([[0, 1, 2, 0, 1, 2],
                             [1, 1, 1, 0, 0, 0]])

        self.assertTupleEqual(res.shape, (2, 6))
        self.assertTrue(np.all(expected == res))

    def test_with_90_degrees(self):
        a = np.array([[1, 0],
                      [15, 25]])

        res = rotate(a, angle=90)

        self.assertEqual([[25, 0], [15, 0]], res.tolist())


if __name__ == '__main__':
    unittest.main()
