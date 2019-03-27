import unittest
import numpy as np
from util import rotate, embed_noise, CoordinateSystem


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

        system = CoordinateSystem(a)
        res = system.map_to_coordinates()
        expected = np.array([[0, 1, 2, 0, 1, 2],
                             [1, 1, 1, 0, 0, 0]])

        self.assertTupleEqual(res.shape, (2, 6))
        self.assertTrue(np.all(expected == res))

    def test_mapping_with_non_standard_origin(self):
        a = np.array([[1, 0, 6],
                      [15, 0, 2]])

        system = CoordinateSystem(a, x0=2, y0=1)

        res = system.map_to_coordinates()
        expected = np.array([[-2, -1, 0, -2, -1, 0],
                             [0, 0, 0, -1, -1, -1]])

        self.assertTupleEqual(res.shape, (2, 6))
        self.assertTrue(np.all(expected == res))

    def test_coordinates_to_cell(self):
        a = np.array([[1, 0, 6],
                      [15, 0, 2]])
        system = CoordinateSystem(a, x0=2, y0=1)

        row, col = system.coordinates_to_cell(-2, 0)
        self.assertEqual(row, 0)
        self.assertEqual(col, 0)

        row, col = system.coordinates_to_cell(-1, 0)
        self.assertEqual(row, 0)
        self.assertEqual(col, 1)

        row, col = system.coordinates_to_cell(-1, -1)
        self.assertEqual(row, 1)
        self.assertEqual(col, 1)

    def test_with_90_degrees(self):
        a = np.array([[1, 0],
                      [15, 25]])

        res = rotate(a, angle=90)

        self.assertEqual([[25, 0], [15, 0]], res.tolist())

    def test_with_without_rotation(self):
        a = np.array([[1, 0],
                      [15, 25]])

        res = rotate(a, angle=0)
        self.assertEqual(a.tolist(), res.tolist())

        res = rotate(a, angle=360)
        self.assertEqual(a.tolist(), res.tolist())

    def test_rotate_around_the_center(self):
        a = np.arange(9).reshape(3, 3)

        res = rotate(a, angle=90, origin=(1, 1))

        expected = np.array([[2, 5, 8],
                             [1, 4, 7],
                             [0, 3, 6]])
        self.assertEqual(expected.tolist(), res.tolist())

        res = rotate(a, angle=180, origin=(1, 1))

        expected = np.array([[8, 7, 6],
                             [5, 4, 3],
                             [2, 1, 0]])
        self.assertEqual(expected.tolist(), res.tolist())


class BatchGeneratorTests(unittest.TestCase):
    def test_on_dummy_files(self):
        from util import training_batches

        extended_x_path = 'X_dummy.txt'
        extended_y_path = 'Y_dummy.txt'
        batches = training_batches(extended_x_path, extended_y_path, batch_size=2)

        b = []
        for batch in batches:
            b.append(batch)

        self.assertEqual(len(b), 2)

        X, Y = b[0]
        self.assertTupleEqual(X.shape, (5, 2))
        self.assertTupleEqual(Y.shape, (1, 2))

        self.assertEqual(X.tolist(), [[5, 1],
                                      [248, 0],
                                      [0, 0],
                                      [0, 123],
                                      [15, 45]])

        self.assertEqual(Y.tolist(), [[8, 4]])

        X, Y = b[1]
        self.assertTupleEqual(X.shape, (5, 1))
        self.assertTupleEqual(Y.shape, (1, 1))


if __name__ == '__main__':
    unittest.main()
