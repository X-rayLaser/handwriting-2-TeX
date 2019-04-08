import unittest
from util import connect_chains, ChainOfPoints, chains_sorted_by_distance


class ChainOfPointsTests(unittest.TestCase):
    def test_distance_is_zero(self):
        points1 = [(0, 1), (2, 4)]
        points2 = [(2, 4)]
        chain1 = ChainOfPoints(points1)
        chain2 = ChainOfPoints(points2)
        self.assertEqual(chain1.distance_to(chain1), 0)
        self.assertEqual(chain1.distance_to(chain2), 0)

    def test_distance_to_neighbor_chain(self):
        points1 = [(0, 1), (2, 4)]
        points2 = [(1, 1), (20, 30)]
        chain1 = ChainOfPoints(points1)
        chain2 = ChainOfPoints(points2)
        self.assertEqual(chain1.distance_to(chain2), 1)

    def test_distance_between_chains_far_apart(self):
        points1 = [(0, 1), (2, 4)]
        points2 = [(12, 14), (22, 24)]

        chain1 = ChainOfPoints(points1)
        chain2 = ChainOfPoints(points2)
        self.assertAlmostEqual(chain1.distance_to(chain2), 200**0.5)


class SortChainsByDistanceTests(unittest.TestCase):
    def test_on_3_chains(self):
        chain1 = ChainOfPoints([(1, 1), (2, 2), (4, 4)])
        chain2 = ChainOfPoints([(11, 12)])
        chain3 = ChainOfPoints([(5, 5), (8, 9)])
        res = chains_sorted_by_distance((0, 0), [chain1, chain2, chain3])

        self.assertEqual(res[0], chain1)
        self.assertEqual(res[1], chain3)
        self.assertEqual(res[2], chain2)


class ConnectChainsTests(unittest.TestCase):
    def test_on_single_points_nearby(self):
        pset1 = ChainOfPoints([(0, 0)])
        pset2 = ChainOfPoints([(0, 1)])
        pset3 = ChainOfPoints([(1, 1)])

        res = connect_chains([pset1, pset2, pset3])
        self.assertEqual(len(res), 1)
        points = [(p.x, p.y) for p in res[0].points]
        self.assertEqual(set(points), set([(0, 0), (0, 1), (1, 1)]))

    def test_on_real_life_chains_configuration(self):
        chain1 = ChainOfPoints([(0, 1), (1, 1)])
        chain2 = ChainOfPoints([(10, 11), (11, 11), (11, 12)])

        chain3 = ChainOfPoints([(245, 246), (248, 247)])
        chain4 = ChainOfPoints([(252, 252), (253, 252)])

        chain5 = ChainOfPoints([(400, 20), (401, 21), (402, 21)])

        res = connect_chains([chain1, chain2, chain3, chain4, chain5])
        self.assertEqual(len(res), 3)

        self.assertEqual(set([(p.x, p.y) for p in res[0].points]),
                         set([(400, 20), (401, 21), (402, 21)]))

        self.assertEqual(set([(p.x, p.y) for p in res[1].points]),
                         set([(0, 1), (1, 1), (10, 11), (11, 11), (11, 12)]))

        self.assertEqual(set([(p.x, p.y) for p in res[2].points]),
                         set([(245, 246), (248, 247), (252, 252), (253, 252)]))


if __name__ == '__main__':
    unittest.main()
