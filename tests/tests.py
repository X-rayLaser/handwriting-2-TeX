import unittest
from util import connect_chains, ChainOfPoints, chains_sorted_by_distance
from building_blocks import Digit
from construction import construct_latex


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


class SubregionAboveTests(unittest.TestCase):
    def test_subregion_above_normal_case(self):
        from construction import RectangularRegion

        region = RectangularRegion(x=20, y=30, width=50, height=40)

        subregion = region.subregion_above(y=50)

        self.assertEqual(subregion.x, 20)
        self.assertEqual(subregion.width, 50)
        self.assertEqual(subregion.y, 30)
        self.assertEqual(subregion.height, 20)

    def test_subregion_above_gives_whole_region(self):
        from construction import RectangularRegion

        region = RectangularRegion(x=20, y=30, width=50, height=40)

        subregion = region.subregion_above(y=150)

        self.assertEqual(subregion.x, 20)
        self.assertEqual(subregion.width, 50)
        self.assertEqual(subregion.y, 30)
        self.assertEqual(subregion.height, 40)

    def test_subregion_above_produces_zero_area_region(self):
        from construction import RectangularRegion

        region = RectangularRegion(x=20, y=30, width=50, height=40)

        subregion = region.subregion_above(y=20)

        self.assertEqual(subregion.x, 20)
        self.assertEqual(subregion.width, 50)
        self.assertEqual(subregion.y, 30)
        self.assertEqual(subregion.height, 0)


class SubregionBelowTests(unittest.TestCase):
    def test_subregion_below_normal_case(self):
        from construction import RectangularRegion

        region = RectangularRegion(x=20, y=30, width=50, height=40)

        subregion = region.subregion_below(y=40)

        self.assertEqual(subregion.x, 20)
        self.assertEqual(subregion.width, 50)
        self.assertEqual(subregion.y, 40)
        self.assertEqual(subregion.height, 30)

    def test_subregion_below_gives_whole_region(self):
        from construction import RectangularRegion

        region = RectangularRegion(x=20, y=30, width=50, height=40)

        subregion = region.subregion_below(y=10)

        self.assertEqual(subregion.x, 20)
        self.assertEqual(subregion.width, 50)
        self.assertEqual(subregion.y, 30)
        self.assertEqual(subregion.height, 40)

    def test_subregion_below_produces_zero_area_region(self):
        from construction import RectangularRegion

        region = RectangularRegion(x=20, y=30, width=50, height=40)

        subregion = region.subregion_below(y=100)

        self.assertEqual(subregion.x, 20)
        self.assertEqual(subregion.width, 50)
        self.assertEqual(subregion.y, 70)
        self.assertEqual(subregion.height, 0)


class LeftSubregionTests(unittest.TestCase):
    def test_typical_case(self):
        from construction import RectangularRegion

        region = RectangularRegion(x=20, y=30, width=120, height=40)
        subregion = region.left_subregion(40)

        self.assertEqual(subregion.x, 20)
        self.assertEqual(subregion.width, 20)
        self.assertEqual(subregion.y, 30)
        self.assertEqual(subregion.height, 40)


class RightSubregionTests(unittest.TestCase):
    def test_typical_case(self):
        from construction import RectangularRegion

        region = RectangularRegion(x=20, y=30, width=50, height=40)
        subregion = region.right_subregion(40)

        self.assertEqual(subregion.x, 40)
        self.assertEqual(subregion.width, 30)
        self.assertEqual(subregion.y, 30)
        self.assertEqual(subregion.height, 40)


class LatexConstructionTests(unittest.TestCase):
    def test_with_single_digit(self):
        segments = [Digit('7', 225, 340)]
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '7')

    def test_with_number(self):
        segments = [Digit('7', 225, 340),
                    Digit('3', 260, 342),
                    Digit('8', 300, 338)]
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '738')

    def test_number1_plus_number2(self):

        num1_segments = self.create_number_segments(200, 400, '43')
        plus_segment = Digit('+', 300, 400)
        num2_segments = self.create_number_segments(350, 400, '538')

        segments = num1_segments + [plus_segment] + num2_segments
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '43 + 538')

    def test_number1_minus_number2(self):
        num1_segments = self.create_number_segments(200, 400, '43')
        minus_segment = Digit('-', 300, 400)
        num2_segments = self.create_number_segments(350, 400, '538')

        segments = num1_segments + [minus_segment] + num2_segments
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '43 - 538')

    def test_number1_times_number2(self):
        num1_segments = self.create_number_segments(200, 400, '43')
        minus_segment = Digit('times', 300, 400)
        num2_segments = self.create_number_segments(350, 400, '538')

        segments = num1_segments + [minus_segment] + num2_segments
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '43 * 538')

    def test_number1_over_number2(self):
        num1_segments = self.create_number_segments(200, 200, '43')
        minus_segment = Digit('div', 198, 260)
        num2_segments = self.create_number_segments(202, 320, '538')

        segments = num1_segments + [minus_segment] + num2_segments
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '\\frac{43}{538}')

    def create_number_segments(self, x, y, digits):
        segments = []
        dx = 45
        for digit in digits:
            segments.append(Digit(digit, x, y))
            x += dx
        return segments

    def test_digit_to_the_digit_power(self):
        segments = [Digit('7', 225, 340),
                    Digit('3', 240, 380)]
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '{7}^{3}')


if __name__ == '__main__':
    unittest.main()
