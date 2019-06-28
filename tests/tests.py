import unittest
from util import connect_chains, ChainOfPoints, chains_sorted_by_distance
from building_blocks import Primitive
from construction import construct_latex
from segmentation import locate_digits
import numpy as np


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


class SubregionCornerCasesTests(unittest.TestCase):
    def test_subregion_formed_outside_original_area(self):
        from construction import RectangularRegion

        region = RectangularRegion(x=20, y=30, width=50, height=40)
        region.left_subregion(-10).subregion_below(400)


class LatexConstructionTests(unittest.TestCase):
    def test_with_single_digit(self):

        segments = [Primitive.new_primitive('7', 225, 340)]
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '7')

    def test_with_number(self):
        segments = [Primitive.new_primitive('7', 225, 340),
                    Primitive.new_primitive('3', 260, 342),
                    Primitive.new_primitive('8', 300, 338)]
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '738')

    def test_number1_plus_number2(self):

        num1_segments = self.create_number_segments(200, 400, '43')
        plus_segment = Primitive.new_primitive('+', 300, 400)
        num2_segments = self.create_number_segments(350, 400, '538')

        segments = num1_segments + [plus_segment] + num2_segments
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '43 + 538')

    def test_number1_minus_number2(self):
        num1_segments = self.create_number_segments(200, 400, '43')
        minus_segment = Primitive.new_primitive('-', 300, 400)
        num2_segments = self.create_number_segments(350, 400, '538')

        segments = num1_segments + [minus_segment] + num2_segments
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '43 - 538')

    def test_number1_times_number2(self):
        num1_segments = self.create_number_segments(200, 400, '43')
        minus_segment = Primitive.new_primitive('times', 300, 400)
        num2_segments = self.create_number_segments(350, 400, '538')

        segments = num1_segments + [minus_segment] + num2_segments
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '43 * 538')

    def test_number1_over_number2(self):
        num1_segments = self.create_number_segments(200, 200, '4')
        div_segment = Primitive.new_primitive('div', 198, 260)
        num2_segments = self.create_number_segments(198, 320, '5')

        segments = num1_segments + [div_segment] + num2_segments
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '\\frac{4}{5}')

    def create_number_segments(self, x, y, digits):
        segments = []
        dx = 45
        for digit in digits:
            segments.append(Primitive.new_primitive(digit, x, y))
            x += dx
        return segments

    def test_digit_to_the_digit_power(self):
        segments = [Primitive.new_primitive('7', 225, 340),
                    Primitive.new_primitive('3', 250, 280)]
        latex = construct_latex(segments=segments, width=500, height=500)
        self.assertEqual(latex, '7^{3}')


class SplitIntervalTests(unittest.TestCase):
    def test_outputs_single_section_for_small_width(self):
        from data_synthesis import split_interval
        sections = split_interval(interval_len=120, n=1)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0], 120)

    def test_outputs_add_to_original_len(self):
        from data_synthesis import split_interval
        sections = split_interval(interval_len=640, n=5)
        self.assertEqual(len(sections), 5)
        self.assertAlmostEqual(sum(sections), 640)

    def test_outputs_are_in_valid_range(self):
        from data_synthesis import split_interval
        sections = split_interval(interval_len=340, n=2)
        self.assertLess(sections[0], 340)
        self.assertLess(sections[1], 340)

        self.assertGreater(sections[0], 0)
        self.assertGreater(sections[1], 0)


class ReduceSectionsTests(unittest.TestCase):
    def test_with_one_section(self):
        from data_synthesis import reduce_sections
        sections = reduce_sections([30], min_size=100)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections, [30])

        sections = reduce_sections([130], min_size=100)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections, [130])

    def test_outputs_sizes_are_in_valid_range(self):
        from data_synthesis import reduce_sections

        sections = reduce_sections([30, 120, 203, 450, 50, 30], min_size=100)
        self.assertEqual(len(sections), 4)
        sections.sort(reverse=True)
        self.assertEqual(sections, [450, 203, 120, 110])

    def test_with_all_sections_being_small(self):
        from data_synthesis import reduce_sections

        sections = reduce_sections([20, 10, 20, 30, 40], min_size=100)
        sections.sort(reverse=True)
        self.assertEqual(sections, [120])

    def test_with_special_sequence(self):
        from data_synthesis import reduce_sections

        sections = reduce_sections([10, 10, 10, 30, 40, 40, 90], min_size=70)
        sections.sort(reverse=True)
        self.assertEqual(sections, [140, 90])


class SplitHorizontallyTests(unittest.TestCase):
    def test_outputs_single_region_for_small_width(self):
        from data_synthesis import split_horizontally
        regions = split_horizontally(width=120, height=390, max_n=1, min_size=30)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].x, 0)
        self.assertEqual(regions[0].y, 0)
        self.assertEqual(regions[0].width, 120)
        self.assertEqual(regions[0].height, 390)

    def test_resulting_regions_are_located_side_by_side(self):
        from data_synthesis import split_horizontally
        regions = split_horizontally(600, 400, max_n=10, min_size=80)
        for i in range(1, len(regions)):
            self.assertAlmostEqual(regions[i].x, regions[i-1].x + regions[i-1].width)


class OverlayImageTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        canvas = np.zeros((3, 3))
        self.canvas = canvas

    def test_overlay_well_within_boundaries(self):
        from data_synthesis import overlay_image
        from building_blocks import RectangularRegion

        import numpy as np
        img = np.array([[2, 4], [3, 99]])

        overlay_image(self.canvas, img, x=1, y=1)

        self.assertTupleEqual(self.canvas.shape, (3, 3))

        a = np.array([[0, 0, 0],
                      [0, 2, 4],
                      [0, 3, 99]])
        self.assertEqual(self.canvas.tolist(), a.tolist())

    def test_overlay_when_position_out_of_bounds(self):
        from data_synthesis import overlay_image
        import numpy as np
        img = np.array([[2, 4], [3, 99]])

        overlay_image(self.canvas, img, x=1, y=3)
        overlay_image(self.canvas, img, x=3, y=2)
        overlay_image(self.canvas, img, x=30, y=20)
        overlay_image(self.canvas, img, x=-3, y=-3)

        self.assertTupleEqual(self.canvas.shape, (3, 3))

        self.assertTrue(np.all(self.canvas == 0))

    def test_overlay_when_part_of_image_out_of_bounds(self):
        from data_synthesis import overlay_image
        import numpy as np
        img = np.array([[2, 4], [3, 99]])

        overlay_image(self.canvas, img, x=2, y=1)

        self.assertTupleEqual(self.canvas.shape, (3, 3))

        a = np.array([[0, 0, 0],
                      [0, 0, 2],
                      [0, 0, 3]])
        self.assertEqual(self.canvas.tolist(), a.tolist())

    def test_overlay_thrice(self):
        from data_synthesis import overlay_image

        import numpy as np
        img = np.array([[2, 4], [3, 99]])

        overlay_image(self.canvas, img, x=1, y=1)
        overlay_image(self.canvas, img, x=1, y=1)
        overlay_image(self.canvas, img, x=1, y=1)

        self.assertTupleEqual(self.canvas.shape, (3, 3))

        a = np.array([[0, 0, 0],
                      [0, 6, 12],
                      [0, 9, 255]])
        self.assertEqual(self.canvas.tolist(), a.tolist())


class SegmentationTests(unittest.TestCase):
    def test_object_intersects_left_image_border(self):
        a = np.zeros((5, 5))
        a[3:4, 2] = 243
        locate_digits(a)

    def test_object_intersects_right_image_border(self):
        a = np.zeros((5, 5))
        a[2, 4] = 243
        locate_digits(a)

    def test_object_intersects_upper_image_border(self):
        a = np.zeros((5, 5))
        a[0, 4] = 243
        locate_digits(a)

    def test_object_intersects_bottom_image_border(self):
        a = np.zeros((5, 5))
        a[4, 4] = 243
        locate_digits(a)


if __name__ == '__main__':
    unittest.main()
