import numpy as np


class YoloVolume:
    def __init__(self, image_width, image_height, grid_size, num_classes, max_boxes=200):
        self.img_width = image_width
        self.img_height = image_height
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.max_boxes = max_boxes

        self._boxes = []
        self._classes = []

    def _prepare_output(self):
        confidence_score = 1
        bounding_box_size = 4
        classes_size = self.num_classes

        depth = confidence_score + bounding_box_size + classes_size
        volume = np.zeros((self.grid_size, self.grid_size, depth), dtype=np.float16)

        assert len(self._boxes) == len(self._classes)

        for i in range(len(self._classes)):

            box = self._boxes[i]
            label = self._classes[i]

            from keras.utils import to_categorical

            class_distribution = to_categorical(label,
                                                num_classes=self.num_classes)

            detection_confidence = 1.0

            xc, yc, width, height = box
            col, row = self.position_to_grid_cell(xc, yc)

            box_vector = self.get_bounding_box(xc, yc, col, row, width=width,
                                               height=height)

            volume[row, col] = np.concatenate(
                ([detection_confidence], box_vector, class_distribution)
            )

        return volume

    def detects_collision(self, bounding_box):
        comparison_box = self._to_shapely_box(bounding_box)

        for box in self._boxes:
            b = self._to_shapely_box(box)
            if comparison_box.intersection(b).area > 0:
                return True

        return False

    def _to_shapely_box(self, bounding_box):
        from shapely.geometry import box

        xc, yc, width, height = bounding_box
        x = xc - width // 2
        y = yc - height // 2
        return box(x, y, x + width, y + height)

    def add_item(self, bounding_box, class_index):
        self._boxes.append(bounding_box)
        self._classes.append(class_index)

    def position_to_grid_cell(self, xc, yc):
        x = xc // self.cell_width
        y = yc // self.cell_height

        return int(x), int(y)

    def relative_position(self, global_position, cell, cell_size):
        cell_position = cell_size * cell
        return (global_position - cell_position) / cell_size

    def get_bounding_box(self, xc, yc, col, row, width, height):
        j = col
        i = row

        xrel = self.relative_position(xc, j, self.cell_width)
        yrel = self.relative_position(yc, i, self.cell_height)

        w = width / self.cell_width
        h = height / self.cell_height

        return np.array([xrel, yrel, w, h])

    def make_prediction_vector(self, confidence, bounding_box, class_index):
        return np.concatenate(([confidence], bounding_box, [class_index]))

    @property
    def output_volume(self):
        return self._prepare_output()

    @property
    def compact_volume(self):
        assert len(self._boxes) == len(self._classes)

        tups = []

        for i in range(len(self._classes)):
            box = self._boxes[i]
            label = self._classes[i]

            xc, yc, width, height = box
            print(xc, yc, width, height)

            tups.append((xc, yc, width, height, label))

        size = min(self.max_boxes, len(self._classes))
        data_array = np.zeros((self.max_boxes, 5), dtype=np.uint16)

        data_array[:size, :] = np.array(tups)[:size, :]
        return [size] + data_array.flatten().tolist()

    @property
    def depth(self):
        confidence_score = 1
        bounding_box_size = 4
        label_size = 1
        return confidence_score + bounding_box_size + label_size

    @property
    def cell_width(self):
        return self.img_width / self.grid_size

    @property
    def cell_height(self):
        return self.img_height / self.grid_size

    def to_raw_data(self):
        vol = self.compact_volume
        return vol

    @staticmethod
    def from_raw_data(image_width, image_height, grid_size, num_classes, raw_data):

        volume = YoloVolume(image_width, image_height, grid_size, num_classes)

        size = raw_data[0]
        data = np.array(raw_data[1:]).reshape(-1, 5)

        for xc, yc, width, height, label in data[:size]:
            box = (xc, yc, width, height)
            volume.add_item(box, label)

        return volume
