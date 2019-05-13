import random
from data_synthesis import Canvas
from .volume import YoloVolume


class BoundingBoxSuggestionError(Exception):
    pass


class YoloDataGenerator:
    def __init__(self, img_width, img_height, primitives_dir, grid_size=9, num_classes=14):
        self.img_width = img_width
        self.img_height = img_height
        self.primitives_dir = primitives_dir
        self.grid_size = grid_size
        self.num_classes = num_classes

    @property
    def cell_width(self):
        return self.img_width / self.grid_size

    @property
    def cell_height(self):
        return self.img_height / self.grid_size

    def choose_category(self):
        from dataset_utils import index_to_class, class_to_index

        c = random.choice(index_to_class)
        index = class_to_index[c]
        return index, c

    def choose_global_position(self):
        x = random.randint(0, self.img_width - 1)
        y = random.randint(0, self.img_height - 1)

        return x, y

    def try_choosing_position(self, volume, max_retries=10):
        for i in range(max_retries):
            xc, yc = self.choose_global_position()

            width, height = 45, 45
            bounding_box = (xc, yc, width, height)

            if not volume.detects_collision(bounding_box):
                return bounding_box

        raise BoundingBoxSuggestionError()

    def make_example(self, elements=40):
        canvas = Canvas(self.img_width, self.img_height, self.primitives_dir)

        volume = YoloVolume(self.img_width, self.img_height, self.grid_size, self.num_classes)

        for _ in range(elements):
            class_index, category = self.choose_category()

            try:
                bounding_box = self.try_choosing_position(volume)
            except BoundingBoxSuggestionError:
                continue
            else:
                volume.add_item(bounding_box, class_index=class_index)

                xc, yc, _, _ = bounding_box
                x = xc - 45 // 2
                y = yc - 45 // 2
                canvas.draw_random_class_image(x, y, category)

        input = canvas.image_data
        return input, volume.to_raw_data()
