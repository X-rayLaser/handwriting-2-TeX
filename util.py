import numpy as np


def augmented_dataset_generator(images, labels, batch_size=32):
    import keras
    from keras.preprocessing.image import ImageDataGenerator

    m = len(images)
    x_train = np.array(images).reshape((m, 28, 28, 1))
    y_train = np.array(labels)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=8,
        width_shift_range=4,
        height_shift_range=4,
        zoom_range=0.25,
        horizontal_flip=False)

    datagen.fit(x_train)
    gen = datagen.flow(x_train, y_train, batch_size=batch_size)
    return gen


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, p):
        dx = self.x - p.x
        dy = self.y - p.y
        return np.hypot(dx, dy)


class ChainOfPoints:
    def __init__(self, points):
        self.points = [Point(x, y) for x, y in points]

    def extend(self, chain):
        self.points.extend(chain.points)

    def neighbor_of(self, chain, max_distance=30):
        return self.distance_to(chain) < max_distance

    def min_x(self):
        X = [p.x for p in self.points]
        return np.min(np.array(X))

    def max_x(self):
        X = [p.x for p in self.points]
        return np.max(np.array(X))

    def min_y(self):
        Y = [p.y for p in self.points]
        return np.min(np.array(Y))

    def max_y(self):
        Y = [p.y for p in self.points]
        return np.max(np.array(Y))

    def distance_to(self, chain):
        pairs = []
        for p1 in self.points:
            for p2 in chain.points:
                pairs.append((p1, p2))

        distance = 10**100
        for p1, p2 in pairs:
            distance = min(p1.distance_to(p2), distance)

        return distance


def chains_sorted_by_distance(source, chains):
    x, y = source
    p = ChainOfPoints([(x, y)])
    return sorted(chains, key=lambda chain: chain.distance_to(p))


def connect_components(components, index_to_point, point_to_index):
    chains = []
    for i in range(len(components)):
        points = [index_to_point(index) for index in components[i]]
        chains.append(ChainOfPoints(points))

    clustered_chains = connect_chains(chains)

    res = []
    for chain in clustered_chains:
        component = set()
        for p in chain.points:
            component = component.union([point_to_index(p.x, p.y)])

        res.append(component)

    return res


def connect_chains(chains):
    res = []
    remaining = chains
    buffer = []
    while remaining:
        drawing = remaining.pop()
        source = (drawing.min_x(), drawing.min_y())

        res.append(drawing)
        sorted_drawings = chains_sorted_by_distance(source, remaining)

        for chain in sorted_drawings:
            if chain.neighbor_of(drawing):
                drawing.extend(chain)
            else:
                buffer.append(chain)

        remaining = buffer
        buffer = []

    return res
