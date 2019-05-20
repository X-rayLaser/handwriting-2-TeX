class SpatialGraph:
    def to_sequence(self):
        pass

    def to_array(self, grid_size=25):
        pass


def to_vector(a, num_classes=14):
    v = [0] * num_classes
    v[a] = 1
    return v


classes = {
    '+': 10,
    '-': 11,
    'x': 12,
    '': 13
}

for i in range(10):
    classes[str(i)] = i


def make_encoding(current_element, neighbors):
    res = []
    all_elements = current_element + neighbors
    for elem in all_elements:
        res += (to_vector(classes[elem]))
    return res


def fraction_graph(a, b, c, d):
    x = []
    va = to_vector(a)
    vb = to_vector(b)
    vc = to_vector(c)
    vd = to_vector(d)

    a_enc = make_encoding(a, ['+', '-', '', ''])
    plus_enc = make_encoding('+', [b, '-', a, ''])
    b_enc = make_encoding()
    a_encoding = va + to_vector(classes['+']) + to_vector(classes['-']) + to_vector(classes['blank']) + to_vector('blank')

    x.append((a, '+'))
