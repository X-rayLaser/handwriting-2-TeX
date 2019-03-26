import numpy as np


def embed_noise(a, noise=50):
    assert a.ndim == 2
    res = np.zeros_like(a, dtype=np.int)
    res[:, :] = np.minimum(np.random.randn(*a.shape) * noise + a, 255)
    return np.maximum(res, 0)


def rotate(a, angle):
    assert a.ndims == 2
    rads = angle * np.pi / 180.0

    cos_phi = np.cos(rads)
    sin_phi = np.sin(rads)
    R = np.array([[cos_phi, -sin_phi],
                  [sin_phi, -cos_phi]])

    indices = []
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            indices.append((j, i))

    X = np.array(indices).T

    Xprime = np.dot(R, X)

    res = np.zeros_like(a)
    for j in range(a.shape[1]):
        col = Xprime[0, j]
        row = Xprime[1, j]

    nelements

    return a
