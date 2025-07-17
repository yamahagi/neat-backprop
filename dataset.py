import random
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np


class DataPoint:
    def __init__(self, x: float, y: float, label: int):
        self.x = x
        self.y = y
        self.label = label


def randf(a: float, b: float) -> float:
    return random.uniform(a, b)


def randn(mean: float = 0.0, stddev: float = 1.0) -> float:
    return np.random.normal(mean, stddev)


def shuffle_data(data: List[DataPoint]) -> List[DataPoint]:
    random.shuffle(data)
    return data


def generate_xor_data(N=200, noise=0.5) -> List[DataPoint]:
    data = []
    for _ in range(N):
        x = randf(-5.0, 5.0) + randn(0, noise)
        y = randf(-5.0, 5.0) + randn(0, noise)
        label = 1 if (x > 0 and y > 0) or (x < 0 and y < 0) else 0
        data.append(DataPoint(x, y, label))
    return data


def generate_spiral_data(N=200, noise=0.5) -> List[DataPoint]:
    data = []
    n = N // 2

    def gen_spiral(delta_t, label):
        for i in range(n):
            r = i / n * 6.0
            t = 1.75 * i / n * 2 * np.pi + delta_t
            x = r * np.sin(t) + randf(-1, 1) * noise
            y = r * np.cos(t) + randf(-1, 1) * noise
            data.append(DataPoint(x, y, label))

    gen_spiral(0, 1)
    gen_spiral(np.pi, 0)
    return data


def generate_gaussian_data(N=200, noise=0.5) -> List[DataPoint]:
    data = []
    n = N // 2

    def gen_cluster(cx, cy, label):
        for _ in range(n):
            x = randn(cx, noise + 1.0)
            y = randn(cy, noise + 1.0)
            data.append(DataPoint(x, y, label))

    gen_cluster(2, 2, 1)
    gen_cluster(-2, -2, 0)
    return data


def generate_circle_data(N=200, noise=0.5, radius=5.0) -> List[DataPoint]:
    data = []
    n = N // 2

    def get_label(x, y):
        return 1 if x**2 + y**2 < (radius * 0.5) ** 2 else 0

    for _ in range(n):
        r = randf(0, radius * 0.5)
        angle = randf(0, 2 * np.pi)
        x = r * np.sin(angle) + randf(-radius, radius) * noise / 3
        y = r * np.cos(angle) + randf(-radius, radius) * noise / 3
        data.append(DataPoint(x, y, get_label(x, y)))

    for _ in range(n):
        r = randf(radius * 0.75, radius)
        angle = randf(0, 2 * np.pi)
        x = r * np.sin(angle) + randf(-radius, radius) * noise / 3
        y = r * np.cos(angle) + randf(-radius, radius) * noise / 3
        data.append(DataPoint(x, y, get_label(x, y)))

    return data


def generate_random_data(choice=None, train_size=200, test_size=200, noise=0.5):
    if choice is None:
        choice = random.randint(0, 3)

    if choice == 0:
        train = generate_circle_data(train_size, noise)
        test = generate_circle_data(test_size, noise)
    elif choice == 1:
        train = generate_xor_data(train_size, noise)
        test = generate_xor_data(test_size, noise)
    elif choice == 2:
        train = generate_gaussian_data(train_size, noise)
        test = generate_gaussian_data(test_size, noise)
    else:
        train = generate_spiral_data(train_size, noise)
        test = generate_spiral_data(test_size, noise)

    train = shuffle_data(train)
    test = shuffle_data(test)
    return train, test


def to_jax(data: List[DataPoint]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    DataPointのリストをJAXのjnp.ndarrayに変換する。

    Returns:
        - X: (N, 2) の float32 配列
        - y: (N,) の float32 配列（整数ラベルが float32 にキャストされる点に注意）
    """
    X = jnp.array([[p.x, p.y] for p in data], dtype=jnp.float32)
    y = jnp.array([p.label for p in data], dtype=jnp.float32)
    return X, y
