from typing import Dict, Tuple


def weights_to_vector(conns: Dict[Tuple[int, int], Tuple[float, int]]):
    import jax.numpy as jnp

    return jnp.array([w for (w, _) in sorted(conns.values(), key=lambda x: x[1])])


def vector_to_weights(vec, conns: Dict[Tuple[int, int], Tuple[float, int]]):
    keys = sorted(conns.keys(), key=lambda k: conns[k][1])
    return {k: v for k, v in zip(keys, vec)}
