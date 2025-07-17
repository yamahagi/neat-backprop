import random
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from node import Node


def gaussian(x):
    return jnp.exp(-(x**2))


def sine(x):
    return jnp.sin(x)


def abs_fn(x):
    return jnp.abs(x)


def mult(x):
    return x * x  # same as square


def add_fn(x):
    return x + x


def square(x):
    return x**2


activation_funcs = {
    "sigmoid": jax.nn.sigmoid,
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "gaussian": gaussian,
    "sine": sine,
    "abs": abs_fn,
    "mult": mult,
    "add": add_fn,
    "square": square,
}

activation_names = list(activation_funcs.keys())


class EvolvingNEATNetwork:
    def __init__(self, tracker):
        self.nodes: Dict[int, Node] = {}
        self.connections: Dict[
            Tuple[int, int], Tuple[float, int]
        ] = {}  # (from, to): (weight, innovation)
        self.next_node_id = 0
        self.tracker = tracker
        self.bias_node_id = self.add_node("bias")

    def __deepcopy__(self, memo):
        from copy import deepcopy

        new_net = EvolvingNEATNetwork(self.tracker)
        new_net.nodes = deepcopy(self.nodes, memo)
        new_net.connections = deepcopy(self.connections, memo)
        new_net.next_node_id = self.next_node_id
        new_net.bias_node_id = self.bias_node_id
        return new_net

    def add_node(self, node_type: str, activation: str = None) -> int:
        if activation is None:
            activation = random.choice(activation_names)
        nid = self.next_node_id
        self.nodes[nid] = Node(nid, node_type, activation)
        self.next_node_id += 1
        return nid

    def add_connection(self, from_id: int, to_id: int, weight: float = None):
        if (from_id, to_id) in self.connections:
            return
        if from_id == to_id:
            return

        temp_weight = (
            weight if weight is not None else random.gauss(0.0, 1.0)
        )  # 正規分布から初期化
        innov = self.tracker.get(from_id, to_id)
        self.connections[(from_id, to_id)] = (temp_weight, innov)

        if self.has_cycle():
            del self.connections[(from_id, to_id)]
            return

    def has_cycle(self) -> bool:
        visited = set()
        rec_stack = set()

        def visit(node_id):
            if node_id not in visited:
                visited.add(node_id)
                rec_stack.add(node_id)
                for src, dst in self.connections:
                    if src == node_id:
                        if dst in rec_stack or visit(dst):
                            return True
                rec_stack.remove(node_id)
            return False

        return any(visit(nid) for nid in self.nodes)

    def topological_sort(self):
        visited = set()
        result = []

        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            for src, dst in self.connections:
                if dst == nid:
                    visit(src)
            result.append(nid)

        for nid in self.nodes:
            visit(nid)

        return list(dict.fromkeys(result))

    def forward(
        self, x: jnp.ndarray, weights_override: Dict[Tuple[int, int], float] = None
    ) -> jnp.ndarray:
        values = {}
        sorted_ids = self.topological_sort()
        input_ids = [nid for nid, node in self.nodes.items() if node.type == "input"]

        for i, nid in enumerate(input_ids):
            values[nid] = x[i]
        values[self.bias_node_id] = 1.0

        for nid in sorted_ids:
            if self.nodes[nid].type in ["input", "bias"]:
                continue
            total = 0.0
            for (src, dst), (w, _) in self.connections.items():
                if dst == nid:
                    if src not in values:
                        raise ValueError(
                            f"Missing value for node {src}. Cannot compute {dst}."
                        )
                    if weights_override and (src, dst) in weights_override:
                        w = weights_override[(src, dst)]
                    total += values[src] * w
            values[nid] = activation_funcs[self.nodes[nid].activation](total)

        output_ids = [nid for nid, node in self.nodes.items() if node.type == "output"]
        return jnp.array([values[nid] for nid in output_ids])
