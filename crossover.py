import random

from individual import Individual
from network import EvolvingNEATNetwork


def crossover(parent1: Individual, parent2: Individual) -> Individual:
    import copy

    child_net = EvolvingNEATNetwork(parent1.net.tracker)

    for nid, node in parent1.net.nodes.items():
        child_net.nodes[nid] = copy.deepcopy(node)

    conn1 = {v[1]: (k, v[0]) for k, v in parent1.net.connections.items()}
    conn2 = {v[1]: (k, v[0]) for k, v in parent2.net.connections.items()}

    for innov in conn1:
        if innov in conn2:
            from_id, to_id = conn1[innov][0]
            chosen_weight = random.choice([conn1[innov][1], conn2[innov][1]])
        else:
            from_id, to_id = conn1[innov][0]
            chosen_weight = conn1[innov][1]

        child_net.add_connection(from_id, to_id, chosen_weight)

    child_net.next_node_id = max(child_net.nodes.keys()) + 1 if child_net.nodes else 0

    return Individual(child_net)
