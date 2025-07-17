import copy
import random

from crossover import crossover
from individual import Individual
from network import EvolvingNEATNetwork
from settings import ADD_CONNECTION_PROB, ADD_NODE_PROB, MAX_SPECIES
from species import Species, compatibility_distance


def mutate_add_node(network):
    if not network.connections:
        return
    conn = random.choice(list(network.connections.keys()))
    from_id, to_id = conn
    old_weight, _ = network.connections.pop(conn)
    new_node_id = network.add_node("hidden")
    network.add_connection(from_id, new_node_id, 1.0)
    network.add_connection(new_node_id, to_id, old_weight)


def mutate_add_connection(network):
    import random

    nodes = list(network.nodes.keys())
    existing_connections = set(network.connections.keys())
    invalid_to_ids = [
        nid for nid, node in network.nodes.items() if node.type in ["input", "bias"]
    ]
    # 自己ループ以外のすべての可能なペアを列挙
    possible_pairs = [
        (from_id, to_id)
        for from_id in nodes
        for to_id in nodes
        if from_id != to_id
        and (from_id, to_id) not in existing_connections
        and to_id not in invalid_to_ids  # input や bias に流れ込む接続を除外
    ]
    # 有効なペアが存在しない場合は何もしない
    if not possible_pairs:
        return
    # 有効なペアの中から1つランダムに選んで接続
    from_id, to_id = random.choice(possible_pairs)
    network.add_connection(from_id, to_id)


class Population:
    def __init__(self, size, input_size, output_size, tracker):
        self.tracker = tracker
        self.individuals = []
        for _ in range(size):
            net = EvolvingNEATNetwork(tracker)
            inputs = [net.add_node("input") for _ in range(input_size)]
            outputs = [
                net.add_node("output", activation="sigmoid") for _ in range(output_size)
            ]
            for i in inputs:
                for o in outputs:
                    net.add_connection(i, o)
            for o in outputs:
                net.add_connection(net.bias_node_id, o)
            self.individuals.append(Individual(net))
        self.species = []
        self.max_species = MAX_SPECIES

    def _merge_closest_species(self):
        min_distance = float("inf")
        pair_to_merge = None

        # すべてのクラスタ対を調べて、互いに最も近いペアを探す
        for i in range(len(self.species)):
            for j in range(i + 1, len(self.species)):
                d = compatibility_distance(
                    self.species[i].representative, self.species[j].representative
                )
                if d < min_distance:
                    min_distance = d
                    pair_to_merge = (i, j)

        if pair_to_merge is not None:
            i, j = pair_to_merge
            sp1 = self.species[i]
            sp2 = self.species[j]

            # sp2のメンバーをsp1に移動
            for member in sp2.members:
                sp1.add(member)

            # sp1の代表個体を更新
            sp1.update_fitness()

            # sp2をリストから削除
            self.species.pop(j)

    def speciate(self, threshold=3.0):
        self.species = []
        for ind in self.individuals:
            placed = False
            for sp in self.species:
                if compatibility_distance(ind, sp.representative) < threshold:
                    sp.add(ind)
                    placed = True
                    break
            if not placed:
                self.species.append(Species(ind))
        while len(self.species) > self.max_species:
            self._merge_closest_species()

    def evolve(
        self, X_train, y_train, X_valid, y_valid, train_steps=10, elite_fraction=0.2
    ):
        for ind in self.individuals:
            if ind.needs_training:
                ind.train(X_train, y_train, steps=train_steps)
            ind.evaluate(X_valid, y_valid)

        self.speciate()
        new_population = []

        for sp in self.species:
            sp.members.sort(key=lambda i: i.fitness, reverse=True)
            sp.update_fitness()

            num_elites = max(1, int(elite_fraction * len(sp.members)))
            elites = sp.members[:num_elites]
            new_population.extend(elites)

            num_offspring = len(sp.members) - num_elites
            for _ in range(num_offspring):
                if len(sp.members) > 1:
                    p1, p2 = random.sample(sp.members, 2)
                    child = crossover(p1, p2)
                    child.needs_training = True
                else:
                    child = copy.deepcopy(elites[0])
                    child.needs_training = True
                if random.random() < ADD_NODE_PROB:
                    mutate_add_node(child.net)
                if random.random() < ADD_CONNECTION_PROB:
                    mutate_add_connection(child.net)
                new_population.append(child)

        self.individuals = new_population
