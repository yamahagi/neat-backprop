def compatibility_distance(ind1, ind2, c1=1.0, c2=1.0, c3=0.4):
    genes1 = ind1.net.connections
    genes2 = ind2.net.connections

    innovs1 = {v[1]: (k, v[0]) for k, v in genes1.items()}
    innovs2 = {v[1]: (k, v[0]) for k, v in genes2.items()}

    all_innovs = set(innovs1.keys()) | set(innovs2.keys())
    max_innov1 = max(innovs1) if innovs1 else -1
    max_innov2 = max(innovs2) if innovs2 else -1
    max_innov = max(max_innov1, max_innov2)

    excess = 0
    disjoint = 0
    weight_diffs = []
    matching = 0

    for innov in all_innovs:
        in1 = innov in innovs1
        in2 = innov in innovs2
        if in1 and in2:
            # matching gene
            w1 = innovs1[innov][1]
            w2 = innovs2[innov][1]
            weight_diffs.append(abs(w1 - w2))
            matching += 1
        elif innov > max_innov1 or innov > max_innov2:
            excess += 1
        else:
            disjoint += 1

    # 正規化係数 N
    N = max(len(genes1), len(genes2))
    if N < 20:
        N = 1

    W = sum(weight_diffs) / matching if matching > 0 else 0.0

    δ = (c1 * excess / N) + (c2 * disjoint / N) + (c3 * W)
    return δ


class Species:
    def __init__(self, representative):
        self.representative = representative
        self.members = [representative]
        self.best_fitness = representative.fitness

    def add(self, ind):
        self.members.append(ind)

    def update_fitness(self):
        for m in self.members:
            if m.fitness > self.best_fitness:
                self.best_fitness = m.fitness
                self.representative = m
