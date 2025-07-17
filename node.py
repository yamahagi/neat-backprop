class Node:
    def __init__(self, node_id: int, node_type: str, activation: str = "sigmoid"):
        self.id = node_id
        self.type = node_type
        self.activation = activation
