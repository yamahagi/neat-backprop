class InnovationTracker:
    def __init__(self):
        self.counter = 0
        self.innovations = {}  # (from_id, to_id) -> innovation number

    def get(self, from_id: int, to_id: int) -> int:
        key = (from_id, to_id)
        if key not in self.innovations:
            self.innovations[key] = self.counter
            self.counter += 1
        return self.innovations[key]
