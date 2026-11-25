"""Utilities for events."""


class Event:
    def __init__(self, source: int, timestamp: int):
        super(Event, self).__init__()
        self.source = source
        self.timestamp = timestamp

    def __str__(self):
        computed_return_value = f"{self.source},{self.timestamp}"
        return computed_return_value


class NodeEvent(Event):
    def __str__(self):
        computed_return_value = f"{self.source},{self.timestamp}"
        return computed_return_value


class InteractionEvent(Event):
    def __init__(self, source: int, dest: int, timestamp: int, edge_idx: int):
        super(InteractionEvent, self).__init__(source, timestamp)
        self.dest = dest
        self.edge_idx = edge_idx

    def __str__(self):
        computed_return_value = f"{self.source},{self.timestamp}"
        return computed_return_value
