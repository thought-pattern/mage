"""Utilities for events."""

from typing import Dict, List

from numpy import ndarray as np_ndarray


class Event:
    def __init__(self, source: int, timestamp: int):
        super(Event, self).__init__()
        self.source = source
        self.timestamp = timestamp

    def __str__(self):
        _return_value = "{source},{timestamp}".format(
            source=self.source, timestamp=self.timestamp
        )
        return _return_value


class NodeEvent(Event):
    def __init__(self, source: int, timestamp: int):
        super(NodeEvent, self).__init__(source, timestamp)

    def __str__(self):
        _return_value = "{source},{timestamp}".format(
            source=self.source, timestamp=self.timestamp
        )
        return _return_value


class InteractionEvent(Event):
    def __init__(self, source: int, dest: int, timestamp: int, edge_idx: int):
        super(InteractionEvent, self).__init__(source, timestamp)
        self.dest = dest
        self.edge_idx = edge_idx

    def __str__(self):
        _return_value = "{source},{timestamp}".format(
            source=self.source, timestamp=self.timestamp
        )
        return _return_value


def create_interaction_events(
    sources: np_ndarray,
    destinations: np_ndarray,
    timestamps: np_ndarray,
    edge_idxs: np_ndarray,
) -> Dict[int, List[InteractionEvent]]:
    "Every event has two interaction events"
    interaction_events: Dict[int, List[InteractionEvent]] = {
        node: [] for node in set(sources).union(set(destinations))
    }
    for i in range(len(sources)):
        interaction_events.get(sources[i], []).append(
            InteractionEvent(
                source=sources[i],
                dest=destinations[i],
                timestamp=timestamps[i],
                edge_idx=edge_idxs[i],
            )
        )
    return interaction_events
