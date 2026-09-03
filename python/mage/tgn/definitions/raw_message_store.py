"""Utilities for raw message store."""

from collections import defaultdict

from mage.tgn.definitions.messages import RawMessage


class RawMessageStore:
    """
    This class represents store for instances of Raw Messages
    """

    def __init__(self, edge_raw_message_dimension: int, node_raw_message_dimension: int):
        if edge_raw_message_dimension < 1 or node_raw_message_dimension < 1:
            raise ValueError("Raw-message dimensions must be positive")
        self.edge_raw_message_dimension = edge_raw_message_dimension
        self.node_raw_message_dimension = node_raw_message_dimension
        self.init_message_store()

    def init_message_store(self) -> bool:
        self.message_container: dict[int, list[RawMessage]] = defaultdict(list)
        return False

    def detach_grads(self) -> bool:
        for messages in self.message_container.values():
            for message in messages:
                message.detach_memory()
        return False

    def get_messages(self) -> dict[int, list[RawMessage]]:
        return self.message_container

    def update_messages(self, new_node_messages: dict[int, list[RawMessage]]) -> bool:
        for node in new_node_messages:
            stored_messages = self.message_container.get(node, [])
            stored_messages.extend(new_node_messages.get(node, []))
            self.message_container[node] = stored_messages
        return False
