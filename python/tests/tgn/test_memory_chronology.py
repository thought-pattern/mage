"""Tests for TGN memory update chronology."""

from numpy import array as np_array
from pytest import (
    mark as pytest_mark,
    raises as pytest_raises,
)
from torch import (
    device as torch_device,
    zeros as torch_zeros,
)

from mage.tgn.constants import (
    MemoryUpdaterType,
    MessageAggregatorType,
    MessageFunctionType,
    TGNLayerType,
)
from mage.tgn.definitions.messages import InteractionRawMessage
from mage.tgn.definitions.tgn import TGN

SOURCE_NODE = 1
DEST_NODE = 2
NUM_EDGE_FEATURES = 2
NUM_NODE_FEATURES = 2

# Edge IDs are event timestamps. float32 holds integers above 2**24 only in steps of two or more, so these IDs are
# not representable in it: 33554438 is the smallest ID that float32 rounds past its successor.
LARGE_EDGE_IDS = [33554438, 67108869, 2**40 + 1]


def build_tgn() -> TGN:
    return TGN(
        num_of_layers=1,
        layer_type=TGNLayerType.GraphSumEmbedding,
        memory_dimension=4,
        time_dimension=4,
        num_edge_features=NUM_EDGE_FEATURES,
        num_node_features=NUM_NODE_FEATURES,
        message_dimension=4,
        num_neighbors=2,
        edge_message_function_type=MessageFunctionType.Identity,
        message_aggregator_type=MessageAggregatorType.Last,
        memory_updater_type=MemoryUpdaterType.GRU,
        device=torch_device("cpu"),
    )


def pending_delta_times(tgn: TGN, node: int) -> list[float]:
    """Delta times of the interaction messages waiting in the store for a node."""
    messages = tgn.raw_message_store.get_messages().get(node, [])
    return [float(message.delta_time.item()) for message in messages if isinstance(message, InteractionRawMessage)]


def admit_interaction(tgn: TGN, edge_id: int) -> bool:
    """Admits one interaction between the two test nodes, with the edge ID as its timestamp."""
    tgn.update_raw_message_store_current_batch(
        sources=np_array([SOURCE_NODE], dtype=int),
        destinations=np_array([DEST_NODE], dtype=int),
        timestamps=np_array([edge_id], dtype=int),
        edge_idxs=np_array([edge_id], dtype=int),
        edge_features={edge_id: torch_zeros(NUM_EDGE_FEATURES)},
        node_features={
            SOURCE_NODE: torch_zeros(NUM_NODE_FEATURES),
            DEST_NODE: torch_zeros(NUM_NODE_FEATURES),
        },
    )
    return False


class TestMemoryChronology:
    """Tests for the last-update time kept with each node's memory."""

    @pytest_mark.parametrize("edge_id", LARGE_EDGE_IDS)
    def test_last_update_keeps_exact_timestamp(self, edge_id: int):
        tgn = build_tgn()
        admit_interaction(tgn, edge_id)
        tgn.internal_process_previous_batches()

        for node in (SOURCE_NODE, DEST_NODE):
            assert int(tgn.memory.get_last_node_update(node).item()) == edge_id

    @pytest_mark.parametrize("edge_id", LARGE_EDGE_IDS)
    def test_next_edge_id_is_admitted_after_memory_update(self, edge_id: int):
        tgn = build_tgn()
        admit_interaction(tgn, edge_id)
        tgn.internal_process_previous_batches()

        admit_interaction(tgn, edge_id + 1)

        for node in (SOURCE_NODE, DEST_NODE):
            assert pending_delta_times(tgn, node) == [1.0]

    @pytest_mark.parametrize("edge_id", LARGE_EDGE_IDS)
    def test_same_timestamp_is_admitted_after_memory_update(self, edge_id: int):
        tgn = build_tgn()
        admit_interaction(tgn, edge_id)
        tgn.internal_process_previous_batches()

        admit_interaction(tgn, edge_id)

        for node in (SOURCE_NODE, DEST_NODE):
            assert pending_delta_times(tgn, node) == [0.0]

    @pytest_mark.parametrize("edge_id", LARGE_EDGE_IDS)
    def test_earlier_event_is_rejected_after_memory_update(self, edge_id: int):
        tgn = build_tgn()
        admit_interaction(tgn, edge_id)
        tgn.internal_process_previous_batches()

        with pytest_raises(ValueError, match="precedes node"):
            admit_interaction(tgn, edge_id - 1)
