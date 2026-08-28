"""Utilities for meta util."""

from collections import defaultdict
from typing import Dict, Iterator, Tuple

from mgp import List as mgp_List
from mgp import Map as mgp_Map
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import read_proc as mgp_read_proc

from mage.meta_util.parameters import Parameter

NodeKeyType = Tuple[str, ...]
RelationshipKeyType = Tuple[NodeKeyType, str, NodeKeyType]


class Counter:
    def __init__(self, initial_value: int = 0):
        self.total_count = initial_value
        self.count_by_property_name = defaultdict(int)

    def increment(self) -> bool:
        self.total_count += 1
        return False

    def increment_property(self, property_name: str) -> bool:
        self.count_by_property_name[property_name] += 1
        return False

    def to_dict(self, include_properties):
        _return_value = (
            {
                Parameter.COUNT.value: self.total_count,
                Parameter.PROPERTIES_COUNT.value: self.count_by_property_name,
            }
            if include_properties
            else {Parameter.COUNT.value: self.total_count}
        )
        return _return_value


@mgp_read_proc
def schema(context: mgp_ProcCtx, include_properties: bool = False) -> mgp_Record(
    nodes=mgp_List[mgp_Map], relationships=mgp_List[mgp_Map]
):
    (
        "\n    Procedure to generate the graph database schema.\n\n    Args:\n        context (mgp.ProcCt"  # Continue literal.
        "x): Reference to the context execution.\n        include_properties (bool): If set to True, t"  # Continue literal.
        "he graph schema will include properties count information.\n\n    Returns:\n        mgp.Record "  # Continue literal.
        "containing a mgp.List of mgp.Map objects representing nodes in the graph schema and a mgp.Li"  # Continue literal.
        "st of mgp.Map objects representing relationships.\n\n    Example:\n        Get graph schema wit"  # Continue literal.
        "hout properties count:\n            `CALL meta_util.schema() YIELD nodes, relationships RETUR"  # Continue literal.
        "N nodes, relationships;`\n        Get graph schema with properties count:\n            `CALL m"  # Continue literal.
        "eta_util.schema(true) YIELD nodes, relationships RETURN nodes, relationships;`\n"
    )

    node_count_by_labels: Dict[NodeKeyType, Counter] = {}
    relationship_count_by_labels: Dict[RelationshipKeyType, Counter] = {}

    node_counter = 0

    for node in context.graph.vertices:
        node_counter += 1
        labels = tuple(sorted(label.name for label in node.labels))
        _update_counts(
            node_count_by_labels,
            key=labels,
            obj=node,
            include_properties=include_properties,
        )

        for relationship in node.out_edges:
            target_labels = tuple(
                sorted(label.name for label in relationship.to_vertex.labels)
            )
            key = (labels, relationship.type.name, target_labels)
            _update_counts(
                relationship_count_by_labels,
                key=key,
                obj=relationship,
                include_properties=include_properties,
            )

    if node_counter == 0:
        raise Exception(
            "Can't generate a graph schema since there is no data in the database."
        )

    node_index_by_labels = {key: i for i, key in enumerate(node_count_by_labels.keys())}
    nodes = list(
        _iter_nodes_as_map(
            node_count_by_labels, node_index_by_labels, include_properties
        )
    )
    relationships = list(
        _iter_relationships_as_map(
            relationship_count_by_labels,
            node_index_by_labels,
            include_properties,
        )
    )

    _return_value = mgp_Record(nodes=nodes, relationships=relationships)
    return _return_value


def _update_counts(
    obj_count_by_key: Dict[object, Counter],
    key: object,
    obj: object,
    include_properties: bool = False,
) -> bool:
    if key not in obj_count_by_key:
        obj_count_by_key[key] = Counter()

    obj_counter = obj_count_by_key.get(key, False)
    obj_counter.increment()

    if include_properties:
        for property_name in obj.properties.keys():
            obj_counter.increment_property(property_name)
    return False


def _iter_nodes_as_map(
    node_count_by_labels: Dict[NodeKeyType, Counter],
    node_index_by_labels: Dict[NodeKeyType, int],
    include_properties: bool,
) -> Iterator[mgp_Map]:
    for labels, counter in node_count_by_labels.items():
        yield {
            Parameter.ID.value: node_index_by_labels.get(labels, False),
            Parameter.LABELS.value: labels,
            Parameter.PROPERTIES.value: counter.to_dict(include_properties),
            Parameter.TYPE.value: Parameter.NODE.value,
        }


def _iter_relationships_as_map(
    relationship_count_by_labels: Dict[RelationshipKeyType, Counter],
    node_index_by_labels: Dict[NodeKeyType, int],
    include_properties: bool,
) -> Iterator[mgp_Map]:
    for i, (
        (source_label, relationship_label, target_label),
        counter,
    ) in enumerate(relationship_count_by_labels.items()):
        source_node_id = node_index_by_labels.get(source_label, False)
        target_node_id = node_index_by_labels.get(target_label, False)

        if source_node_id is not False and target_node_id is not False:
            yield {
                Parameter.ID.value: i,
                Parameter.START.value: source_node_id,
                Parameter.END.value: target_node_id,
                Parameter.LABEL.value: relationship_label,
                Parameter.PROPERTIES.value: counter.to_dict(include_properties),
                Parameter.TYPE.value: Parameter.RELATIONSHIP.value,
            }
