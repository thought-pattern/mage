"""Utilities for meta util."""

from mgp import List as mgp_List, Map as mgp_Map, ProcCtx as mgp_ProcCtx, Record as mgp_Record, read_proc as mgp_read_proc

from mage.meta_util.parameters import Parameter


@mgp_read_proc
def schema(
    context: mgp_ProcCtx, include_properties: bool = False
) -> mgp_Record(nodes=mgp_List[mgp_Map], relationships=mgp_List[mgp_Map]):
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

    # Each key's counts dict is also its schema "properties" value: {count} or, with properties, {count, properties_count}.
    node_count_by_labels: dict[tuple, dict] = {}
    relationship_count_by_labels: dict[tuple, dict] = {}

    node_counter = 0

    for node in context.graph.vertices:
        node_counter += 1
        labels = tuple(sorted(label.name for label in node.labels))
        update_counts(
            node_count_by_labels,
            key=labels,
            obj=node,
            include_properties=include_properties,
        )

        for relationship in node.out_edges:
            target_labels = tuple(sorted(label.name for label in relationship.to_vertex.labels))
            key = (labels, relationship.type.name, target_labels)
            update_counts(
                relationship_count_by_labels,
                key=key,
                obj=relationship,
                include_properties=include_properties,
            )

    if node_counter == 0:
        raise Exception("Can't generate a graph schema since there is no data in the database.")

    node_index_by_labels = {key: i for i, key in enumerate(node_count_by_labels.keys())}
    nodes = [
        {
            Parameter.ID.value: node_index,
            Parameter.LABELS.value: labels,
            Parameter.PROPERTIES.value: counts,
            Parameter.TYPE.value: Parameter.NODE.value,
        }
        for node_index, (labels, counts) in enumerate(node_count_by_labels.items())
    ]

    # Relationship IDs number every grouped relationship; one whose endpoint label set has no node entry is omitted
    # without renumbering the rest.
    relationships = []
    for relationship_index, ((source_label, relationship_label, target_label), counts) in enumerate(
        relationship_count_by_labels.items()
    ):
        if source_label not in node_index_by_labels or target_label not in node_index_by_labels:
            continue
        relationships.append(
            {
                Parameter.ID.value: relationship_index,
                Parameter.START.value: node_index_by_labels.get(source_label, 0),
                Parameter.END.value: node_index_by_labels.get(target_label, 0),
                Parameter.LABEL.value: relationship_label,
                Parameter.PROPERTIES.value: counts,
                Parameter.TYPE.value: Parameter.RELATIONSHIP.value,
            }
        )

    computed_return_value = mgp_Record(nodes=nodes, relationships=relationships)
    return computed_return_value


def update_counts(
    obj_count_by_key: dict[tuple, dict],
    key: tuple,
    obj,
    include_properties: bool = False,
) -> bool:
    """Counts one node or relationship under its key and, with properties, each of its property names."""
    if key not in obj_count_by_key:
        obj_count_by_key[key] = (
            {Parameter.COUNT.value: 0, Parameter.PROPERTIES_COUNT.value: {}} if include_properties else {Parameter.COUNT.value: 0}
        )

    counts = obj_count_by_key.get(key, {})
    counts[Parameter.COUNT.value] = counts.get(Parameter.COUNT.value, 0) + 1

    if include_properties:
        property_counts = counts.get(Parameter.PROPERTIES_COUNT.value, {})
        for property_name in obj.properties.keys():
            property_counts[property_name] = property_counts.get(property_name, 0) + 1
    return False
