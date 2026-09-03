"""Utilities for llm util."""

from mgp import Any as mgp_Any
from mgp import Edge as mgp_Edge
from mgp import Map as mgp_Map
from mgp import ProcCtx as mgp_ProcCtx
from mgp import Record as mgp_Record
from mgp import read_proc as mgp_read_proc

from mage.llm_util.parameters import OutputType, Parameter


class SchemaGenerator(object):
    def __init__(self, context, output_type):
        self.internal_type = output_type.lower()
        self.internal_node_counter = 0
        self.all_node_properties_dict = {}
        self.all_relationship_properties_dict = {}
        self.all_relationships_list = []

        self.generate_schema(context)

    @property
    def type(self):
        return self.internal_type

    @property
    def node_counter(self):
        return self.internal_node_counter

    def get_schema(self) -> object:
        if self.internal_type == OutputType.RAW.value:
            computed_return_value = self.get_raw_schema()
            return computed_return_value
        elif self.internal_type == OutputType.PROMPT_READY.value:
            computed_return_value = self.get_prompt_ready_schema()
            return computed_return_value
        else:
            raise Exception(
                f"Can't generate a graph schema since the provided output_type is not correct. Please choose "
                f"{OutputType.RAW.value} or {OutputType.PROMPT_READY.value}."
            )

    def generate_schema(self, context: mgp_ProcCtx):
        for node in context.graph.vertices:
            self.internal_node_counter += 1

            labels = tuple(sorted(label.name for label in node.labels))

            for label in labels:
                self.update_properties_dict(node, self.all_node_properties_dict, label)

            for relationship in node.out_edges:
                target_labels = tuple(sorted(label.name for label in relationship.to_vertex.labels))

                self.update_all_relationships_list(labels, relationship, target_labels)

                self.update_properties_dict(
                    relationship,
                    self.all_relationship_properties_dict,
                    relationship.type.name,
                )
        return False

    def update_all_relationships_list(
        self,
        start_labels: tuple[str],
        relationship: mgp_Edge,
        target_labels: tuple[str],
    ):
        for start_label in start_labels:
            for target_label in target_labels:
                full_relationship = {
                    Parameter.START.value: start_label,
                    Parameter.TYPE.value: relationship.type.name,
                    Parameter.END.value: target_label,
                }
                if full_relationship not in self.all_relationships_list:
                    self.all_relationships_list.append(full_relationship)
        return False

    def get_raw_schema(self) -> mgp_Map:
        return {
            Parameter.NODE_PROPS.value: self.all_node_properties_dict,
            Parameter.REL_PROPS.value: self.all_relationship_properties_dict,
            Parameter.RELATIONSHIPS.value: self.all_relationships_list,
        }

    def get_prompt_ready_schema(self) -> str:
        prompt_ready_schema = "Node properties are the following:\n"
        for label in self.all_node_properties_dict.keys():
            prompt_ready_schema += "Node name: '{label}', Node properties: {properties}\n".format(
                label=label,
                properties=sorted(
                    self.all_node_properties_dict.get(label, []),
                    key=lambda prop: prop[Parameter.PROPERTY.value],
                ),
            )

        prompt_ready_schema += "\nRelationship properties are the following:\n"
        for rel in self.all_relationship_properties_dict.keys():
            prompt_ready_schema += "Relationship name: '{name}', Relationship properties: {properties}\n".format(
                name=rel,
                properties=sorted(
                    self.all_relationship_properties_dict.get(rel, []),
                    key=lambda prop: prop[Parameter.PROPERTY.value],
                ),
            )

        prompt_ready_schema += "\nThe relationships are the following:\n"

        for relationship in self.all_relationships_list:
            prompt_ready_schema += (
                f"['(:{relationship[Parameter.START.value]})-[:{relationship[Parameter.TYPE.value]}]->(:"
                f"{relationship[Parameter.END.value]})']\n"
            )

        return prompt_ready_schema

    def update_properties_dict(
        self,
        graph_object: mgp_Any,
        all_properties_dict: dict[str, mgp_Any],
        key: str,
    ):
        for property_name in graph_object.properties.keys():
            if not all_properties_dict.get(key, False):
                all_properties_dict[key] = [
                    {
                        Parameter.PROPERTY.value: property_name,
                        Parameter.TYPE.value: type(graph_object.properties.get(property_name, False)).__name__,
                    }
                ]
                continue

            if property_name in [d.get(Parameter.PROPERTY.value, False) for d in all_properties_dict.get(key, False)]:
                continue

            all_properties_dict.get(key, []).append(
                {
                    Parameter.PROPERTY.value: property_name,
                    Parameter.TYPE.value: type(graph_object.properties.get(property_name, False)).__name__,
                }
            )
        return False


@mgp_read_proc
def schema(
    context: mgp_ProcCtx,
    output_type: str = OutputType.PROMPT_READY.value,
) -> mgp_Record:
    (
        "\n    Procedure to generate the graph database schema in a prompt-ready or raw format.\n\n    A"  # Continue literal.
        "rgs:\n        context (mgp.ProcCtx): Reference to the context execution.\n        output_type "  # Continue literal.
        "(str): By default (set to 'prompt_ready'), the graph schema will include additional context "  # Continue literal.
        "and it will be prompt-ready. If set to 'raw', it will produce a simpler version that can be "  # Continue literal.
        "adjusted for the prompt.\n\n    Returns:\n        schema (mgp.Any): `str` containing prompt-rea"  # Continue literal.
        "dy graph schema description in a format suitable for large language models (LLMs), or `mgp.L"  # Continue literal.
        "ist` containing information on graph schema in raw format which can customized for LLMs.\n\n  "  # Continue literal.
        "  Example:\n        Get prompt-ready graph schema:\n            `CALL llm_util.schema() YIELD "  # Continue literal.
        "schema RETURN schema;`\n            or\n            `CALL llm_util.schema('prompt_ready') YIEL"  # Continue literal.
        "D schema RETURN schema;`\n        Get raw graph schema:\n            `CALL llm_util.schema('ra"  # Continue literal.
        "w') YIELD schema RETURN schema;`\n"
    )

    schema_generator = SchemaGenerator(context, output_type)

    if schema_generator.node_counter == 0:
        raise Exception("Can't generate a graph schema since there is no data in the database.")

    computed_return_value = mgp_Record(schema=schema_generator.get_schema())
    return computed_return_value
