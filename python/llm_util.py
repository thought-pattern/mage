"""Utilities for llm util."""

from mgp import Any as mgp_Any
from mgp import Edge as mgp_Edge
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
        # Membership index for all_relationships_list, which keeps discovery order for the external records.
        self.all_relationship_keys = set()

        self.generate_schema(context)

    @property
    def type(self):
        return self.internal_type

    @property
    def node_counter(self):
        return self.internal_node_counter

    def get_schema(self) -> object:
        if self.internal_type == OutputType.RAW.value:
            raw_schema = {
                Parameter.NODE_PROPS.value: self.all_node_properties_dict,
                Parameter.REL_PROPS.value: self.all_relationship_properties_dict,
                Parameter.RELATIONSHIPS.value: self.all_relationships_list,
            }
            return raw_schema
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
        relationship_type = relationship.type.name
        for start_label in start_labels:
            for target_label in target_labels:
                relationship_key = (start_label, relationship_type, target_label)
                if relationship_key in self.all_relationship_keys:
                    continue
                self.all_relationship_keys.add(relationship_key)
                self.all_relationships_list.append(
                    {
                        Parameter.START.value: start_label,
                        Parameter.TYPE.value: relationship_type,
                        Parameter.END.value: target_label,
                    }
                )
        return False

    def get_prompt_ready_schema(self) -> str:
        prompt_ready_schema = "Node properties are the following:\n"
        for label in self.all_node_properties_dict.keys():
            prompt_ready_schema += "Node name: '{label}', Node properties: {properties}\n".format(
                label=label,
                properties=sorted(
                    self.all_node_properties_dict.get(label, []),
                    key=lambda prop: prop.get(Parameter.PROPERTY.value, ""),
                ),
            )

        prompt_ready_schema += "\nRelationship properties are the following:\n"
        for rel in self.all_relationship_properties_dict.keys():
            prompt_ready_schema += "Relationship name: '{name}', Relationship properties: {properties}\n".format(
                name=rel,
                properties=sorted(
                    self.all_relationship_properties_dict.get(rel, []),
                    key=lambda prop: prop.get(Parameter.PROPERTY.value, ""),
                ),
            )

        prompt_ready_schema += "\nThe relationships are the following:\n"

        for relationship in self.all_relationships_list:
            start = relationship.get(Parameter.START.value, "")
            relationship_type = relationship.get(Parameter.TYPE.value, "")
            end = relationship.get(Parameter.END.value, "")
            prompt_ready_schema += f"['(:{start})-[:{relationship_type}]->(:{end})']\n"

        return prompt_ready_schema

    def update_properties_dict(
        self,
        graph_object: mgp_Any,
        all_properties_dict: dict[str, mgp_Any],
        key: str,
    ):
        for property_name, property_value in graph_object.properties.items():
            property_record = {
                Parameter.PROPERTY.value: property_name,
                Parameter.TYPE.value: type(property_value).__name__,
            }
            if not all_properties_dict.get(key, []):
                all_properties_dict[key] = [property_record]
                continue

            if property_name in [d.get(Parameter.PROPERTY.value, "") for d in all_properties_dict.get(key, [])]:
                continue

            all_properties_dict.get(key, []).append(property_record)
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
