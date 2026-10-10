"""Expose actual stateless child graphs without changing checkpoint ownership.

LangGraph omits checkpointer=False children from automatic discovery. The
parent already invokes these exact objects; registering them exposes topology
and nested traces, and never creates another execution path.
"""


def expose_child(parent, node_name, child):
    if child is not None:
        parent.nodes[node_name].subgraphs = [child]
    return parent
