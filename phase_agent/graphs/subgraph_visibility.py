"""Expose read-only child graph views without changing business checkpoint ownership.

The actual callback graphs remain stateless. Inspection views accept a saver
borrowed from the server so absent child history returns an empty list rather
than breaking the Studio history panel. No child is run by this registration.
"""


def expose_child(parent, node_name, child):
    if child is not None:
        # This is an introspection view, not the object the business callback runs.
        # A False checkpointer refuses the server's borrowed saver even for reads.
        view = child.copy(update={"checkpointer": None}) if child.checkpointer is False else child
        parent.nodes[node_name].subgraphs = [view]
    return parent
