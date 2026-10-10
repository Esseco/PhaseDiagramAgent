"""Documentation projections must stay tied to executable topology."""
from pathlib import Path
from phase_agent.graphs.graph_overview import render_overview, mermaid_overview


def test_overview_is_read_only_and_projects_actual_edges():
    from types import SimpleNamespace

    class Graph:
        def get_graph(self):
            return SimpleNamespace(nodes={"__start__": {}, "tool__example": {}, "__end__": {}},
                edges=[SimpleNamespace(source="__start__", target="tool__example", conditional=True),
                       SimpleNamespace(source="tool__example", target="__end__", conditional=False)])

        def invoke(self, *args, **kwargs):
            raise AssertionError("Overview must never execute a graph")

    text = mermaid_overview(Graph(), group_tools=True)
    assert "n0 -.-> n1" in text
    assert "n1 --> n2" in text
    assert "tool__example" not in text


def test_checked_in_overview_matches_production_topology():
    document = Path(__file__).resolve().parents[1] / "docs" / "architecture" / "flow.md"
    assert document.read_text(encoding="utf-8") == render_overview()
