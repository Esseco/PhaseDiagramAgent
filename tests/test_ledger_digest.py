import hashlib
import json

import pytest

from phase_agent.persistence.ledger.phase_data_manager import _data_digest


@pytest.mark.parametrize(
    "data",
    [
        {},
        [],
        None,
        "Co钴",
        [1, None, {"b": -0.0, "a": [float("nan"), float("inf")]}],
        {"中文": [{"z": 3.14, "a": "Na"}], "a": {"nested": [1, 2, 3]}},
        {1: "numeric key", 2: "other"},
    ],
)
def test_digest_preserves_canonical_json_bytes(data):
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256("".join(encoder.iterencode(data)).encode("utf8")).hexdigest()
    assert _data_digest(data) == expected
