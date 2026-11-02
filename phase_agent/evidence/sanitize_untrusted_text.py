"""Bound remote text and label it as data, never instructions."""


def untrusted_text(value, *, limit=500):
    if value is None:
        return None
    text = "".join(
        character if character.isprintable() or character in "\n\t" else "?"
        for character in str(value)
    )[:limit]
    return {"trust": "untrusted_remote_data", "text": text, "truncated": len(str(value)) > limit}
