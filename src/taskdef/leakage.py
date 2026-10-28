"""Leakage rules between H and G (TASKDEFINITION_SPEC.md §5.3, §9)."""

from __future__ import annotations


def canonical(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return " ".join(str(value).split()).casefold()


def value_leaks(hidden, given, min_substring_len: int) -> bool:
    """True if the given value reveals the hidden one.

    Equal after canonicalization, or, for strings, the hidden value is
    contained in the given one (only when it has at least ``min_substring_len``
    characters, so that short values like "2" don't match everything).
    """
    h, g = canonical(hidden), canonical(given)
    if h == g:
        return True
    return isinstance(hidden, str) and isinstance(given, str) and len(h) >= min_substring_len and h in g


class LeakagePairs:
    """Groups of prop_keys that reveal each other (``properties.leakage_pairs``)."""

    def __init__(self, groups: list[list[str]]):
        self._related: dict[str, set[str]] = {}
        for group in groups:
            members = set(group)
            for key in group:
                self._related.setdefault(key, set()).update(members)
        # Merge overlapping groups transitively.
        changed = True
        while changed:
            changed = False
            for key, members in self._related.items():
                merged = set().union(*(self._related.get(m, {m}) for m in members))
                if merged != members:
                    self._related[key] = merged
                    changed = True

    def related(self, key: str) -> set[str]:
        return self._related.get(key, {key})

    def blocked_by(self, keys) -> set[str]:
        blocked: set[str] = set()
        for key in keys:
            blocked |= self.related(key)
        return blocked
