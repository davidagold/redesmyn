from __future__ import annotations


def slugify(value: str, *, fallback: str = "item") -> str:
    slug_chars: list[str] = []
    prev_dash = False
    for ch in value.strip().lower():
        if ch.isalnum():
            slug_chars.append(ch)
            prev_dash = False
            continue
        if ch in {" ", "-", "_"} and not prev_dash:
            slug_chars.append("-")
            prev_dash = True
    slug = "".join(slug_chars).strip("-")
    return slug or fallback
