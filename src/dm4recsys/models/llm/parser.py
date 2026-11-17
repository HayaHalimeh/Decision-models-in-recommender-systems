"""Parse a listwise LLM answer into a full permutation of option IDs.

Rules: items are read in the order they are written, named either by ID or by title. Bracketed IDs ("[7]") are preferred; if none appear, bare integers are
used. Titles are matched case-sensitively as whole words, as shown in the prompt or without a trailing "(year)";
title text is masked before IDs are searched, so numbers inside titles never count as IDs. A title right after an
ID ("[3] Halo 3") is the same item, not a second one; a repeated title (e.g. Yelp chains) takes the next candidate
with that title. The first occurrence of each item is kept, invalid IDs are dropped, and missing IDs are appended in
random order (fixed seed); for NDCG they count as one tied block (run/runner.py: ndcg_fields).
A response is *compliant* iff it names at least `min_ranked` distinct candidates (default 5 = the NDCG@5 cutoff, so
the top 5 come from the model, not from the random fill). *complete* = the stricter
rule: every valid ID exactly once and nothing invalid (kept per row in `info.complete`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...utils import rng_for

_BRACKET = re.compile(r"\[\s*(\d+)\s*\]")
_BARE = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
_YEAR = re.compile(r"\s*\(\d{4}\)\s*$")
_GLUE = re.compile(r"[\s:\-–—.)]*")  # what may stand between an ID and its own title ("[3] Halo 3", "[3]: Halo 3")


@dataclass
class ParseResult:
    order: list[str]
    compliant: bool
    complete: bool
    n_found_valid: int
    duplicates: int
    invalid: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    mode: str = "bracket"
    n_by_title: int = 0


def _clip(t: str, n: int | None) -> str:  # = prompts.render._clip (the title as shown in the prompt)
    return t if not n or len(t) <= n else t[: n - 1] + "…"


def _title_forms(titles: list[str], ids: list[str], max_chars: int | None) -> dict[str, list[str]]:
    """form -> candidate IDs with that form, in option order. Forms: the title as shown, without a trailing '…' if
    clipped, and without a trailing '(year)' (only if >= 3 characters remain, so e.g. 'M (1931)' needs its year)."""
    forms: dict[str, list[str]] = {}
    for i, t in zip(ids, titles):
        shown = _clip(t, max_chars)
        cands = {shown, shown.rstrip("…").rstrip(), t}
        cands |= {y for y in (_YEAR.sub("", c) for c in list(cands)) if len(y) >= 3}
        for f in cands:
            if f and i not in forms.setdefault(f, []):
                forms[f].append(i)
    return forms


def parse_ranking(text: str | None, valid_ids: list[str], *seed_parts, min_ranked: int = 5,
                  titles: list[str] | None = None, max_title_chars: int | None = None) -> ParseResult:
    text = text or ""
    hits: list[tuple[int, int, str, object]] = []  # (start, end, "id" | "title", ID string | candidate IDs)
    masked = text
    if titles:
        for form, fids in sorted(_title_forms(titles, valid_ids, max_title_chars).items(), key=lambda x: -len(x[0])):
            for m in re.finditer(r"(?<!\w)" + re.escape(form) + r"(?!\w)", masked):
                hits.append((m.start(), m.end(), "title", fids))
                masked = masked[:m.start()] + " " * (m.end() - m.start()) + masked[m.end():]
    id_hits, mode = list(_BRACKET.finditer(masked)), "bracket"
    if not id_hits:
        id_hits, mode = list(_BARE.finditer(masked)), "bare"
    hits += [(m.start(), m.end(), "id", str(int(m.group(1)))) for m in id_hits]  # "07" -> "7"
    hits.sort(key=lambda h: h[0])

    valid = set(valid_ids)
    seen: list[str] = []
    dup, invalid, n_title = 0, [], 0
    prev: tuple[int, int, str, object] | None = None
    for h in hits:
        start, end, kind, val = h
        if kind == "id":
            if val in valid:
                if val in seen:
                    dup += 1
                else:
                    seen.append(val)
            else:
                invalid.append(val)
        else:
            own_title = (prev is not None and prev[2] == "id" and prev[3] in val
                         and _GLUE.fullmatch(text[prev[1]:start]) is not None)
            nxt = next((i for i in val if i not in seen), None)
            if not own_title and nxt is not None:  # a title of an item already named is not a new item
                seen.append(nxt)
                n_title += 1
        prev = h
    missing = [v for v in valid_ids if v not in seen]
    rng = rng_for("llm-parse", *seed_parts)
    fill = [missing[i] for i in rng.permutation(len(missing))]
    complete = len(seen) == len(valid_ids) and dup == 0 and not invalid and n_title == 0
    compliant = len(seen) >= min(min_ranked, len(valid_ids))
    if not id_hits:
        mode = "title" if n_title else "none"
    elif n_title:
        mode += "+title"
    return ParseResult(seen + fill, compliant, complete, len(seen), dup, invalid, missing, mode, n_title)
