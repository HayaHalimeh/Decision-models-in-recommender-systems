"""Text rendering shared by all text-based models."""
from __future__ import annotations


def _clip(t: str, n: int | None) -> str:
    return t if not n or len(t) <= n else t[: n - 1] + "…"


def render_history_lines(titles: list[str], max_chars: int | None = None) -> str:
    if not titles:
        return "(no history)"
    return "\n".join(f"{i + 1}. {_clip(t, max_chars)}" for i, t in enumerate(titles))


def render_candidate_lines(titles: list[str], ids: list[str], max_chars: int | None = None) -> str:
    return "\n".join(f"[{i}] {_clip(t, max_chars)}" for i, t in zip(ids, titles))


DECISION_QUESTION = "Question: Which of the options is the user most likely to interact with next?"
QUESTION_PREFIX = "Question: "


def decision_question(n: int) -> str:
    """The one instruction every text-based model gets: the question + the rank
    sentence. n = number of options."""
    return f"{DECISION_QUESTION} Rank all {n} options from most to least likely."


def pointwise_question(noun_singular: str) -> str:
    """The shared question without any reference to the options, for models that score one candidate at a time and
    never see the candidate list (EmbSim, CrossEnc)."""
    return f"Which {noun_singular} is the user most likely to interact with next?"


def decision_context(history: list[str], noun: str, n_options: int, max_chars: int | None = None,
                     include_history: bool = True) -> str:
    """Context (state + instruction) for decision models; options are passed separately.

    include_history=False (RQ2 'no history'): the history block (its lead-in sentence and the list) is removed;
    the instruction is unchanged. This is the only wording change."""
    q = decision_question(n_options)
    if not include_history:
        return q
    return (
        f"A user has interacted with the following {noun}, in chronological order from oldest to newest:\n"
        f"{render_history_lines(history, max_chars)}\n\n"
        f"{q}"
    )


def options_block(cands: list[str], ids: list[str], max_chars: int | None = None) -> str:
    """'Options:' + one '[ID] title' line per candidate (LLM prompt and local decision models)."""
    return "Options:\n" + render_candidate_lines(cands, ids, max_chars)


def output_format_instruction(n: int) -> str:
    """The only text an LLM gets beyond the decision models: how to write the ranking."""
    return f"Output all {n} IDs exactly once, in the format [ID], [ID], ... Output only the IDs, without explanations."


def listwise_prompt(history: list[str], cands: list[str], ids: list[str], noun: str,
                    max_chars: int | None = None, include_history: bool = True) -> str:
    """The single listwise prompt shared by every LLM ranker: exactly the decision-model text (decision_context:
    history block + the same instruction), then the options as '[ID] title' (as the local decision models see
    them), then the output format."""
    return (f"{decision_context(history, noun, len(ids), max_chars, include_history)}\n\n"
            f"{options_block(cands, ids, max_chars)}\n\n"
            f"{output_format_instruction(len(ids))}")

