"""
app/core/process_summary.py
─────────────────────────────────────────────────────────────────────────────
Aggregates per-step anomalies / errors / actionable-suggestion counts into a
single `summary` object that is attached at the root of a process payload by
both `GET /api/processes` and `GET /api/processes/<id>`.

WHY THIS LIVES IN ITS OWN MODULE
    The list endpoint and the detail endpoint must produce *identical* summary
    semantics. Keeping the logic here means there is exactly one place that
    decides "what counts as an anomaly / error / suggestion", so the two
    endpoints can never drift apart.

FIELD MAPPING (important — read before changing)
    The current data model does NOT yet store explicit `anomalies` / `errors`
    arrays on a step. To stay faithful to today's schema *and* be ready for
    tomorrow's, each metric scans an ordered list of candidate field names and
    uses the FIRST one that is present on the step. That means:

      • The instant the analysis pipeline starts writing a real `anomalies`
        (or `errors`) array onto steps, these counts populate automatically —
        no change required here.
      • Until then, `anomalies` falls back to `pain_points`, which is the only
        anomaly-like signal the step currently carries. If you do NOT want
        pain_points treated as anomalies, remove it from ANOMALY_FIELDS below.

    Suggestions are counted from the dedicated `automation_suggestions`
    collection (passed in as `suggestions`), not from a step field, because
    that is where they actually live.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

# ── Field mapping ────────────────────────────────────────────────────────────
# Ordered by priority. The first field present on a step (even if empty) wins,
# so we never double-count (e.g. a real `anomalies` array is NOT added on top
# of `pain_points`).
ANOMALY_FIELDS: Sequence[str] = ("anomalies", "anomaly_list", "pain_points")
ERROR_FIELDS:   Sequence[str] = ("errors", "error_list")

# Step-level scalar error signals (counted as 1 error each when truthy).
STEP_ERROR_FLAG_FIELDS: Sequence[str] = ("error",)
STEP_ERROR_STATUS_VALUES = {"error", "failed"}


def _count_field(step: Dict[str, Any], candidates: Sequence[str]) -> int:
    """
    Return the count contributed by the FIRST present candidate field.

    Counting rules (so the helper tolerates whatever shape the data takes):
      • list / tuple / set  -> number of elements
      • int / float         -> the numeric value (cast to int, clamped >= 0)
      • truthy scalar       -> 1
      • missing / falsy      -> 0
    """
    for name in candidates:
        if name not in step:
            continue
        value = step.get(name)
        if value is None:
            return 0
        if isinstance(value, (list, tuple, set)):
            return len(value)
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return max(0, int(value))
        # Any other truthy scalar (e.g. a non-empty string) counts as one.
        return 1 if value else 0
    return 0


def _step_error_count(step: Dict[str, Any]) -> int:
    """Errors contributed by a single step: explicit error arrays + flags."""
    count = _count_field(step, ERROR_FIELDS)
    for flag in STEP_ERROR_FLAG_FIELDS:
        if step.get(flag):
            count += 1
    if str(step.get("status", "")).lower() in STEP_ERROR_STATUS_VALUES:
        count += 1
    return count


def _build_short_text(anomalies: int, errors: int, suggestions: int) -> str:
    """
    Human-readable one-liner, e.g.
        "Found 2 anomalies and 5 actionable suggestions in this document."
    Only non-zero metrics are mentioned; falls back to a clean message when
    everything is zero.
    """
    def plural(n: int, word: str) -> str:
        return f"{n} {word}" if n == 1 else f"{n} {word}s"

    parts: List[str] = []
    if anomalies:
        parts.append(plural(anomalies, "anomaly").replace("anomalys", "anomalies"))
    if errors:
        parts.append(plural(errors, "error"))
    if suggestions:
        parts.append(plural(suggestions, "actionable suggestion"))

    if not parts:
        return "No anomalies, errors, or actionable suggestions were found in this document."

    if len(parts) == 1:
        body = parts[0]
    elif len(parts) == 2:
        body = f"{parts[0]} and {parts[1]}"
    else:
        body = f"{', '.join(parts[:-1])}, and {parts[-1]}"

    return f"Found {body} in this document."


def build_summary(
    steps: Iterable[Dict[str, Any]] | None,
    suggestions: Iterable[Dict[str, Any]] | None = None,
    *,
    suggestion_count: int | None = None,
) -> Dict[str, Any]:
    """
    Aggregate counts across a process's steps.

    Args:
        steps:            iterable of step documents (each a dict).
        suggestions:      iterable of automation-suggestion documents. Used when
                          an explicit `suggestion_count` is not supplied.
        suggestion_count: pre-computed suggestion count (e.g. from an AQL
                          subquery) — avoids materialising suggestion docs in
                          the list endpoint. Takes precedence over `suggestions`.

    Returns:
        {
          "total_anomalies":   int,
          "total_errors":      int,
          "total_suggestions": int,
          "short_text":        str,
        }
    """
    step_list = list(steps or [])

    total_anomalies = sum(_count_field(s, ANOMALY_FIELDS) for s in step_list)
    total_errors = sum(_step_error_count(s) for s in step_list)

    if suggestion_count is not None:
        total_suggestions = max(0, int(suggestion_count))
    else:
        total_suggestions = len(list(suggestions or []))

    return {
        "total_anomalies":   total_anomalies,
        "total_errors":      total_errors,
        "total_suggestions": total_suggestions,
        "short_text":        _build_short_text(
            total_anomalies, total_errors, total_suggestions
        ),
    }
