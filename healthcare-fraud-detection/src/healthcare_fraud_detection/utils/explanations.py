from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from healthcare_fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You write brief, professional summaries for healthcare operations and \
special investigations teams. Use plain business language—no statistics jargon, no mention \
of "models," "coefficients," or "algorithms." Do not state that fraud is proven; describe \
elevated risk or routine alignment. At most four short sentences."""


def _json_safe(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return str(value)[:19]
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            return str(value)
    return str(value)


def _claim_to_mapping(claim_data: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    if isinstance(claim_data, pd.Series):
        raw = claim_data.to_dict()
    else:
        raw = dict(claim_data)
    return {str(k): _json_safe(v) for k, v in raw.items()}


def _format_important_features(
    important_features: Sequence[tuple[str, float] | Mapping[str, Any]],
) -> str:
    lines: list[str] = []
    for item in important_features:
        if isinstance(item, Mapping):
            name = str(item.get("feature", item.get("name", "")))
            w = item.get("weight", item.get("value", item.get("importance")))
            lines.append(f"- {name}: {w}")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            name, w = str(item[0]), item[1]
            if isinstance(w, (float, np.floating)):
                lines.append(f"- {name}: {float(w):.4f}")
            else:
                lines.append(f"- {name}: {w}")
        else:
            lines.append(f"- {item!s}")
    return "\n".join(lines) if lines else "(no feature list provided)"


def generate_fraud_explanation(
    fraud_score: float,
    claim_data: Mapping[str, Any] | pd.Series,
    important_features: Sequence[tuple[str, float] | Mapping[str, Any]],
    *,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
) -> str:
    """
    Call the OpenAI API to produce a short, business-friendly narrative for a claim.

    Parameters
    ----------
    fraud_score :
        Score in ``[0, 1]`` where higher values indicate higher model-estimated risk.
    claim_data :
        Row-level fields (e.g. patient, provider, amounts, dates) as a mapping or ``Series``.
    important_features :
        Ranked signals such as ``[(\"claim_amount_zscore\", 2.1), ...]`` or
        ``[{\"feature\": \"...\", \"weight\": 0.12}, ...]``.
    model :
        Chat model id (default ``gpt-4o-mini``).
    api_key :
        OpenAI API key. If omitted, uses env ``OPENAI_API_KEY``.
    timeout_seconds :
        HTTP timeout for the API call.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required. Install with: pip install openai"
        ) from exc

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY or pass api_key= to generate_fraud_explanation().")

    claim_json = json.dumps(_claim_to_mapping(claim_data), indent=2, default=str)
    feature_text = _format_important_features(important_features)
    pct = round(float(fraud_score) * 100, 1)

    user_prompt = f"""A scoring tool assigned this claim an estimated risk score of {pct}% \
(where 100% is the highest risk the tool shows for a claim like this).

Claim details (structured):
{claim_json}

Factors the tool weighted most heavily (for context only):
{feature_text}

Write a concise explanation for a business reader: is this claim relatively aligned with \
normal billing patterns, or does it show notable risk signals worth a closer look—and why?"""

    client = OpenAI(api_key=key, timeout=timeout_seconds)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=400,
            temperature=0.4,
        )
    except Exception:
        logger.exception("OpenAI explanation request failed")
        raise

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned an empty explanation.")
    return content.strip()
