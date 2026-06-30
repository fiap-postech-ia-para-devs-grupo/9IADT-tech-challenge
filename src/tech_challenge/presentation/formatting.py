from __future__ import annotations

import json
from typing import Any


def chat_answer_text(answer: Any) -> str:
    if isinstance(answer, dict):
        return str(answer.get("resposta") or answer.get("answer") or answer.get("response") or answer)

    text = str(answer).strip()
    if not text.startswith("{"):
        return text

    try:
        payload = json.loads(text)
    except ValueError:
        return text

    if isinstance(payload, dict):
        return str(payload.get("resposta") or payload.get("answer") or payload.get("response") or text)
    return text


def ag_result_rows(experiments: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "Modelo": baseline["name"],
            "F1": baseline["metrics"]["f1"],
            "Recall": baseline["metrics"].get("recall"),
            "Equity Gap": baseline["metrics"].get("equity_gap"),
            "Pop": "-",
            "Gen": "-",
            "Mut": "-",
        }
    ]
    for experiment in experiments:
        metrics = experiment["test_metrics"]
        rows.append(
            {
                "Modelo": experiment["name"],
                "F1": metrics["f1"],
                "Recall": metrics.get("recall"),
                "Equity Gap": metrics.get("equity_gap"),
                "Pop": str(experiment["population"]),
                "Gen": str(experiment["generations"]),
                "Mut": f"{experiment['mutation_rate']:.0%}",
            }
        )
    return rows
