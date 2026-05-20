"""
위험 점수 계산 로직 — 프론트엔드 JS evaluate() 함수와 동일한 로직
"""

from typing import Any

DEFAULT_WEIGHTS = {
    "overtime": 35,
    "braden": 30,
    "moisture": 15,
    "mobility": 10,
    "nutrition": 10,
}

MOISTURE_SCORES = {"dry": 0, "occasionally": 33, "often": 66, "constantly": 100}
ROTATION = {"Left": "Right", "Right": "Supine", "Supine": "Left"}
POSTURE_LABELS = {"Left": "좌측위", "Right": "우측위", "Supine": "앙와위"}


def overtime_score(elapsed: float, interval: int = 120) -> float:
    if elapsed <= interval:
        return (elapsed / interval) * 40
    return min(40 + ((elapsed - interval) / 30) * 10, 100)


def braden_to_risk(braden: int) -> float:
    """Braden 합계(6~23) → 위험 점수(0~100). 낮을수록 위험."""
    return round((23 - braden) / 17 * 100, 1)


def mobility_to_risk(score: int) -> float:
    """Braden 이동성(1~4) → 위험 점수(0~100)."""
    return round((4 - score) / 3 * 100, 1)


def nutrition_to_risk(score: int) -> float:
    """Braden 영양(1~4) → 위험 점수(0~100)."""
    return round((4 - score) / 3 * 100, 1)


def evaluate(patient: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    elapsed = patient.get("elapsedMinutes") or patient.get("elapsed_min", 0)
    interval = patient.get("turn_interval_min", 120)
    braden = patient.get("bradenScore") or patient.get("braden_score", 18)
    moisture = patient.get("moisture", "dry")
    mob = patient.get("mobilityScore") or patient.get("mobility_score", 4)
    nut = patient.get("nutritionScore") or patient.get("nutrition_score", 4)
    posture = patient.get("currentPosition") or patient.get("current_posture", "Supine")

    remaining = interval - elapsed
    is_overtime = elapsed > interval

    o_score   = overtime_score(elapsed, interval)
    b_score   = braden_to_risk(braden)
    m_score   = MOISTURE_SCORES.get(moisture, 0)
    mob_score = mobility_to_risk(mob)
    n_score   = nutrition_to_risk(nut)

    total = (
        o_score   * weights["overtime"]  +
        b_score   * weights["braden"]    +
        m_score   * weights["moisture"]  +
        mob_score * weights["mobility"]  +
        n_score   * weights["nutrition"]
    ) / 100

    alert = (
        "immediate" if (total >= 55 or is_overtime) else
        "caution"   if total >= 25 else
        "stable"
    )

    next_posture = ROTATION.get(posture, "Left")
    next_label   = POSTURE_LABELS.get(next_posture, next_posture)
    action = (
        f"{next_label}으로 즉시 변경" if alert == "immediate" else
        f"{next_label}로 변경 준비"   if alert == "caution"   else
        "상태 재평가"
    )

    pid = patient.get("id") or patient.get("patient_id", "unknown")

    return {
        "patient_id":      pid,
        "name":            patient.get("name"),
        "room":            patient.get("room", ""),
        "age":             patient.get("age"),
        "gender":          patient.get("gender"),
        "diagnosis":       patient.get("diagnosis"),
        "current_posture": posture,
        "next_posture":    next_posture,
        "elapsed_min":     round(elapsed, 1),
        "remaining_min":   round(remaining, 1),
        "is_overtime":     is_overtime,
        "braden_score":    braden,
        "total_score":     round(total, 1),
        "alert":           alert,
        "action":          action,
        "breakdown": {
            "overtime":  round(o_score, 1),
            "braden":    round(b_score, 1),
            "moisture":  round(m_score, 1),
            "mobility":  round(mob_score, 1),
            "nutrition": round(n_score, 1),
        },
    }


def sort_and_rank(patients: list[dict], weights: dict) -> list[dict]:
    results = [evaluate(p, weights) for p in patients]
    results.sort(key=lambda x: x["total_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results
