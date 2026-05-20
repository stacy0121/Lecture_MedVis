from fastapi import APIRouter
from app.schemas import CalculateRequest, CalculateResponse, WeightsInput, PatientInput
from app.services.scorer import evaluate, sort_and_rank, DEFAULT_WEIGHTS

router = APIRouter()


@router.post("/calculate", response_model=CalculateResponse)
def calculate_priority(body: CalculateRequest):
    """환자 목록 일괄 우선순위 산정"""
    w = body.weights.model_dump() if body.weights else DEFAULT_WEIGHTS
    patients_raw = [p.model_dump() for p in body.patients]
    # 필드명 통일 (snake_case → 내부 키)
    for p in patients_raw:
        p.setdefault("elapsedMinutes", p.pop("elapsed_min", 0))
        p.setdefault("bradenScore", p.pop("braden_score", 18))
        p.setdefault("mobilityScore", p.pop("mobility_score", 4))
        p.setdefault("nutritionScore", p.pop("nutrition_score", 4))
        p.setdefault("currentPosition", p.pop("current_posture", "Supine"))
        p.setdefault("id", p.pop("patient_id", "unknown"))
    results = sort_and_rank(patients_raw, w)
    return CalculateResponse(
        patients=results,
        weights_used=WeightsInput(**w),
    )


@router.post("/single")
def calculate_single(patient: PatientInput, weights: WeightsInput = None):
    """단일 환자 점수 조회"""
    w = weights.model_dump() if weights else DEFAULT_WEIGHTS
    raw = patient.model_dump()
    raw["elapsedMinutes"] = raw.pop("elapsed_min")
    raw["bradenScore"]    = raw.pop("braden_score")
    raw["mobilityScore"]  = raw.pop("mobility_score")
    raw["nutritionScore"] = raw.pop("nutrition_score")
    raw["currentPosition"]= raw.pop("current_posture")
    raw["id"]             = raw.pop("patient_id")
    return evaluate(raw, w)


@router.post("/simulate")
def simulate(body: CalculateRequest):
    """커스텀 가중치로 시뮬레이션 (재산정용)"""
    return calculate_priority(body)
