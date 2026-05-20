from fastapi import APIRouter
from app.schemas import WeightsInput
from app.services import scorer

router = APIRouter()


@router.get("/weights")
def get_weights():
    """현재 기본 가중치 조회"""
    return scorer.DEFAULT_WEIGHTS


@router.post("/weights")
def update_weights(body: WeightsInput):
    """기본 가중치 업데이트 (런타임 적용)"""
    scorer.DEFAULT_WEIGHTS.update(body.model_dump())
    return {"message": "가중치가 업데이트되었습니다.", "weights": scorer.DEFAULT_WEIGHTS}
