from pydantic import BaseModel, Field
from typing import Literal, Optional


class PatientInput(BaseModel):
    patient_id: str
    room: str
    current_posture: Literal["Left", "Right", "Supine"]
    elapsed_min: float = Field(..., ge=0, description="현재 체위 유지 시간 (분)")
    braden_score: int = Field(..., ge=6, le=23, description="Braden Scale 합계 점수")
    moisture: Literal["dry", "occasionally", "often", "constantly"] = "dry"
    mobility_score: int = Field(..., ge=1, le=4, description="Braden 이동성 항목 점수 (1=완전제한 ~ 4=제한없음)")
    nutrition_score: int = Field(..., ge=1, le=4, description="Braden 영양 항목 점수 (1=매우불량 ~ 4=우수)")
    turn_interval_min: int = Field(120, description="체위 변경 목표 간격 (분)")
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    diagnosis: Optional[str] = None


class WeightsInput(BaseModel):
    overtime: float = Field(35, ge=0, le=100)
    braden: float = Field(30, ge=0, le=100)
    moisture: float = Field(15, ge=0, le=100)
    mobility: float = Field(10, ge=0, le=100)
    nutrition: float = Field(10, ge=0, le=100)


class BreakdownOutput(BaseModel):
    overtime: float
    braden: float
    moisture: float
    mobility: float
    nutrition: float


class PatientResult(BaseModel):
    patient_id: str
    rank: int
    name: Optional[str]
    room: str
    age: Optional[int]
    gender: Optional[str]
    diagnosis: Optional[str]
    current_posture: str
    next_posture: str
    elapsed_min: float
    remaining_min: float
    is_overtime: bool
    braden_score: int
    total_score: float
    alert: Literal["immediate", "caution", "stable"]
    action: str
    breakdown: BreakdownOutput


class CalculateRequest(BaseModel):
    patients: list[PatientInput]
    weights: Optional[WeightsInput] = None


class CalculateResponse(BaseModel):
    patients: list[PatientResult]
    weights_used: WeightsInput
