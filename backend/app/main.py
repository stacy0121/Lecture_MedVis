from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import priority, weights as weights_router

app = FastAPI(
    title="욕창 예방 우선순위 산정 API",
    description="MIMIC-IV 기반 환자 체위 위험도 산정 엔진",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 프론트엔드 도메인으로 제한
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(priority.router, prefix="/api/v1/priority", tags=["priority"])
app.include_router(weights_router.router, prefix="/api/v1/priority", tags=["weights"])


@app.get("/api/v1/demo", tags=["demo"])
def get_demo():
    """patients.json 기반 데모 데이터 반환"""
    import json, os
    path = os.path.join(os.path.dirname(__file__), "..", "patients.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    from app.services.scorer import evaluate, sort_and_rank
    from app.services.scorer import DEFAULT_WEIGHTS
    results = sort_and_rank(data["patients"], DEFAULT_WEIGHTS)
    return {"summary": data["summary"], "patients": results}
