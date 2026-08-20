"""판정 API.

신청서 1건을 받아 score와 decision을 반환하고 예측 로그 1줄을 남긴다.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import model_loader
import predict_log


class PredictRequest(BaseModel):
    application_id: str | None = None
    # 피처는 학습 시점 목록을 모델에서 읽어 검증하므로 여기서 열거하지 않는다.
    features: dict = Field(..., description="신청서 1건의 피처 딕셔너리")


class PredictResponse(BaseModel):
    application_id: str
    score: float
    decision: str
    model_version: str
    threshold: float


def create_app() -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["loaded"] = model_loader.load()
        yield

    app = FastAPI(title="BAF fraud decision API", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        loaded = state.get("loaded")
        if loaded is None:
            raise HTTPException(status_code=503, detail="모델 미로드")
        return {
            "status": "ok",
            "model_version": loaded.version,
            "threshold": loaded.threshold,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        loaded = state.get("loaded")
        if loaded is None:
            raise HTTPException(status_code=503, detail="모델 미로드")

        try:
            df = model_loader.prepare(req.features, loaded)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

        score = model_loader.score(loaded, df)
        decision = "ALERT" if score >= loaded.threshold else "PASS"
        application_id = req.application_id or str(uuid.uuid4())

        predict_log.write(
            application_id=application_id,
            score=score,
            decision=decision,
            model_version=loaded.version,
            threshold=loaded.threshold,
        )

        return PredictResponse(
            application_id=application_id,
            score=score,
            decision=decision,
            model_version=loaded.version,
            threshold=loaded.threshold,
        )

    return app


app = create_app()
