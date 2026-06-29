from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tech_challenge.diagnosis import diagnose_patient
from tech_challenge.experiments import load_ag_results
from tech_challenge.explanation import explain_diagnosis

app = FastAPI(title="Tech Challenge Fase 2 API", version="0.1.0")


class FeatureImpact(BaseModel):
    feature: str
    value: float
    impact: float


class DiagnoseRequest(BaseModel):
    patient_index: int


class DiagnoseResponse(BaseModel):
    prediction: str
    confidence: float
    top_features: list[FeatureImpact]


class ExplainRequest(BaseModel):
    prediction: str
    confidence: float
    top_features: list[FeatureImpact]


class ExplainResponse(BaseModel):
    explanation: str
    disclaimer: str


class AGExperiment(BaseModel):
    name: str
    population: int
    generations: int
    mutation_rate: float
    best_f1: float
    convergence: list[float]


class BestConfig(BaseModel):
    experiment: str
    n_estimators: int
    max_depth: int
    min_samples_split: int
    max_features: str


class AGResultsResponse(BaseModel):
    experiments: list[AGExperiment]
    baseline_f1: float
    best_config: BestConfig


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest) -> DiagnoseResponse:
    try:
        result = diagnose_patient(req.patient_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DiagnoseResponse(
        prediction=result.prediction,
        confidence=result.confidence,
        top_features=[FeatureImpact(**asdict(feature)) for feature in result.top_features],
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> ExplainResponse:
    result = explain_diagnosis(
        prediction=req.prediction,
        confidence=req.confidence,
        top_features=[feature.model_dump() for feature in req.top_features],
    )
    return ExplainResponse(explanation=result.explanation, disclaimer=result.disclaimer)


@app.get("/ag-results", response_model=AGResultsResponse)
def ag_results() -> AGResultsResponse:
    return AGResultsResponse(**asdict(load_ag_results()))
