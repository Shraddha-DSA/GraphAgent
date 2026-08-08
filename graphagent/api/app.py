from fastapi import FastAPI

from graphagent.api.predictor import Predictor
from graphagent.api.recommender import WorkflowRecommender
from graphagent.api.schemas import (
    WorkflowRequest,
    PredictionResponse,
    RecommendationResponse,
)

app = FastAPI(
    title="GraphAgent API",
    description="Graph Neural Network based Multi-Agent Workflow Analyzer",
    version="1.0.0",
)

predictor = Predictor()
recommender = WorkflowRecommender()


@app.get("/")
def home():

    return {
        "message": "Welcome to GraphAgent API",
        "status": "running",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: WorkflowRequest):
    workflow = request.model_dump()
    result = predictor.predict(workflow)
    return PredictionResponse(
        **result
    )


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(request: WorkflowRequest):

    workflow = request.model_dump()

    
    result = recommender.recommend(workflow)

    return RecommendationResponse(
        **result
    )