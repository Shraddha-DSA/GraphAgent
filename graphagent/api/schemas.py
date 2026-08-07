"""
Pydantic request/response schemas.
"""

from pydantic import BaseModel
from typing import List


class WorkflowRequest(BaseModel):
    task: str

    agents: List[str]

    edges: List[List[str]]

    latency: float

    token_usage: int

    cost: float


class PredictionResponse(BaseModel):

    prediction: str

    success_probability: float


class RecommendationResponse(BaseModel):

    recommendation: List[str]

    estimated_success: float