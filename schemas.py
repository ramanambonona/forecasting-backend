from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class AnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_variable: Optional[str] = None

class ForecastRequest(BaseModel):
    data: List[Dict[str, Any]]
    model_type: str
    periods: int
    params: Dict[str, Any]
    target_variables: List[str]
    orientation: str = "dates_in_rows"

class ForecastResponse(BaseModel):
    success: bool
    forecasts: Dict[str, Any]
    metrics: Dict[str, Any]
    plots: Dict[str, Any]

class ModelInfo(BaseModel):
    name: str
    category: str
    description: str
    parameters: Dict[str, Any]