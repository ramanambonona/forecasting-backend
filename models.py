from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

class ModelCategory(str, Enum):
    STATISTICAL = "Statistical"
    MULTIVARIATE = "Multivariate"
    MACHINE_LEARNING = "Machine Learning"
    NEURAL_NETWORKS = "Neural Networks"
    BAYESIAN = "Bayesian"
    SPECIALIZED = "Specialized"

class ForecastHorizon(str, Enum):
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"

class DataOrientation(str, Enum):
    DATES_IN_ROWS = "dates_in_rows"
    DATES_IN_COLUMNS = "dates_in_columns"

class DataUploadRequest(BaseModel):
    filename: str
    file_type: str
    orientation: Optional[DataOrientation] = DataOrientation.DATES_IN_ROWS

class AnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_variable: Optional[str] = None
    analysis_type: str = "full"

class ForecastRequest(BaseModel):
    data: List[Dict[str, Any]]
    model_type: str
    periods: int
    params: Dict[str, Any] = {}
    target_variables: List[str]
    orientation: DataOrientation = DataOrientation.DATES_IN_ROWS
    confidence_interval: Optional[float] = 0.95

class ModelComparisonRequest(BaseModel):
    data: List[Dict[str, Any]]
    models: List[str]
    periods: int
    target_variable: str
    validation_split: float = 0.2

class BaseResponse(BaseModel):
    success: bool
    message: str
    timestamp: datetime = datetime.now()

class DataUploadResponse(BaseResponse):
    data_preview: Optional[List[Dict[str, Any]]] = None
    summary: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseResponse):
    analysis_results: Dict[str, Any]
    recommendations: List[str]
    plots: Optional[Dict[str, Any]] = None

class ForecastResponse(BaseResponse):
    forecasts: Dict[str, Any]
    metrics: Dict[str, Any]
    model_info: Dict[str, Any]
    plots: Dict[str, Any]
    export_data: Optional[Dict[str, Any]] = None

class ModelInfoResponse(BaseResponse):
    models: Dict[str, Any]
    categories: Dict[str, List[str]]
    recommended_models: List[str]

class TimeSeriesPoint(BaseModel):
    date: str
    value: float

class ForecastPoint(BaseModel):
    date: str
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

class ModelMetrics(BaseModel):
    mape: float
    rmse: float
    mae: float
    r_squared: Optional[float] = None
    auc: Optional[float] = None

class TimeSeriesAnalysis(BaseModel):
    variable: str
    trend: str
    seasonality: str
    stationarity: str
    volatility: float
    recommendations: List[str]
    decomposition: Optional[Dict[str, Any]] = None

class ForecastModel(BaseModel):
    name: str
    category: ModelCategory
    description: str
    parameters: Dict[str, Any]
    requirements: Dict[str, Any]
    is_available: bool = True

class User(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime
    preferences: Dict[str, Any] = {}

class ForecastJob(BaseModel):
    id: str
    user_id: str
    model_type: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class SavedForecast(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    model_type: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    created_at: datetime
    updated_at: datetime