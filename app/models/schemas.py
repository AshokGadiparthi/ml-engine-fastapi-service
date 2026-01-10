"""Pydantic models for API requests and responses."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


# ============ ENUMS ============

class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class Algorithm(str, Enum):
    # Classification
    LOGISTIC = "logistic"
    LOGISTIC_REGRESSION = "logistic_regression"  # Spring Boot sends this
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    SVM = "svm"
    GRADIENT_BOOSTING = "gradient_boosting"
    DECISION_TREE = "decision_tree"
    # Regression
    LINEAR = "linear"
    LINEAR_REGRESSION = "linear_regression"  # Spring Boot sends this
    RIDGE = "ridge"
    LASSO = "lasso"
    SVR = "svr"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"
    XGBOOST_REGRESSOR = "xgboost_regressor"


# ============ TRAINING REQUESTS ============

class TrainingConfig(BaseModel):
    """Configuration for model training."""
    use_feature_engineering: bool = False
    scaling_method: Optional[str] = "standard"  # standard, minmax, robust
    n_features: Optional[int] = None
    polynomial_degree: Optional[int] = None
    cv_folds: int = 5
    tune_hyperparameters: bool = False
    tuning_search_type: str = "grid"  # grid, random
    tuning_iterations: int = 50
    explain_model: bool = False


class TrainingRequest(BaseModel):
    """Request to start manual training."""
    dataset_id: Optional[str] = None  # Optional for Spring Boot compatibility
    dataset_path: Optional[str] = None
    target_column: str
    algorithm: str  # Accept string instead of enum for flexibility
    problem_type: str  # Accept string instead of enum for flexibility
    experiment_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None  # Accept dict for flexibility
    
    class Config:
        use_enum_values = True


class AutoMLRequest(BaseModel):
    """Request to start AutoML job."""
    dataset_id: str
    dataset_path: Optional[str] = None
    target_column: str
    problem_type: ProblemType
    experiment_name: Optional[str] = None
    cv_folds: int = 5
    max_training_time_minutes: int = 60
    use_feature_engineering: bool = True
    scaling_method: Optional[str] = "standard"
    explain_model: bool = False
    
    class Config:
        use_enum_values = True


# ============ JOB RESPONSES ============

class PhaseInfo(BaseModel):
    """Information about a training phase."""
    name: str
    label: str
    status: str  # pending, running, completed, failed
    progress: int = 0
    message: Optional[str] = None


class AlgorithmResult(BaseModel):
    """Result for a single algorithm in AutoML."""
    rank: int
    algorithm: str
    score: float
    std: float
    training_time_seconds: float
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc: Optional[float] = None
    # Regression metrics
    r2: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mse: Optional[float] = None


class FeatureImportance(BaseModel):
    """Feature importance entry."""
    feature: str
    importance: float
    rank: int


class JobResponse(BaseModel):
    """Response for job creation."""
    job_id: str
    status: JobStatus
    message: str
    created_at: datetime


class JobProgress(BaseModel):
    """Progress information for a running job."""
    job_id: str
    status: JobStatus
    progress: int  # 0-100
    current_phase: Optional[str] = None
    current_algorithm: Optional[str] = None
    phases: List[PhaseInfo] = []
    algorithms_completed: int = 0
    algorithms_total: int = 0
    current_best_score: Optional[float] = None
    current_best_algorithm: Optional[str] = None
    elapsed_seconds: int = 0
    estimated_remaining_seconds: Optional[int] = None
    logs: List[Dict[str, Any]] = []
    error_message: Optional[str] = None


class TrainingResult(BaseModel):
    """Result of completed training."""
    job_id: str
    status: JobStatus
    problem_type: ProblemType
    algorithm: str
    
    # Scores
    train_score: float
    test_score: float
    cv_score: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Metrics (depends on problem type)
    metrics: Dict[str, float] = {}
    
    # Model info
    model_id: str
    model_path: str
    feature_engineer_path: Optional[str] = None
    feature_names: List[str] = []
    n_features: int = 0
    
    # Dataset info
    dataset_id: str
    target_column: str
    train_size: int = 0
    test_size: int = 0
    
    # Feature importance
    feature_importance: List[FeatureImportance] = []
    
    # Timing
    training_time_seconds: float = 0
    completed_at: datetime


class AutoMLResult(BaseModel):
    """Result of completed AutoML job."""
    job_id: str
    status: JobStatus
    problem_type: ProblemType
    
    # Best model
    best_algorithm: str
    best_score: float
    best_metric: str
    
    # Leaderboard
    leaderboard: List[AlgorithmResult] = []
    
    # Model info
    model_id: str
    model_path: str
    feature_engineer_path: Optional[str] = None
    
    # Dataset info
    dataset_id: str
    target_column: str
    train_size: int = 0
    test_size: int = 0
    n_features: int = 0
    
    # Feature importance
    feature_importance: List[FeatureImportance] = []
    
    # Comparison CSV
    comparison_csv_path: Optional[str] = None
    
    # Timing
    total_training_time_seconds: float = 0
    completed_at: datetime


# ============ PREDICTION REQUESTS/RESPONSES ============

class PredictionRequest(BaseModel):
    """Request for single prediction."""
    model_id: str
    features: Dict[str, Any]


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    model_id: str
    dataset_path: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None


class PredictionResponse(BaseModel):
    """Response for prediction."""
    prediction: Any
    probability: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[Any]
    probabilities: Optional[List[Dict[str, float]]] = None
    output_path: Optional[str] = None
    count: int


# ============ MODEL MANAGEMENT ============

class ModelInfo(BaseModel):
    """Information about a saved model."""
    model_id: str
    name: str
    algorithm: str
    problem_type: ProblemType
    score: float
    metric: str
    model_path: str
    feature_engineer_path: Optional[str] = None
    created_at: datetime
    is_deployed: bool = False
    deployment_endpoint: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ModelListResponse(BaseModel):
    """List of models."""
    models: List[ModelInfo]
    total: int


# ============ EVALUATION ============

class EvaluationRequest(BaseModel):
    """Request for model evaluation."""
    model_id: str
    dataset_path: Optional[str] = None
    test_data: Optional[List[Dict[str, Any]]] = None
    test_labels: Optional[List[Any]] = None


class ClassificationMetrics(BaseModel):
    """Classification evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: Optional[float] = None
    confusion_matrix: List[List[int]] = []
    classification_report: Dict[str, Any] = {}


class RegressionMetrics(BaseModel):
    """Regression evaluation metrics."""
    r2: float
    mae: float
    mse: float
    rmse: float
    mape: Optional[float] = None


class EvaluationResponse(BaseModel):
    """Response for model evaluation."""
    model_id: str
    problem_type: ProblemType
    metrics: Dict[str, float]
    classification_metrics: Optional[ClassificationMetrics] = None
    regression_metrics: Optional[RegressionMetrics] = None
    plots: Dict[str, str] = {}  # plot_name -> path


# ============ EXPLAINABILITY ============

class ExplainabilityRequest(BaseModel):
    """Request for model explainability."""
    model_id: str
    data: Optional[List[Dict[str, Any]]] = None
    dataset_path: Optional[str] = None
    max_samples: int = 100


class ExplainabilityResponse(BaseModel):
    """Response for model explainability."""
    model_id: str
    feature_importance: List[FeatureImportance]
    shap_summary_plot: Optional[str] = None
    shap_values: Optional[Dict[str, List[float]]] = None


# ============ DATASET ============

class DatasetInfo(BaseModel):
    """Information about a dataset."""
    dataset_id: str
    name: str
    path: str
    row_count: int
    column_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    sample: List[Dict[str, Any]] = []


class DatasetUploadResponse(BaseModel):
    """Response for dataset upload."""
    dataset_id: str
    name: str
    path: str
    row_count: int
    column_count: int
    columns: List[str]
    message: str


# ============ HEALTH CHECK ============

class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    version: str
    ml_engine_status: str
    timestamp: datetime
