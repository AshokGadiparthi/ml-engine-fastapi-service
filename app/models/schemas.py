"""Pydantic models for API requests and responses."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
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


# ============ EDA ANALYSIS ============

class EDARequest(BaseModel):
    """Request for EDA analysis."""
    dataset_id: str
    target_column: Optional[str] = None
    sample_rows: Optional[int] = None


class EDAResponse(BaseModel):
    """Response for EDA analysis."""
    eda_id: str
    dataset_id: str
    status: str
    quality_score: float
    insights_count: int
    features_analyzed: int
    timestamp: str


class EDASummary(BaseModel):
    """Executive summary of EDA analysis."""
    dataset_id: str
    timestamp: str
    quality_score: float
    assessment: str
    completeness: float
    uniqueness: float
    features: int
    insights_count: int
    issues_critical: int
    issues_high: int
    issues_medium: int
    top_concern: Optional[str] = None
    recommendation: Optional[str] = None


# ============ LAYER 3: ENHANCED EVALUATION ============

class ThresholdEvaluationRequest(BaseModel):
    """Request for threshold-based evaluation."""
    y_true: List[int]  # True labels (0/1)
    y_pred_proba: List[float]  # Predicted probabilities
    threshold: float = 0.5  # Classification threshold
    target_names: Optional[Tuple[str, str]] = None  # Class names


class ConfusionMatrixData(BaseModel):
    """Confusion matrix with counts."""
    tn: int  # True negatives
    fp: int  # False positives
    fn: int  # False negatives
    tp: int  # True positives
    total: int


class MetricsData(BaseModel):
    """Classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float


class RatesData(BaseModel):
    """Classification rates."""
    false_positive_rate: float
    false_negative_rate: float
    true_positive_rate: float
    true_negative_rate: float


class ClassImbalanceData(BaseModel):
    """Class imbalance information."""
    class_0_count: int
    class_1_count: int
    imbalance_ratio: float
    class_distribution: str  # "balanced", "slightly_imbalanced", "imbalanced"


class ThresholdEvaluationResponse(BaseModel):
    """Response for threshold evaluation."""
    model_id: str
    threshold: float
    confusion_matrix: ConfusionMatrixData
    metrics: MetricsData
    rates: RatesData
    class_imbalance: ClassImbalanceData


class BusinessImpactRequest(BaseModel):
    """Request for business impact calculation."""
    evaluation_result: Dict[str, Any]  # Result from threshold evaluation
    cost_false_positive: float = 500  # Cost of FP in dollars
    cost_false_negative: float = 2000  # Cost of FN in dollars
    revenue_true_positive: float = 1000  # Revenue from TP in dollars
    volume: float = 10000  # Prediction volume


class CostsData(BaseModel):
    """Cost breakdown."""
    cost_per_false_positive: float
    cost_per_false_negative: float
    total_false_positive_cost: float
    total_false_negative_cost: float
    total_cost: float


class RevenueData(BaseModel):
    """Revenue breakdown."""
    revenue_per_true_positive: float
    total_revenue_from_tp: float


class FinancialData(BaseModel):
    """Financial summary."""
    net_profit: float
    approval_rate: float
    roi_improvement_percent: float


class BusinessImpactResponse(BaseModel):
    """Response for business impact calculation."""
    model_id: str
    costs: CostsData
    revenue: RevenueData
    financial: FinancialData


class OptimalThresholdRequest(BaseModel):
    """Request for optimal threshold finding."""
    y_true: List[int]  # True labels
    y_pred_proba: List[float]  # Predicted probabilities
    cost_false_positive: float = 500  # Cost of FP
    cost_false_negative: float = 2000  # Cost of FN
    revenue_true_positive: float = 1000  # Revenue from TP


class OptimalThresholdResponse(BaseModel):
    """Response for optimal threshold finding."""
    model_id: str
    current_threshold: float
    optimal_threshold: float
    current_profit: float
    optimal_profit: float
    improvement: float
    recommendation: str


class ProductionReadinessCriteria(BaseModel):
    """Single production readiness criteria."""
    category: str  # Category name
    name: str  # Criteria name
    status: str  # pass, fail, pending
    details: str  # Details about status


class ProductionReadinessSummary(BaseModel):
    """Summary of production readiness."""
    total_criteria: int
    passed: int
    failed: int
    pending: int
    pass_rate: float


class ProductionReadinessResponse(BaseModel):
    """Response for production readiness assessment."""
    model_id: str
    overall_status: str  # ready, pending, not_ready
    summary: ProductionReadinessSummary
    criteria: List[ProductionReadinessCriteria]
    deployment_recommendation: str
    estimated_time_to_production: str


class ROCCurveData(BaseModel):
    """ROC curve data."""
    fpr: List[float]
    tpr: List[float]
    auc: float


class PRCurveData(BaseModel):
    """Precision-Recall curve data."""
    precision: List[float]
    recall: List[float]
    ap: float


class CurveData(BaseModel):
    """ROC and PR curve data."""
    roc_curve: ROCCurveData
    pr_curve: PRCurveData


class LearningCurveData(BaseModel):
    """Learning curve analysis."""
    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    train_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    gap: float  # Train-test gap
    overfitting_status: str  # none, mild, moderate, severe


class FeatureImportanceItem(BaseModel):
    """Single feature importance entry."""
    rank: int
    name: str
    importance_score: float
    importance_percent: float
    correlation_with_target: float
    correlation_strength: str


class FeatureImportanceData(BaseModel):
    """Feature importance analysis."""
    features: List[FeatureImportanceItem]
    top_3_importance_percent: float
    total_features: int
    strong_correlations: int


class CompleteEvaluationRequest(BaseModel):
    """Request for complete model evaluation."""
    y_test: List[int]  # Test labels
    y_pred_proba: List[float]  # Test predictions
    X_test: Optional[List[List[float]]] = None  # Test features (optional)
    y_train: Optional[List[int]] = None  # Train labels (optional)
    X_train: Optional[List[List[float]]] = None  # Train features (optional)
    y_pred_train: Optional[List[float]] = None  # Train predictions (optional)
    feature_names: Optional[List[str]] = None  # Feature names
    cost_false_positive: float = 500
    cost_false_negative: float = 2000
    revenue_true_positive: float = 1000
    threshold: float = 0.5


class CompleteEvaluationResponse(BaseModel):
    """Response for complete model evaluation."""
    model_id: str
    timestamp: str
    evaluation: Dict[str, Any]  # Metrics and confusion matrix
    business_impact: Dict[str, Any]  # Costs and profit
    curves: Dict[str, Any]  # ROC and PR curve data
    learning_curve: Dict[str, Any]  # Overfitting analysis
    feature_importance: Dict[str, Any]  # Feature ranking
    optimal_threshold: Dict[str, Any]  # Profit-optimizing threshold
    production_readiness: Dict[str, Any]  # Deployment checklist


# ============ HEALTH CHECK ============

class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    version: str
    ml_engine_status: str
    timestamp: datetime
