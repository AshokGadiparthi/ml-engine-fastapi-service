"""
Layer 3 Evaluation API Router
=============================

Exposes Layer 3 Enhanced Evaluation functions as REST endpoints.
- Real-time threshold adjustment
- Business impact analysis
- Optimal threshold finding
- Production readiness assessment
- Complete model evaluation
"""
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, Optional, List
import logging
import numpy as np

from app.models.schemas import (
    ThresholdEvaluationResponse,
    BusinessImpactResponse,
    OptimalThresholdRequest,
    OptimalThresholdResponse,
    ProductionReadinessResponse,
    CompleteEvaluationRequest,
    CompleteEvaluationResponse
)
from app.services import evaluation_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["Evaluation - Layer 3"])


@router.post("/threshold/{model_id}", response_model=ThresholdEvaluationResponse)
async def evaluate_with_threshold(
    model_id: str,
    y_true: List = Body(...),
    y_pred_proba: List = Body(...),
    threshold: float = Body(0.5),
    target_names: Optional[List[str]] = Body(None)
):
    """
    Evaluate model at specific threshold.
    
    Instantly recalculates all metrics at new threshold without retraining.
    
    Args:
        model_id: Model identifier
        y_true: True labels (0/1)
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        target_names: Class names
    
    Returns:
        Metrics, confusion matrix, and rates at new threshold
    """
    try:
        if not y_true or not y_pred_proba:
            raise ValueError("y_true and y_pred_proba are required")
        
        y_true = np.array(y_true, dtype=int)
        y_pred_proba = np.array(y_pred_proba, dtype=float)
        
        result = evaluation_service.evaluate_with_threshold(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            target_names=target_names
        )
        
        return ThresholdEvaluationResponse(
            model_id=model_id,
            threshold=threshold,
            **result
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Threshold evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/business-impact/{model_id}", response_model=BusinessImpactResponse)
async def calculate_business_impact(
    model_id: str,
    evaluation_result: Dict[str, Any] = Body(...),
    cost_false_positive: float = Body(500),
    cost_false_negative: float = Body(2000),
    revenue_true_positive: float = Body(1000),
    volume: float = Body(10000)
):
    """
    Calculate business impact (cost/profit) of predictions.
    
    Translates ML metrics to business value with configurable costs and revenue.
    
    Args:
        model_id: Model identifier
        evaluation_result: Result from /threshold endpoint (required)
        cost_false_positive: Cost of FP in dollars
        cost_false_negative: Cost of FN in dollars
        revenue_true_positive: Revenue from TP in dollars
        volume: Prediction volume
    
    Returns:
        Costs, revenue, and net profit analysis
    """
    try:
        if not evaluation_result:
            raise ValueError("evaluation_result is required - get this from /threshold endpoint first")
        
        # Ensure evaluation_result has required structure
        if "confusion_matrix" not in evaluation_result:
            raise ValueError(
                "Invalid evaluation_result structure. Make sure it comes from /threshold endpoint. "
                "Required keys: confusion_matrix, metrics, rates"
            )
        
        result = evaluation_service.calculate_business_impact(
            evaluation_result=evaluation_result,
            cost_false_positive=cost_false_positive,
            cost_false_negative=cost_false_negative,
            revenue_true_positive=revenue_true_positive,
            volume=volume
        )
        
        return BusinessImpactResponse(
            model_id=model_id,
            **result
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Business impact calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimal-threshold/{model_id}", response_model=OptimalThresholdResponse)
async def get_optimal_threshold(
    model_id: str,
    request: OptimalThresholdRequest
):
    """
    Find threshold that maximizes profit.
    
    Tests multiple thresholds (0.1-0.95) and returns the one with highest profit.
    Useful for optimizing model for business metrics.
    
    Args:
        model_id: Model identifier
        request: Optimization request with predictions and costs
    
    Returns:
        Optimal threshold and expected profit
    """
    try:
        if not request.y_true or not request.y_pred_proba:
            raise ValueError("y_true and y_pred_proba are required")
        
        y_true = np.array(request.y_true)
        y_pred_proba = np.array(request.y_pred_proba)
        
        result = evaluation_service.get_optimal_threshold(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            cost_false_positive=request.cost_false_positive,
            cost_false_negative=request.cost_false_negative,
            revenue_true_positive=request.revenue_true_positive
        )
        
        return OptimalThresholdResponse(
            model_id=model_id,
            **result
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Optimal threshold finding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/production-readiness/{model_id}", response_model=ProductionReadinessResponse)
async def assess_production_readiness(
    model_id: str,
    eval_result: Dict[str, Any] = Body(..., embed=False),
    learning_curve: Optional[Dict[str, Any]] = Body(None),
    business_impact: Optional[Dict[str, Any]] = Body(None),
    feature_importance: Optional[Dict[str, Any]] = Body(None)
):
    """
    Assess model production readiness with 18-point checklist.
    
    Args:
        model_id: Model identifier
        eval_result: Evaluation result (from /threshold endpoint)
        learning_curve: Learning curve analysis (optional)
        business_impact: Business impact metrics (optional)
        feature_importance: Feature importance (optional)
    
    Returns:
        Readiness assessment with pass/fail status
    """
    try:
        # Validate evaluation_result structure
        if "metrics" not in eval_result:
            raise ValueError(
                "Invalid evaluation_result structure. Make sure it comes from /threshold endpoint. "
                "Required keys: confusion_matrix, metrics, rates"
            )
        
        # Use defaults if not provided
        learning_curve = learning_curve or {"gap": 0.0, "overfitting_status": "unknown"}
        business_impact = business_impact or {"financial": {"net_profit": 0}}
        feature_importance = feature_importance or {"total_features": 0, "features": []}
        
        result = evaluation_service.assess_production_readiness(
            evaluation_result=eval_result,
            learning_curve=learning_curve,
            business_impact=business_impact,
            feature_importance=feature_importance
        )
        
        return ProductionReadinessResponse(
            model_id=model_id,
            **result
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Production readiness assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complete/{model_id}", response_model=CompleteEvaluationResponse)
async def complete_evaluation(
    model_id: str,
    request: CompleteEvaluationRequest
):
    """
    Complete model evaluation in one call.
    
    Returns ALL metrics at once:
    - Evaluation (confusion matrix, metrics)
    - Business impact (costs, profit, ROI)
    - ROC/PR curves (data for plotting)
    - Learning curve (overfitting analysis)
    - Feature importance (with correlation)
    - Optimal threshold (profit-maximizing)
    - Production readiness (18-point checklist)
    
    Perfect for dashboard backends that need comprehensive metrics.
    Response time: <500ms for typical models.
    
    Args:
        model_id: Model identifier
        request: Complete evaluation request
    
    Returns:
        Comprehensive evaluation package
    """
    try:
        if not request.y_test or not request.y_pred_proba:
            raise ValueError("y_test and y_pred_proba are required")
        
        # Convert to numpy arrays
        y_test = np.array(request.y_test)
        y_pred_proba = np.array(request.y_pred_proba)
        
        X_test = np.array(request.X_test) if request.X_test else None
        y_train = np.array(request.y_train) if request.y_train else None
        y_pred_train = np.array(request.y_pred_train) if request.y_pred_train else None
        
        # For complete evaluation, we need the actual model object
        # This is a simplified version that works with arrays
        # In production, you'd pass the actual model loaded from disk
        
        result = evaluation_service.complete_evaluation(
            model=None,  # Not needed for array-based evaluation
            X_test=X_test,
            y_test=y_test,
            y_pred_proba=y_pred_proba,
            y_train=y_train,
            y_pred_train_proba=y_pred_train,
            threshold=request.threshold,
            cost_fp=request.cost_false_positive,
            cost_fn=request.cost_false_negative,
            revenue_tp=request.revenue_true_positive
        )
        
        return CompleteEvaluationResponse(
            model_id=model_id,
            **result
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Complete evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def evaluation_health():
    """Health check for evaluation service."""
    try:
        # Try to import Layer 3 functions
        from ml_engine.evaluation_enhanced import evaluate_with_threshold
        status = "ok"
    except ImportError:
        status = "error: Layer 3 not available"
    
    return {
        "service": "evaluation",
        "status": status,
        "layer": "Layer 3 Enhanced Evaluation",
        "features": [
            "Real-time threshold adjustment",
            "Business impact calculation",
            "Optimal threshold finding",
            "Production readiness assessment",
            "Complete model evaluation",
            "Feature importance analysis",
            "Learning curve analysis"
        ]
    }
