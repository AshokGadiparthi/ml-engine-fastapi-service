"""
Layer 3 Evaluation API Router
=============================

Exposes Layer 3 Enhanced Evaluation functions as REST endpoints.
"""
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
import logging
import numpy as np

from app.models.schemas import (
    ThresholdEvaluationResponse,
    BusinessImpactResponse,
    OptimalThresholdResponse,
    ProductionReadinessResponse,
    CompleteEvaluationResponse
)
from app.services import evaluation_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["Evaluation - Layer 3"])


@router.post("/threshold/{model_id}", response_model=ThresholdEvaluationResponse)
async def evaluate_with_threshold(
    model_id: str,
    body: Dict[str, Any] = Body(...)
):
    """
    Evaluate model at specific threshold.
    
    JSON Body:
    {
      "y_true": [0, 1, 1, 0],
      "y_pred_proba": [0.1, 0.9, 0.8, 0.2],
      "threshold": 0.5,
      "target_names": null
    }
    """
    try:
        y_true = body.get("y_true")
        y_pred_proba = body.get("y_pred_proba")
        threshold = body.get("threshold", 0.5)
        target_names = body.get("target_names")
        
        if not y_true or not y_pred_proba:
            raise ValueError("y_true and y_pred_proba are required")
        
        y_true = np.array(y_true, dtype=int)
        y_pred_proba = np.array(y_pred_proba, dtype=float)
        
        result = evaluation_service.evaluate_with_threshold(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            threshold=float(threshold),
            target_names=target_names
        )
        
        return ThresholdEvaluationResponse(
            model_id=model_id,
            threshold=float(threshold),
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
    body: Dict[str, Any] = Body(...)
):
    """
    Calculate business impact (cost/profit) of predictions.
    
    JSON Body:
    {
      "evaluation_result": {
        "confusion_matrix": {...},
        "metrics": {...},
        "rates": {...}
      },
      "cost_false_positive": 500,
      "cost_false_negative": 2000,
      "revenue_true_positive": 1000,
      "volume": 10000
    }
    """
    try:
        evaluation_result = body.get("evaluation_result")
        
        if not evaluation_result:
            raise ValueError("evaluation_result is required - get this from /threshold endpoint first")
        
        if "confusion_matrix" not in evaluation_result:
            raise ValueError(
                "Invalid evaluation_result structure. Make sure it comes from /threshold endpoint. "
                "Required keys: confusion_matrix, metrics, rates"
            )
        
        cost_fp = body.get("cost_false_positive", 500)
        cost_fn = body.get("cost_false_negative", 2000)
        revenue_tp = body.get("revenue_true_positive", 1000)
        volume = body.get("volume", 10000)
        
        result = evaluation_service.calculate_business_impact(
            evaluation_result=evaluation_result,
            cost_false_positive=float(cost_fp),
            cost_false_negative=float(cost_fn),
            revenue_true_positive=float(revenue_tp),
            volume=float(volume)
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
    request: Optional[Dict[str, Any]] = Body(None)
):
    """
    Find optimal threshold for profit maximization.
    """
    try:
        if request is None:
            request = {}
            
        y_true = request.get("y_true")
        y_pred_proba = request.get("y_pred_proba")
        cost_fp = request.get("cost_false_positive", 500)
        cost_fn = request.get("cost_false_negative", 2000)
        revenue_tp = request.get("revenue_true_positive", 1000)
        
        if not y_true or not y_pred_proba:
            raise ValueError("y_true and y_pred_proba are required")
        
        y_true = np.array(y_true, dtype=int)
        y_pred_proba = np.array(y_pred_proba, dtype=float)
        
        result = evaluation_service.get_optimal_threshold(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            cost_false_positive=float(cost_fp),
            cost_false_negative=float(cost_fn),
            revenue_true_positive=float(revenue_tp)
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
    body: Dict[str, Any] = Body(...)
):
    """
    Assess model production readiness with 18-point checklist.
    
    JSON Body:
    {
      "eval_result": {...},  or "evaluation_result": {...}
      "learning_curve": null,
      "business_impact": null,
      "feature_importance": null
    }
    """
    try:
        eval_result = body.get("eval_result") or body.get("evaluation_result")
        
        if not eval_result or "metrics" not in eval_result:
            raise ValueError(
                "Invalid evaluation_result structure. Make sure it comes from /threshold endpoint. "
                "Required keys: confusion_matrix, metrics, rates"
            )
        
        learning_curve = body.get("learning_curve") or {"gap": 0.0, "overfitting_status": "unknown"}
        business_impact = body.get("business_impact") or {"financial": {"net_profit": 0}}
        feature_importance = body.get("feature_importance") or {"total_features": 0, "features": []}
        
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
    body: Dict[str, Any] = Body(...)
):
    """
    Complete model evaluation in one call.
    
    Returns ALL metrics at once.
    """
    try:
        y_test = body.get("y_test")
        y_pred_proba = body.get("y_pred_proba")
        
        if not y_test or not y_pred_proba:
            raise ValueError("y_test and y_pred_proba are required")
        
        y_test = np.array(y_test, dtype=int)
        y_pred_proba = np.array(y_pred_proba, dtype=float)
        
        X_test = np.array(body.get("X_test")) if body.get("X_test") else None
        y_train = np.array(body.get("y_train"), dtype=int) if body.get("y_train") else None
        y_pred_train = np.array(body.get("y_pred_train"), dtype=float) if body.get("y_pred_train") else None
        
        threshold = body.get("threshold", 0.5)
        cost_fp = body.get("cost_false_positive", 500)
        cost_fn = body.get("cost_false_negative", 2000)
        revenue_tp = body.get("revenue_true_positive", 1000)
        
        result = evaluation_service.complete_evaluation(
            model=None,
            X_test=X_test,
            y_test=y_test,
            y_pred_proba=y_pred_proba,
            y_train=y_train,
            y_pred_train_proba=y_pred_train,
            threshold=float(threshold),
            cost_fp=float(cost_fp),
            cost_fn=float(cost_fn),
            revenue_tp=float(revenue_tp)
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
            "Complete model evaluation"
        ]
    }
