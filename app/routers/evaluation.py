"""
Layer 3 Evaluation API Router - FIXED VERSION
=============================

Exposes Layer 3 Enhanced Evaluation functions as REST endpoints.

FIX: Now loads and passes real feature names to evaluation functions
"""
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional, List
import logging
import numpy as np
import json
import os
import joblib
from pathlib import Path

from app.services import evaluation_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["Evaluation - Layer 3"])


def convert_nan_to_none(obj):
    """Recursively convert NaN/inf values to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, np.ndarray)):
        return obj.item() if isinstance(obj, np.ndarray) else int(obj)
    return obj


def load_feature_names() -> list[str] | None:
    paths_to_try = [
        Path.home() / "work/latest/kedro-ml-engine/models/feature_names.pkl",
        Path(__file__).resolve().parents[2] / "models" / "feature_names.pkl",  # adjust if needed
    ]

    for p in paths_to_try:
        if p.exists():
            feature_names = joblib.load(p)
            feature_names = [str(n).replace("-", "_").replace(" ", "_") for n in feature_names]
            logger.info(f"Loaded {len(feature_names)} feature names from {p}")
            return feature_names

    logger.warning("feature_names.pkl not found")
    return None


# ============================================================================
# THRESHOLD EVALUATION
# ============================================================================

@router.post("/threshold/{model_id}")
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
        
        # Convert NaN values to None for JSON serialization
        result = convert_nan_to_none(result)
        result["model_id"] = model_id
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Threshold evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BUSINESS IMPACT
# ============================================================================

@router.post("/business-impact/{model_id}")
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
        eval_result = body.get("evaluation_result")
        cost_fp = body.get("cost_false_positive", 500)
        cost_fn = body.get("cost_false_negative", 2000)
        revenue_tp = body.get("revenue_true_positive", 1000)
        volume = body.get("volume", 10000)
        
        if not eval_result:
            raise ValueError("evaluation_result is required")
        
        result = evaluation_service.calculate_business_impact(
            evaluation_result=eval_result,
            cost_false_positive=float(cost_fp),
            cost_false_negative=float(cost_fn),
            revenue_true_positive=float(revenue_tp),
            volume=float(volume)
        )
        
        result = convert_nan_to_none(result)
        result["model_id"] = model_id
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Business impact calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# OPTIMAL THRESHOLD
# ============================================================================

@router.post("/optimal-threshold/{model_id}")
async def get_optimal_threshold(
    model_id: str,
    body: Dict[str, Any] = Body(...)
):
    """
    Find optimal threshold for profit maximization.
    
    JSON Body:
    {
      "y_true": [0, 1, 1, 0],
      "y_pred_proba": [0.1, 0.9, 0.8, 0.2],
      "cost_false_positive": 500,
      "cost_false_negative": 2000,
      "revenue_true_positive": 1000
    }
    """
    try:
        y_true = body.get("y_true")
        y_pred_proba = body.get("y_pred_proba")
        cost_fp = body.get("cost_false_positive", 500)
        cost_fn = body.get("cost_false_negative", 2000)
        revenue_tp = body.get("revenue_true_positive", 1000)
        
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
        
        result = convert_nan_to_none(result)
        result["model_id"] = model_id
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Optimal threshold finding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PRODUCTION READINESS
# ============================================================================

@router.post("/production-readiness/{model_id}")
async def assess_production_readiness(
    model_id: str,
    body: Dict[str, Any] = Body(...)
):
    """
    Assess production readiness with 19-point checklist.
    
    JSON Body:
    {
      "evaluation_result": {...},
      "learning_curve": {...},
      "business_impact": {...},
      "feature_importance": {...}
    }
    """
    try:
        eval_result = body.get("evaluation_result")
        learning_curve = body.get("learning_curve")
        business_impact = body.get("business_impact")
        feature_importance = body.get("feature_importance")
        
        if not all([eval_result, learning_curve, business_impact, feature_importance]):
            raise ValueError("All result types are required")
        
        result = evaluation_service.assess_production_readiness(
            evaluation_result=eval_result,
            learning_curve=learning_curve,
            business_impact=business_impact,
            feature_importance=feature_importance
        )
        
        result = convert_nan_to_none(result)
        result["model_id"] = model_id
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Production readiness assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# COMPLETE EVALUATION - FIXED TO LOAD AND PASS FEATURE NAMES
# ============================================================================

@router.post("/complete/{model_id}")
async def complete_evaluation(
    model_id: str,
    body: Dict[str, Any] = Body(...)
):
    """
    Complete model evaluation in one call.
    
    Returns ALL metrics at once including feature importance with REAL feature names.
    
    FIXED: Now loads feature names from saved model metadata and passes them
    through the evaluation pipeline.
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
        
        # ✅ FIX: Load feature names from saved model metadata
        #feature_names = load_feature_names()
        
        # ✅ FIX: Pass feature_names to evaluation service
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
            revenue_tp=float(revenue_tp),
            feature_names=load_feature_names()
        )
        
        # Convert NaN values to None for JSON serialization
        result = convert_nan_to_none(result)
        result["model_id"] = model_id
        return result
    except ValueError as e:
        logger.error(f"ValueError: {e}", exc_info=True)  # ✅ Add exc_info
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Complete evaluation failed: {e}", exc_info=True)  # ✅ Add exc_info
        import traceback
        traceback.print_exc()  # ✅ Print full traceback
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH CHECK
# ============================================================================

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
            "Complete model evaluation",
            "Real feature names (FIXED)"
        ]
    }