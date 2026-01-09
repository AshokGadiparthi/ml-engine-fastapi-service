"""Predictions API Router."""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from app.models.schemas import (
    PredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse
)
from app.services import ml_service

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("/realtime/{model_id}", response_model=PredictionResponse)
async def predict_realtime(model_id: str, features: Dict[str, Any]):
    """
    Make real-time prediction for a single record.
    
    Send features as JSON object, get prediction back.
    """
    try:
        result = ml_service.predict(model_id, features)
        return PredictionResponse(
            prediction=result["prediction"],
            probability=result.get("probability"),
            confidence=result.get("confidence")
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/batch/{model_id}", response_model=BatchPredictionResponse)
async def predict_batch(model_id: str, request: BatchPredictionRequest):
    """
    Make batch predictions for multiple records.
    
    Send list of feature records, get list of predictions back.
    """
    if not request.data:
        raise HTTPException(status_code=400, detail="No data provided")
    
    try:
        result = ml_service.predict_batch(model_id, request.data)
        return BatchPredictionResponse(
            predictions=result["predictions"],
            probabilities=result.get("probabilities"),
            count=result["count"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/explain/{model_id}")
async def explain_prediction(model_id: str, features: Dict[str, Any]):
    """
    Get explanation for a prediction using SHAP.
    
    Returns feature contributions to the prediction.
    """
    model_info = ml_service.get_model(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # TODO: Implement SHAP explanation for single prediction
    # For now, return feature importance from model
    
    result = ml_service.predict(model_id, features)
    
    return {
        "prediction": result["prediction"],
        "probability": result.get("probability"),
        "feature_importance": ml_service.models_registry[model_id].get("metadata", {}).get("feature_importance", []),
        "explanation": "Feature importance shows global model importance. For local SHAP values, use /explain endpoint."
    }
