"""Models Management API Router."""
from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime

from app.models.schemas import ModelInfo, ModelListResponse, ProblemType
from app.services import ml_service

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=ModelListResponse)
async def list_models():
    """List all trained models."""
    models = ml_service.list_models()
    
    return ModelListResponse(
        models=[
            ModelInfo(
                model_id=m["model_id"],
                name=m["name"],
                algorithm=m["algorithm"],
                problem_type=ProblemType(m["problem_type"]),
                score=m["score"],
                metric=m["metric"],
                model_path=m["model_path"],
                feature_engineer_path=m.get("feature_engineer_path"),
                created_at=datetime.fromisoformat(m["created_at"]) if isinstance(m["created_at"], str) else m["created_at"],
                is_deployed=m.get("is_deployed", False),
                deployment_endpoint=f"/api/predictions/realtime/{m['model_id']}" if m.get("is_deployed") else None,
                metadata=m.get("metadata", {})
            )
            for m in models
        ],
        total=len(models)
    )


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get model details."""
    model = ml_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return ModelInfo(
        model_id=model["model_id"],
        name=model["name"],
        algorithm=model["algorithm"],
        problem_type=ProblemType(model["problem_type"]),
        score=model["score"],
        metric=model["metric"],
        model_path=model["model_path"],
        feature_engineer_path=model.get("feature_engineer_path"),
        created_at=datetime.fromisoformat(model["created_at"]) if isinstance(model["created_at"], str) else model["created_at"],
        is_deployed=model.get("is_deployed", False),
        deployment_endpoint=f"/api/predictions/realtime/{model['model_id']}" if model.get("is_deployed") else None,
        metadata=model.get("metadata", {})
    )


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model."""
    success = ml_service.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"message": "Model deleted", "model_id": model_id}


@router.post("/{model_id}/deploy")
async def deploy_model(model_id: str):
    """
    Deploy a model for serving predictions.
    
    After deployment, predictions can be made at:
    POST /api/predictions/realtime/{model_id}
    """
    model = ml_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Update deployment status
    ml_service.models_registry[model_id]["is_deployed"] = True
    ml_service._save_registry()
    
    endpoint = f"/api/predictions/realtime/{model_id}"
    
    return {
        "message": "Model deployed successfully",
        "model_id": model_id,
        "endpoint": endpoint,
        "status": "deployed"
    }


@router.post("/{model_id}/undeploy")
async def undeploy_model(model_id: str):
    """Undeploy a model."""
    model = ml_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    ml_service.models_registry[model_id]["is_deployed"] = False
    ml_service._save_registry()
    
    return {
        "message": "Model undeployed",
        "model_id": model_id,
        "status": "undeployed"
    }


@router.get("/{model_id}/feature-importance")
async def get_feature_importance(model_id: str):
    """Get feature importance for a model."""
    model = ml_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Try to get from metadata
    importance = model.get("metadata", {}).get("feature_importance", [])
    
    return {
        "model_id": model_id,
        "feature_importance": importance
    }
