"""
EDA Analysis Router
===================

This router integrates with the external ML Engine (separate project).
It imports EDAAnalyticsEngine from ml_engine and provides REST endpoints.

The EDA engine must be installed separately:
    pip install <path-to-ml-engine> -e .

Or add ML Engine to PYTHONPATH:
    export PYTHONPATH="$PYTHONPATH:/path/to/ml-engine/src"
"""

from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from app.models.schemas import EDARequest, EDAResponse, EDASummary
from app.config import settings

# Import EDA Engine from external ML Engine project
try:
    from ml_engine.eda_analytics_engine import EDAAnalyticsEngine
    EDA_AVAILABLE = True
except ImportError:
    EDA_AVAILABLE = False
    EDAAnalyticsEngine = None

router = APIRouter(prefix="/eda", tags=["EDA"])

# Results cache
eda_cache = {}


@router.post("/analyze", response_model=EDAResponse)
async def analyze_dataset(request: EDARequest):
    """
    Analyze dataset using external ML Engine's EDAAnalyticsEngine.
    
    Requires:
    - dataset_id: ID of uploaded dataset
    - target_column (optional): Target column for analysis
    
    Returns:
    - eda_id: ID for retrieving results
    - quality_score: Data quality (0-100)
    - insights_count: Number of insights found
    - features_analyzed: Number of features
    """
    if not EDA_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ML Engine not available. Install ML Engine: pip install <ml-engine-path>"
        )
    
    try:
        # Get dataset from registry
        from app.routers.datasets import datasets_registry
        
        if request.dataset_id not in datasets_registry:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_info = datasets_registry[request.dataset_id]
        
        # Load CSV
        df = pd.read_csv(dataset_info["path"])
        
        # Sample if needed
        if request.sample_rows and len(df) > request.sample_rows:
            df = df.sample(n=request.sample_rows, random_state=42)
        
        # Validate target column
        if request.target_column and request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail="Target column not found")
        
        # Run EDA using external ML Engine
        eda_engine = EDAAnalyticsEngine()
        results = eda_engine.analyze(df, target_column=request.target_column)
        
        # Cache results
        eda_id = str(uuid.uuid4())
        eda_cache[eda_id] = {
            "dataset_id": request.dataset_id,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        return EDAResponse(
            eda_id=eda_id,
            dataset_id=request.dataset_id,
            status="completed",
            quality_score=results['quality']['overall_score'],
            insights_count=len(results['insights']),
            features_analyzed=len(results['features']),
            timestamp=eda_cache[eda_id]['timestamp']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EDA analysis failed: {str(e)}")


@router.get("/results/{eda_id}")
async def get_results(eda_id: str):
    """Get complete EDA results."""
    if eda_id not in eda_cache:
        raise HTTPException(status_code=404, detail="Results not found")
    
    cached = eda_cache[eda_id]
    return {
        "eda_id": eda_id,
        "dataset_id": cached["dataset_id"],
        "timestamp": cached["timestamp"],
        "data": cached["results"]
    }


@router.get("/quality/{eda_id}")
async def get_quality(eda_id: str):
    """Get quality metrics."""
    if eda_id not in eda_cache:
        raise HTTPException(status_code=404, detail="Results not found")
    
    results = eda_cache[eda_id]["results"]
    quality = results["quality"]
    
    return {
        "overall_score": quality.get("overall_score", 0),
        "assessment": quality.get("assessment", "Unknown"),
        "dimensions": quality.get("dimensions", {}),
        "recommendations": quality.get("recommendations", [])
    }


@router.get("/insights/{eda_id}")
async def get_insights(eda_id: str):
    """Get all insights."""
    if eda_id not in eda_cache:
        raise HTTPException(status_code=404, detail="Results not found")
    
    insights = eda_cache[eda_id]["results"]["insights"]
    
    return {
        "total": len(insights),
        "insights": insights,
        "by_severity": {
            "CRITICAL": sum(1 for i in insights if i.get("severity") == "CRITICAL"),
            "HIGH": sum(1 for i in insights if i.get("severity") == "HIGH"),
            "MEDIUM": sum(1 for i in insights if i.get("severity") == "MEDIUM"),
            "LOW": sum(1 for i in insights if i.get("severity") == "LOW"),
        }
    }


@router.get("/features/{eda_id}")
async def get_features(eda_id: str, limit: int = Query(10)):
    """Get feature importance."""
    if eda_id not in eda_cache:
        raise HTTPException(status_code=404, detail="Results not found")
    
    features = eda_cache[eda_id]["results"]["feature_importance"]
    
    return {
        "total": len(features),
        "features": features[:limit]
    }


@router.get("/summary/{eda_id}", response_model=EDASummary)
async def get_summary(eda_id: str):
    """Get executive summary."""
    if eda_id not in eda_cache:
        raise HTTPException(status_code=404, detail="Results not found")
    
    cached = eda_cache[eda_id]
    results = cached["results"]
    quality = results["quality"]
    insights = results["insights"]
    
    dims = quality.get("dimensions", {})
    
    return EDASummary(
        dataset_id=cached["dataset_id"],
        timestamp=cached["timestamp"],
        quality_score=quality.get("overall_score", 0),
        assessment=quality.get("assessment", "Unknown"),
        completeness=dims.get("completeness", 0),
        uniqueness=dims.get("uniqueness", 0),
        features=len(results["features"]),
        insights_count=len(insights),
        issues_critical=sum(1 for i in insights if i.get("severity") == "CRITICAL"),
        issues_high=sum(1 for i in insights if i.get("severity") == "HIGH"),
        issues_medium=sum(1 for i in insights if i.get("severity") == "MEDIUM"),
        top_concern=insights[0]["title"] if insights else None,
        recommendation=quality.get("recommendations", ["None"])[0]
    )


@router.get("/results/{eda_id}/histogram")
async def get_histogram_data(eda_id: str):
    """Get histogram data for numeric features."""
    if eda_id not in eda_cache:
        return {"error": "Results not found", "detail": "Not Found"}
    
    results = eda_cache[eda_id]["results"]
    
    # Return numeric features analysis or empty dict if not available
    return {
        "eda_id": eda_id,
        "type": "histogram",
        "data": results.get("numeric_features", {}),
        "timestamp": eda_cache[eda_id]["timestamp"]
    }


@router.get("/results/{eda_id}/categorical")
async def get_categorical_data(eda_id: str):
    """Get categorical data for categorical features."""
    if eda_id not in eda_cache:
        return {"error": "Results not found", "detail": "Not Found"}
    
    results = eda_cache[eda_id]["results"]
    
    # Return categorical features analysis or empty dict if not available
    return {
        "eda_id": eda_id,
        "type": "categorical",
        "data": results.get("categorical_features", {}),
        "timestamp": eda_cache[eda_id]["timestamp"]
    }


@router.get("/results/{eda_id}/missing-pattern")
async def get_missing_pattern(eda_id: str):
    """Get missing data patterns."""
    if eda_id not in eda_cache:
        return {"error": "Results not found", "detail": "Not Found"}
    
    results = eda_cache[eda_id]["results"]
    
    # Return missing data analysis or empty dict if not available
    return {
        "eda_id": eda_id,
        "type": "missing_pattern",
        "data": results.get("missing_values", {}),
        "timestamp": eda_cache[eda_id]["timestamp"]
    }


@router.get("/results/{eda_id}/outliers")
async def get_outliers_data(eda_id: str):
    """Get outlier detection results."""
    if eda_id not in eda_cache:
        return {"error": "Results not found", "detail": "Not Found"}
    
    results = eda_cache[eda_id]["results"]
    
    # Return outliers analysis or empty dict if not available
    return {
        "eda_id": eda_id,
        "type": "outliers",
        "data": results.get("outliers", {}),
        "timestamp": eda_cache[eda_id]["timestamp"]
    }


@router.get("/results/{eda_id}/correlation")
async def get_correlation_data(eda_id: str):
    """Get correlation matrix and strong correlations."""
    if eda_id not in eda_cache:
        return {"error": "Results not found", "detail": "Not Found"}
    
    results = eda_cache[eda_id]["results"]
    
    # Return correlation analysis or empty dict if not available
    return {
        "eda_id": eda_id,
        "type": "correlation",
        "data": results.get("correlations", {}),
        "timestamp": eda_cache[eda_id]["timestamp"]
    }


@router.get("/health")
async def health():
    """Check if EDA engine is available."""
    return {
        "service": "eda",
        "available": EDA_AVAILABLE,
        "ml_engine": "available" if EDA_AVAILABLE else "not installed",
        "message": "ML Engine EDAAnalyticsEngine is available" if EDA_AVAILABLE else "Install ML Engine to use EDA"
    }
