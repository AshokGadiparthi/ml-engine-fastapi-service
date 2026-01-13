"""
Enhanced EDA Analysis Router
==============================

Comprehensive EDA endpoints with standalone analytics service.
No external dependencies required.
All analysis self-contained.
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
import pandas as pd
import io
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from app.services.eda_analytics import EDAAnalyticsService
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eda", tags=["EDA Analysis"])

# Results cache
eda_cache = {}

# Initialize EDA service
eda_service = EDAAnalyticsService()


# ==================== HELPER FUNCTIONS ====================

def _get_dataset_from_csv(csv_data: bytes) -> pd.DataFrame:
    """Load CSV data into DataFrame"""
    try:
        return pd.read_csv(io.BytesIO(csv_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")


# ==================== MAIN ANALYSIS ENDPOINTS ====================

@router.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = None,
    sample_rows: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze dataset - Complete EDA Analysis
    
    Runs all analyses:
    - Histogram (numeric features)
    - Categorical distributions
    - Missing patterns
    - Outlier detection
    - Correlations
    - Data quality assessment
    
    Returns: Complete analysis results with eda_id for retrieval
    """
    try:
        logger.info(f"Starting analysis for file: {file.filename}")
        
        # Read file
        contents = await file.read()
        df = _get_dataset_from_csv(contents)
        
        # Sample if needed
        if sample_rows and len(df) > sample_rows:
            df = df.sample(n=sample_rows, random_state=42)
            logger.info(f"Sampled to {sample_rows} rows")
        
        logger.info(f"Dataset shape: {df.shape}")
        
        # Validate target column if provided
        if target_column and target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        # Run complete analysis
        results = eda_service.analyze_dataset(df, target_column)
        
        # Cache results
        eda_id = str(uuid.uuid4())
        eda_cache[eda_id] = {
            "filename": file.filename,
            "results": results,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Analysis completed: {eda_id}")
        
        return {
            "eda_id": eda_id,
            "filename": file.filename,
            "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
            "quality_score": results['quality']['overall_score'],
            "assessment": results['quality']['assessment'],
            "timestamp": eda_cache[eda_id]['timestamp'],
            "message": "Analysis completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ==================== SPECIFIC ANALYSIS ENDPOINTS ====================

@router.post("/histogram")
async def histogram_analysis(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Histogram Analysis - Numeric Feature Distributions
    
    Analyzes all numeric features:
    - Mean, median, std, min, max
    - Quartiles (Q25, Q75)
    - Skewness and kurtosis
    - Distribution information
    """
    try:
        logger.info("Running histogram analysis")
        
        contents = await file.read()
        df = _get_dataset_from_csv(contents)
        
        results = eda_service._histogram_analysis(df)
        
        return {
            "analysis_type": "histogram",
            "total_numeric_features": len(results),
            "features": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Histogram analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/categorical")
async def categorical_analysis(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Categorical Analysis - Category Distributions
    
    Analyzes all categorical features:
    - Unique value counts
    - Mode and frequency
    - Top 10 values
    - Missing data
    """
    try:
        logger.info("Running categorical analysis")
        
        contents = await file.read()
        df = _get_dataset_from_csv(contents)
        
        results = eda_service._categorical_analysis(df)
        
        return {
            "analysis_type": "categorical",
            "total_categorical_features": len(results),
            "features": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Categorical analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/missing-pattern")
async def missing_pattern_analysis(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Missing Pattern Analysis - Data Completeness
    
    Analyzes missing data:
    - Total missing count and percentage
    - Missing per column
    - Rows with missing values
    - Fully missing columns
    """
    try:
        logger.info("Running missing pattern analysis")
        
        contents = await file.read()
        df = _get_dataset_from_csv(contents)
        
        results = eda_service._missing_pattern_analysis(df)
        
        return {
            "analysis_type": "missing_pattern",
            "total_missing": results['total_missing'],
            "total_missing_percentage": results['total_missing_percentage'],
            "rows_with_missing": results['rows_with_missing'],
            "fully_missing_columns": results['fully_missing_columns'],
            "missing_per_column": results['missing_per_column'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Missing pattern analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/outliers")
async def outlier_detection(
    file: UploadFile = File(...),
    method: str = Query("IQR", description="Detection method: IQR")
) -> Dict[str, Any]:
    """
    Outlier Detection - Anomaly Identification
    
    Detects outliers using IQR method:
    - Lower and upper bounds
    - Outlier count and percentage
    - Outlier indices (first 100)
    """
    try:
        logger.info(f"Running outlier detection with method: {method}")
        
        contents = await file.read()
        df = _get_dataset_from_csv(contents)
        
        results = eda_service._outlier_detection(df)
        
        total_outliers = sum(r.get('outlier_count', 0) for r in results.values())
        
        return {
            "analysis_type": "outliers",
            "method": method,
            "total_numeric_features": len(results),
            "total_outliers_detected": total_outliers,
            "features": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Outlier detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/correlation")
async def correlation_analysis(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Correlation Analysis - Feature Relationships
    
    Analyzes correlations:
    - Full correlation matrix
    - Strong correlations (>0.7)
    - Correlation strength classification
    """
    try:
        logger.info("Running correlation analysis")
        
        contents = await file.read()
        df = _get_dataset_from_csv(contents)
        
        results = eda_service._correlation_analysis(df)
        
        return {
            "analysis_type": "correlation",
            "total_strong_correlations": results.get('total_strong_correlations', 0),
            "strong_correlations": results.get('strong_correlations', []),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Correlation analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality")
async def data_quality_assessment(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Data Quality Assessment - Overall Quality Score
    
    Calculates quality metrics:
    - Overall quality score (0-100)
    - Assessment (Excellent/Good/Fair/Poor)
    - Completeness and uniqueness scores
    - Duplicate and missing percentages
    - Recommendations
    """
    try:
        logger.info("Running data quality assessment")
        
        contents = await file.read()
        df = _get_dataset_from_csv(contents)
        
        results = eda_service._calculate_data_quality(df)
        
        return {
            "analysis_type": "quality",
            "overall_score": results['overall_score'],
            "assessment": results['assessment'],
            "completeness": results['completeness'],
            "uniqueness": results['uniqueness'],
            "missing_percentage": results['missing_percentage'],
            "duplicate_percentage": results['duplicate_percentage'],
            "outlier_count": results['outlier_count'],
            "recommendations": results['recommendations'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Quality assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RETRIEVAL ENDPOINTS ====================

@router.get("/results/{eda_id}")
async def get_results(eda_id: str) -> Dict[str, Any]:
    """Get complete cached analysis results"""
    if eda_id not in eda_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    cached = eda_cache[eda_id]
    return {
        "eda_id": eda_id,
        "filename": cached["filename"],
        "shape": cached["shape"],
        "timestamp": cached["timestamp"],
        "results": cached["results"]
    }


@router.get("/results/{eda_id}/quality")
async def get_quality_results(eda_id: str) -> Dict[str, Any]:
    """Get quality assessment from cached results"""
    if eda_id not in eda_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    quality = eda_cache[eda_id]["results"]["quality"]
    
    return {
        "eda_id": eda_id,
        "overall_score": quality['overall_score'],
        "assessment": quality['assessment'],
        "completeness": quality['completeness'],
        "uniqueness": quality['uniqueness'],
        "recommendations": quality['recommendations']
    }


@router.get("/results/{eda_id}/histogram")
async def get_histogram_results(eda_id: str) -> Dict[str, Any]:
    """Get histogram analysis from cached results"""
    if eda_id not in eda_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    histogram = eda_cache[eda_id]["results"]["histogram"]
    
    return {
        "eda_id": eda_id,
        "total_numeric_features": len(histogram),
        "features": histogram
    }


@router.get("/results/{eda_id}/categorical")
async def get_categorical_results(eda_id: str) -> Dict[str, Any]:
    """Get categorical analysis from cached results"""
    if eda_id not in eda_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    categorical = eda_cache[eda_id]["results"]["categorical"]
    
    return {
        "eda_id": eda_id,
        "total_categorical_features": len(categorical),
        "features": categorical
    }


@router.get("/results/{eda_id}/outliers")
async def get_outliers_results(eda_id: str) -> Dict[str, Any]:
    """Get outlier detection from cached results"""
    if eda_id not in eda_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    outliers = eda_cache[eda_id]["results"]["outliers"]
    total_outliers = sum(r.get('outlier_count', 0) for r in outliers.values())
    
    return {
        "eda_id": eda_id,
        "total_outliers": total_outliers,
        "features": outliers
    }


@router.get("/results/{eda_id}/correlation")
async def get_correlation_results(eda_id: str) -> Dict[str, Any]:
    """Get correlation analysis from cached results"""
    if eda_id not in eda_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    correlations = eda_cache[eda_id]["results"]["correlations"]
    
    return {
        "eda_id": eda_id,
        "total_strong_correlations": correlations.get('total_strong_correlations', 0),
        "strong_correlations": correlations.get('strong_correlations', [])
    }


# ==================== UTILITY ENDPOINTS ====================

@router.get("/health")
async def health() -> Dict[str, Any]:
    """Check EDA service health"""
    return {
        "service": "eda",
        "status": "healthy",
        "cached_analyses": len(eda_cache),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/cache/clear")
async def clear_cache() -> Dict[str, str]:
    """Clear all cached analysis results"""
    count = len(eda_cache)
    eda_cache.clear()
    
    return {
        "message": f"Cleared {count} cached analyses",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/cache/status")
async def cache_status() -> Dict[str, Any]:
    """Get cache status"""
    return {
        "total_cached": len(eda_cache),
        "cached_ids": list(eda_cache.keys()),
        "timestamp": datetime.now().isoformat()
    }
