"""
ML Engine FastAPI Service
=========================

Real ML engine API wrapping ml_engine core modules.
Provides REST endpoints for:
- AutoML: Automatic algorithm selection
- Training: Manual model training
- Predictions: Real-time and batch predictions
- Models: Model management and deployment
- Datasets: Dataset upload and management
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import sys
import os

# Add ml_engine to path
ML_ENGINE_PATH = os.getenv("ML_ENGINE_PATH", "/home/claude/kedro-ml-engine/src")
if ML_ENGINE_PATH not in sys.path:
    sys.path.insert(0, ML_ENGINE_PATH)

from app.config import settings
from app.routers import (
    automl_router,
    training_router,
    predictions_router,
    models_router,
    datasets_router
)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="""
## ML Engine API

Complete machine learning pipeline API providing:

### 🤖 AutoML
- Automatic algorithm selection
- Cross-validation comparison
- Best model training

### 🎯 Training
- Manual algorithm selection
- Hyperparameter tuning
- Feature engineering

### 🔮 Predictions
- Real-time single predictions
- Batch predictions
- Probability scores

### 📦 Models
- Model management
- Deployment
- Feature importance

### 📊 Datasets
- File upload
- Preview & statistics
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(automl_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(predictions_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check if ml_engine is available
    ml_engine_status = "ok"
    try:
        from ml_engine.automl import automl_find_best_model
    except ImportError:
        ml_engine_status = "error: ml_engine not found"
    
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "ml_engine_status": ml_engine_status,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "automl": "/api/automl",
            "training": "/api/training",
            "predictions": "/api/predictions",
            "models": "/api/models",
            "datasets": "/api/datasets"
        }
    }


# Startup event
@app.on_event("startup")
async def startup():
    """Run on startup."""
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                     ML Engine API Started                          ║
╠═══════════════════════════════════════════════════════════════════╣
║  Version:    {settings.APP_VERSION:<52} ║
║  Docs:       http://localhost:{settings.PORT}/docs{' ' * 34}║
║  ML Engine:  {ML_ENGINE_PATH:<52} ║
╚═══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
