"""Training API Router."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from datetime import datetime
import shutil
import uuid
from pathlib import Path

from app.models.schemas import (
    TrainingRequest, TrainingConfig, JobResponse, JobProgress, TrainingResult,
    JobStatus, ProblemType
)
from app.services import job_manager, ml_service
from app.services.job_manager import JobStatus as JMJobStatus
from app.config import settings

router = APIRouter(prefix="/training", tags=["Training"])


@router.post("/start", response_model=JobResponse)
async def start_training(request: TrainingRequest):
    """
    Start a manual training job with specific algorithm.
    
    Use this when you want to train a specific algorithm
    with specific hyperparameters (vs AutoML which tries all).
    """
    # Validate dataset - either dataset_path or dataset_id must be provided
    dataset_path = request.dataset_path
    if not dataset_path and request.dataset_id:
        # Try to find dataset by ID in uploads
        dataset_path = str(settings.UPLOADS_DIR / f"{request.dataset_id}.csv")
        if not Path(dataset_path).exists():
            # Try without .csv extension
            for ext in ['', '.csv', '.xlsx', '.parquet']:
                potential_path = str(settings.UPLOADS_DIR / f"{request.dataset_id}{ext}")
                if Path(potential_path).exists():
                    dataset_path = potential_path
                    break
    
    if not dataset_path or not Path(dataset_path).exists():
        raise HTTPException(status_code=400, detail=f"Dataset not found: {dataset_path or request.dataset_id}")
    
    # Normalize algorithm name
    algorithm = request.algorithm.lower().strip() if request.algorithm else "random_forest"
    problem_type = request.problem_type.lower().strip() if request.problem_type else "classification"
    
    # Build request data
    request_data = {
        "dataset_id": request.dataset_id or str(uuid.uuid4()),
        "dataset_path": dataset_path,
        "target_column": request.target_column,
        "algorithm": algorithm,
        "problem_type": problem_type,
        "experiment_name": request.experiment_name,
        "config": request.config or {}
    }
    
    # Create job
    job = job_manager.create_job(
        job_type="training",
        request_data=request_data
    )
    
    # Submit for execution
    job_manager.submit_job(job, ml_service.run_training)
    
    return JobResponse(
        job_id=job.id,
        status=JobStatus.QUEUED,
        message=f"Training job started with {algorithm}",
        created_at=job.created_at
    )


@router.post("/start-with-file", response_model=JobResponse)
async def start_training_with_file(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    algorithm: str = Form(...),
    problem_type: str = Form(...),
    use_feature_engineering: bool = Form(False),
    scaling_method: Optional[str] = Form("standard"),
    cv_folds: int = Form(5),
    tune_hyperparameters: bool = Form(False)
):
    """
    Start training with file upload.
    """
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = settings.UPLOADS_DIR / f"{file_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Normalize values
    algorithm = algorithm.lower().strip()
    problem_type = problem_type.lower().strip()
    
    # Create request
    request_data = {
        "dataset_id": file_id,
        "dataset_path": str(file_path),
        "target_column": target_column,
        "algorithm": algorithm,
        "problem_type": problem_type,
        "config": {
            "use_feature_engineering": use_feature_engineering,
            "scaling_method": scaling_method,
            "cv_folds": cv_folds,
            "tune_hyperparameters": tune_hyperparameters
        }
    }
    
    # Create and submit job
    job = job_manager.create_job(job_type="training", request_data=request_data)
    job_manager.submit_job(job, ml_service.run_training)
    
    return JobResponse(
        job_id=job.id,
        status=JobStatus.QUEUED,
        message=f"Training job started with {algorithm}",
        created_at=job.created_at
    )


@router.get("/jobs/{job_id}/progress", response_model=JobProgress)
async def get_job_progress(job_id: str):
    """Get training job progress."""
    progress = job_manager.get_job_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobProgress(**progress)


@router.get("/jobs/{job_id}/results", response_model=TrainingResult)
async def get_job_results(job_id: str):
    """Get training job results."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JMJobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.status.value}"
        )
    
    if not job.result:
        raise HTTPException(status_code=500, detail="Results not available")
    
    r = job.result
    return TrainingResult(
        job_id=r["job_id"],
        status=JobStatus.COMPLETED,
        problem_type=ProblemType(r["problem_type"]),
        algorithm=r["algorithm"],
        train_score=r["train_score"],
        test_score=r["test_score"],
        cv_score=r.get("cv_score"),
        cv_std=r.get("cv_std"),
        metrics={},
        model_id=r["model_id"],
        model_path=r["model_path"],
        feature_engineer_path=r.get("feature_engineer_path"),
        feature_names=r["feature_names"],
        n_features=r["n_features"],
        dataset_id=r.get("dataset_id", ""),
        target_column=r["target_column"],
        train_size=r["train_size"],
        test_size=r["test_size"],
        feature_importance=r["feature_importance"],
        training_time_seconds=r["training_time_seconds"],
        completed_at=datetime.fromisoformat(r["completed_at"])
    )


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    """Stop a running training job."""
    success = job_manager.stop_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not stop job")
    
    return {"message": "Stop requested", "job_id": job_id}


@router.get("/jobs")
async def list_jobs(status: Optional[str] = None):
    """List all training jobs."""
    status_filter = None
    if status:
        try:
            status_filter = job_manager.JobStatus(status)
        except ValueError:
            pass
    
    jobs = job_manager.list_jobs(job_type="training", status=status_filter)
    
    return {
        "jobs": [
            {
                "job_id": j.id,
                "status": j.status.value,
                "progress": j.progress,
                "algorithm": j.request_data.get("algorithm"),
                "created_at": j.created_at.isoformat(),
                "completed_at": j.completed_at.isoformat() if j.completed_at else None
            }
            for j in jobs
        ],
        "total": len(jobs)
    }


@router.get("/algorithms")
async def list_algorithms():
    """List available algorithms."""
    return {
        "classification": [
            {"id": "logistic", "name": "Logistic Regression", "description": "Simple and interpretable"},
            {"id": "random_forest", "name": "Random Forest", "description": "Ensemble of decision trees"},
            {"id": "xgboost", "name": "XGBoost", "description": "Gradient boosting, often best performer"},
            {"id": "svm", "name": "SVM", "description": "Support Vector Machine"},
            {"id": "gradient_boosting", "name": "Gradient Boosting", "description": "Sequential ensemble"}
        ],
        "regression": [
            {"id": "linear", "name": "Linear Regression", "description": "Simple linear model"},
            {"id": "ridge", "name": "Ridge Regression", "description": "L2 regularization"},
            {"id": "lasso", "name": "Lasso Regression", "description": "L1 regularization"},
            {"id": "random_forest", "name": "Random Forest", "description": "Ensemble of decision trees"},
            {"id": "xgboost", "name": "XGBoost", "description": "Gradient boosting"},
            {"id": "gradient_boosting", "name": "Gradient Boosting", "description": "Sequential ensemble"},
            {"id": "svr", "name": "SVR", "description": "Support Vector Regression"}
        ]
    }
