"""AutoML API Router."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from datetime import datetime
import shutil
import uuid
from pathlib import Path

from app.models.schemas import (
    AutoMLRequest, JobResponse, JobProgress, AutoMLResult,
    JobStatus, ProblemType
)
from app.services import job_manager, ml_service
from app.services.job_manager import JobStatus as JMStatus
from app.config import settings

router = APIRouter(prefix="/automl", tags=["AutoML"])


@router.post("/start", response_model=JobResponse)
async def start_automl(request: AutoMLRequest):
    """
    Start a new AutoML job.
    
    This will:
    1. Load and validate the dataset
    2. Apply feature engineering (if enabled)
    3. Test multiple algorithms with cross-validation
    4. Select the best performing model
    5. Train final model and save it
    """
    # Validate dataset path
    dataset_path = request.dataset_path
    if not dataset_path or not Path(dataset_path).exists():
        raise HTTPException(status_code=400, detail=f"Dataset not found: {dataset_path}")
    
    # Create job
    job = job_manager.create_job(
        job_type="automl",
        request_data=request.model_dump()
    )
    
    # Submit for execution
    job_manager.submit_job(job, ml_service.run_automl)
    
    return JobResponse(
        job_id=job.id,
        status=JobStatus.QUEUED,
        message="AutoML job started",
        created_at=job.created_at
    )


@router.post("/start-with-file", response_model=JobResponse)
async def start_automl_with_file(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    problem_type: ProblemType = Form(...),
    cv_folds: int = Form(5),
    use_feature_engineering: bool = Form(True),
    scaling_method: Optional[str] = Form("standard")
):
    """
    Start AutoML with file upload.
    """
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = settings.UPLOADS_DIR / f"{file_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create request
    request_data = {
        "dataset_id": file_id,
        "dataset_path": str(file_path),
        "target_column": target_column,
        "problem_type": problem_type.value,
        "cv_folds": cv_folds,
        "use_feature_engineering": use_feature_engineering,
        "scaling_method": scaling_method
    }
    
    # Create and submit job
    job = job_manager.create_job(job_type="automl", request_data=request_data)
    job_manager.submit_job(job, ml_service.run_automl)
    
    return JobResponse(
        job_id=job.id,
        status=JobStatus.QUEUED,
        message="AutoML job started with uploaded file",
        created_at=job.created_at
    )


@router.get("/jobs/{job_id}/progress", response_model=JobProgress)
async def get_job_progress(job_id: str):
    """Get AutoML job progress."""
    progress = job_manager.get_job_progress(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobProgress(**progress)


@router.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """Get AutoML job results (only available when completed)."""
    try:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Check status using string comparison
        if job.status.value != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job not completed. Current status: {job.status.value}"
            )
        
        if not job.result:
            raise HTTPException(status_code=500, detail="Results not available")
        
        # Build response manually to avoid any serialization issues
        response = {
            "job_id": str(job.id),
            "status": "completed",
            "problem_type": job.result.get("problem_type", "classification"),
            "best_algorithm": job.result.get("best_algorithm", "Unknown"),
            "best_score": float(job.result.get("best_score", 0)),
            "best_metric": job.result.get("best_metric", "Accuracy"),
            "leaderboard": job.result.get("leaderboard", []),
            "model_id": job.result.get("model_id", ""),
            "model_path": job.result.get("model_path", ""),
            "feature_engineer_path": job.result.get("feature_engineer_path"),
            "dataset_id": job.result.get("dataset_id", ""),
            "target_column": job.result.get("target_column", ""),
            "train_size": int(job.result.get("train_size", 0)),
            "test_size": int(job.result.get("test_size", 0)),
            "n_features": int(job.result.get("n_features", 0)),
            "feature_importance": job.result.get("feature_importance", []),
            "metrics": job.result.get("metrics", {}),
            "completed_at": job.result.get("completed_at", ""),
            "total_training_time_seconds": sum(
                r.get("training_time_seconds", 0) for r in job.result.get("leaderboard", [])
            )
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    """Stop a running AutoML job."""
    success = job_manager.stop_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not stop job")
    
    return {"message": "Stop requested", "job_id": job_id}


@router.get("/jobs")
async def list_jobs(status: Optional[str] = None):
    """List all AutoML jobs."""
    status_filter = None
    if status:
        try:
            status_filter = job_manager.JobStatus(status)
        except ValueError:
            pass
    
    jobs = job_manager.list_jobs(job_type="automl", status=status_filter)
    
    return {
        "jobs": [
            {
                "job_id": j.id,
                "status": j.status.value,
                "progress": j.progress,
                "current_best_algorithm": j.current_best_algorithm,
                "current_best_score": j.current_best_score,
                "created_at": j.created_at.isoformat(),
                "completed_at": j.completed_at.isoformat() if j.completed_at else None
            }
            for j in jobs
        ],
        "total": len(jobs)
    }
