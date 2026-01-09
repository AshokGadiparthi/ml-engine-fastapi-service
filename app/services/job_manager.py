"""Job Manager - Handles background job execution and tracking."""
import uuid
import threading
import traceback
from datetime import datetime
from typing import Dict, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class Job:
    """Represents a background job."""
    id: str
    job_type: str  # "training", "automl", "prediction", "evaluation"
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    current_phase: str = ""
    current_algorithm: str = ""
    phases: list = field(default_factory=list)
    algorithms_completed: int = 0
    algorithms_total: int = 0
    current_best_score: Optional[float] = None
    current_best_algorithm: Optional[str] = None
    logs: list = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    request_data: Dict[str, Any] = field(default_factory=dict)
    
    # Internal
    _stop_requested: bool = False


class JobManager:
    """
    Manages background job execution with thread pool.
    Tracks job status, progress, and results.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.jobs: Dict[str, Job] = {}
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_JOBS)
        self._initialized = True
        print(f"✅ JobManager initialized with {settings.MAX_CONCURRENT_JOBS} workers")
    
    def create_job(self, job_type: str, request_data: Dict[str, Any]) -> Job:
        """Create a new job and return it."""
        job_id = str(uuid.uuid4())
        
        job = Job(
            id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            request_data=request_data
        )
        
        self.jobs[job_id] = job
        self._add_log(job, "INFO", f"Job created: {job_type}")
        
        return job
    
    def submit_job(self, job: Job, task_func: Callable, *args, **kwargs) -> str:
        """Submit a job for execution."""
        job.status = JobStatus.QUEUED
        self._add_log(job, "INFO", "Job queued for execution")
        
        def wrapper():
            try:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()
                self._add_log(job, "INFO", "Job started")
                
                # Execute the task
                result = task_func(job, *args, **kwargs)
                
                if job._stop_requested:
                    job.status = JobStatus.STOPPED
                    self._add_log(job, "WARN", "Job stopped by user")
                else:
                    job.status = JobStatus.COMPLETED
                    job.result = result
                    job.progress = 100
                    self._add_log(job, "INFO", "Job completed successfully")
                
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                self._add_log(job, "ERROR", f"Job failed: {str(e)}")
                print(f"❌ Job {job.id} failed: {traceback.format_exc()}")
            
            finally:
                job.completed_at = datetime.now()
        
        self.executor.submit(wrapper)
        return job.id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def get_job_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job progress as dict."""
        job = self.get_job(job_id)
        if not job:
            return None
        
        elapsed = 0
        if job.started_at:
            elapsed = int((datetime.now() - job.started_at).total_seconds())
        
        return {
            "job_id": job.id,
            "status": job.status.value,
            "progress": job.progress,
            "current_phase": job.current_phase,
            "current_algorithm": job.current_algorithm,
            "phases": job.phases,
            "algorithms_completed": job.algorithms_completed,
            "algorithms_total": job.algorithms_total,
            "current_best_score": job.current_best_score,
            "current_best_algorithm": job.current_best_algorithm,
            "elapsed_seconds": elapsed,
            "logs": job.logs[-50:],  # Last 50 logs
            "error_message": job.error_message
        }
    
    def stop_job(self, job_id: str) -> bool:
        """Request job to stop."""
        job = self.get_job(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING]:
            job._stop_requested = True
            self._add_log(job, "WARN", "Stop requested")
            return True
        
        return False
    
    def update_progress(
        self, 
        job: Job, 
        progress: int = None,
        phase: str = None,
        algorithm: str = None,
        best_score: float = None,
        best_algorithm: str = None
    ):
        """Update job progress."""
        if progress is not None:
            job.progress = min(100, max(0, progress))
        if phase is not None:
            job.current_phase = phase
        if algorithm is not None:
            job.current_algorithm = algorithm
        if best_score is not None:
            job.current_best_score = best_score
        if best_algorithm is not None:
            job.current_best_algorithm = best_algorithm
    
    def update_phase(self, job: Job, phase_name: str, status: str, message: str = None):
        """Update or add phase info."""
        for phase in job.phases:
            if phase["name"] == phase_name:
                phase["status"] = status
                if message:
                    phase["message"] = message
                return
        
        # Add new phase
        job.phases.append({
            "name": phase_name,
            "label": phase_name.replace("_", " ").title(),
            "status": status,
            "progress": 0,
            "message": message
        })
    
    def add_algorithm_result(self, job: Job, algorithm: str, score: float, time_seconds: float):
        """Track algorithm completion."""
        job.algorithms_completed += 1
        self._add_log(job, "INFO", f"Algorithm {algorithm}: {score:.4f} ({time_seconds:.1f}s)")
        
        if job.current_best_score is None or score > job.current_best_score:
            job.current_best_score = score
            job.current_best_algorithm = algorithm
            self._add_log(job, "INFO", f"🏆 New best: {algorithm} ({score:.4f})")
    
    def is_stop_requested(self, job: Job) -> bool:
        """Check if job should stop."""
        return job._stop_requested
    
    def list_jobs(self, job_type: str = None, status: JobStatus = None) -> list:
        """List jobs with optional filtering."""
        jobs = list(self.jobs.values())
        
        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return jobs
    
    def _add_log(self, job: Job, level: str, message: str):
        """Add log entry to job."""
        job.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        })


# Global instance
job_manager = JobManager()
