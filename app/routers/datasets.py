"""Datasets API Router."""
from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
import uuid
import shutil
from pathlib import Path

from app.models.schemas import DatasetInfo, DatasetUploadResponse
from app.config import settings

router = APIRouter(prefix="/datasets", tags=["Datasets"])


# In-memory dataset registry (in production, use database)
datasets_registry = {}


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset file (CSV).
    
    Returns dataset ID for use in training/AutoML.
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Generate ID and save file
    dataset_id = str(uuid.uuid4())
    file_path = settings.DATA_DIR / f"{dataset_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Read and analyze dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        # Clean up on error
        file_path.unlink()
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {str(e)}")
    
    # Register dataset
    datasets_registry[dataset_id] = {
        "dataset_id": dataset_id,
        "name": file.filename,
        "path": str(file_path),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    
    return DatasetUploadResponse(
        dataset_id=dataset_id,
        name=file.filename,
        path=str(file_path),
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        message="Dataset uploaded successfully"
    )


@router.get("/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str):
    """Get dataset information."""
    if dataset_id not in datasets_registry:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    info = datasets_registry[dataset_id]
    
    # Read sample
    df = pd.read_csv(info["path"], nrows=5)
    sample = df.to_dict(orient='records')
    
    return DatasetInfo(
        dataset_id=info["dataset_id"],
        name=info["name"],
        path=info["path"],
        row_count=info["row_count"],
        column_count=info["column_count"],
        columns=info["columns"],
        dtypes=info["dtypes"],
        sample=sample
    )


@router.get("")
async def list_datasets():
    """List all uploaded datasets."""
    return {
        "datasets": list(datasets_registry.values()),
        "total": len(datasets_registry)
    }


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    if dataset_id not in datasets_registry:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    info = datasets_registry[dataset_id]
    
    # Delete file
    file_path = Path(info["path"])
    if file_path.exists():
        file_path.unlink()
    
    del datasets_registry[dataset_id]
    
    return {"message": "Dataset deleted", "dataset_id": dataset_id}


@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, rows: int = 10):
    """Preview first N rows of dataset."""
    if dataset_id not in datasets_registry:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    info = datasets_registry[dataset_id]
    df = pd.read_csv(info["path"], nrows=rows)
    
    return {
        "dataset_id": dataset_id,
        "rows": rows,
        "data": df.to_dict(orient='records'),
        "columns": list(df.columns)
    }


@router.get("/{dataset_id}/statistics")
async def get_dataset_statistics(dataset_id: str):
    """Get statistical summary of dataset."""
    if dataset_id not in datasets_registry:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    info = datasets_registry[dataset_id]
    df = pd.read_csv(info["path"])
    
    # Numeric statistics
    numeric_stats = df.describe().to_dict()
    
    # Missing values
    missing = df.isnull().sum().to_dict()
    
    # Data types
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Unique values per column
    unique_counts = {col: int(df[col].nunique()) for col in df.columns}
    
    return {
        "dataset_id": dataset_id,
        "row_count": len(df),
        "column_count": len(df.columns),
        "numeric_statistics": numeric_stats,
        "missing_values": missing,
        "data_types": dtypes,
        "unique_counts": unique_counts
    }
