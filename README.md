# ML Engine FastAPI Service

Real machine learning API that wraps the `ml_engine` core Python modules.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Architecture                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Angular UI  →  Spring Boot  →  FastAPI  →  ML-Engine Core    │
│   (Port 4200)    (Port 8080)     (Port 8000)   (Python)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment
export ML_ENGINE_PATH="/path/to/kedro-ml-engine/src"

# 3. Run server
chmod +x run.sh
./run.sh

# Or directly:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Health Check
```
GET /health
GET /
```

### AutoML
```
POST /api/automl/start              # Start AutoML job
POST /api/automl/start-with-file    # Start with file upload
GET  /api/automl/jobs/{job_id}/progress
GET  /api/automl/jobs/{job_id}/results
POST /api/automl/jobs/{job_id}/stop
GET  /api/automl/jobs
```

### Manual Training
```
POST /api/training/start            # Start training job
POST /api/training/start-with-file  # Start with file upload
GET  /api/training/jobs/{job_id}/progress
GET  /api/training/jobs/{job_id}/results
POST /api/training/jobs/{job_id}/stop
GET  /api/training/jobs
GET  /api/training/algorithms       # List available algorithms
```

### Predictions
```
POST /api/predictions/realtime/{model_id}  # Single prediction
POST /api/predictions/batch/{model_id}     # Batch predictions
POST /api/predictions/explain/{model_id}   # Explain prediction
```

### Models
```
GET    /api/models                  # List all models
GET    /api/models/{model_id}       # Get model details
DELETE /api/models/{model_id}       # Delete model
POST   /api/models/{model_id}/deploy
POST   /api/models/{model_id}/undeploy
GET    /api/models/{model_id}/feature-importance
```

### Datasets
```
POST   /api/datasets/upload         # Upload dataset
GET    /api/datasets                # List datasets
GET    /api/datasets/{dataset_id}   # Get dataset info
DELETE /api/datasets/{dataset_id}   # Delete dataset
GET    /api/datasets/{dataset_id}/preview
GET    /api/datasets/{dataset_id}/statistics
```

## Usage Examples

### 1. AutoML with File Upload

```bash
curl -X POST "http://localhost:8000/api/automl/start-with-file" \
  -F "file=@data.csv" \
  -F "target_column=churn" \
  -F "problem_type=classification" \
  -F "cv_folds=5"
```

### 2. Check Job Progress

```bash
curl "http://localhost:8000/api/automl/jobs/{job_id}/progress"
```

### 3. Get Results

```bash
curl "http://localhost:8000/api/automl/jobs/{job_id}/results"
```

### 4. Make Prediction

```bash
curl -X POST "http://localhost:8000/api/predictions/realtime/{model_id}" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "monthly_charges": 50.0, "contract": "Month-to-month"}'
```

## Spring Boot Integration

Configure in `application.yml`:

```yaml
ml-engine:
  api:
    base-url: http://localhost:8000/api
    timeout: 30000
```

Service call example:

```java
@Service
public class MLEngineClient {
    
    private final RestTemplate restTemplate;
    private final String baseUrl;
    
    public AutoMLResponse startAutoML(AutoMLRequest request) {
        return restTemplate.postForObject(
            baseUrl + "/automl/start",
            request,
            AutoMLResponse.class
        );
    }
    
    public JobProgress getProgress(String jobId) {
        return restTemplate.getForObject(
            baseUrl + "/automl/jobs/" + jobId + "/progress",
            JobProgress.class
        );
    }
}
```

## Project Structure

```
ml-engine-api/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py    # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── job_manager.py    # Background job execution
│   │   └── ml_service.py     # ML operations wrapper
│   └── routers/
│       ├── __init__.py
│       ├── automl.py         # AutoML endpoints
│       ├── training.py       # Training endpoints
│       ├── predictions.py    # Prediction endpoints
│       ├── models.py         # Model management
│       └── datasets.py       # Dataset management
├── data/                 # Uploaded datasets
├── models/               # Trained models
├── logs/                 # Application logs
├── uploads/              # Temporary uploads
├── requirements.txt
├── run.sh
└── README.md
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| ML_ENGINE_PATH | Path to ml_engine source | /home/claude/kedro-ml-engine/src |
| HOST | Server host | 0.0.0.0 |
| PORT | Server port | 8000 |
| DEBUG | Enable debug mode | True |

## API Documentation

Interactive documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
