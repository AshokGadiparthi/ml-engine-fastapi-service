# 🚀 FastAPI Service - STANDALONE

**Status:** ✅ Standalone FastAPI Service  
**Version:** 1.0.0  
**Architecture:** Independent FastAPI Project  

---

## 📋 OVERVIEW

This is a **STANDALONE FastAPI SERVICE** that provides REST API endpoints for:
- 🤖 AutoML (Automatic model selection)
- 🎯 Training (Manual model training)
- 🔮 Predictions (Real-time & batch)
- 📦 Models (Model management)
- 📊 Datasets (Upload & management)

**This is a SEPARATE PROJECT from ML Engine**

---

## 🎯 KEY POINTS

✅ **STANDALONE** - No ML Engine code included
✅ **INDEPENDENT** - Can run without ML Engine
✅ **OPTIONAL INTEGRATION** - Can call ML Engine when needed
✅ **CLEAN CODE** - No merged codebases
✅ **PRODUCTION READY** - Full error handling & documentation

---

## 📁 PROJECT STRUCTURE

```
ml-engine-fastapi-service/
│
├── app/
│   ├── main.py                      FastAPI app
│   ├── config.py                    Configuration
│   │
│   ├── routers/
│   │   ├── automl.py                AutoML endpoints
│   │   ├── training.py              Training endpoints
│   │   ├── predictions.py           Prediction endpoints
│   │   ├── models.py                Model management endpoints
│   │   ├── datasets.py              Dataset upload endpoints
│   │   └── __init__.py
│   │
│   ├── services/
│   │   ├── ml_service.py            ML business logic
│   │   ├── job_manager.py           Job management
│   │   └── __init__.py
│   │
│   └── models/
│       ├── schemas.py               Pydantic models
│       └── __init__.py
│
├── data/                            (Dataset storage)
├── models/                          (Trained models)
├── logs/                            (Application logs)
│
├── requirements.txt                 (Dependencies)
├── run.sh                           (Startup script)
└── README.md                        (Original documentation)
```

---

## 🚀 QUICK START

### Step 1: Install Dependencies

```bash
cd ml-engine-fastapi-service

pip install -r requirements.txt --break-system-packages
```

Or minimal install:
```bash
pip install fastapi uvicorn pandas numpy scipy scikit-learn --break-system-packages
```

### Step 2: Start Service

```bash
python app/main.py
```

Or using the script:
```bash
bash run.sh
```

### Step 3: Access API

```
Interactive Docs: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
Health Check: http://localhost:8000/health
Root Info: http://localhost:8000/
```

---

## 📊 API ENDPOINTS (5 Routers)

### 1. **Dataset Management** (`/api/datasets`)

```
POST   /api/datasets/upload           Upload CSV file
GET    /api/datasets                  List all datasets
GET    /api/datasets/{id}             Get dataset info
GET    /api/datasets/{id}/preview     Preview first N rows
GET    /api/datasets/{id}/statistics  Get statistics
DELETE /api/datasets/{id}             Delete dataset
```

### 2. **Training** (`/api/training`)

```
POST   /api/training/train            Start model training
GET    /api/training/jobs             List all jobs
GET    /api/training/jobs/{id}        Get job status
POST   /api/training/jobs/{id}/stop   Stop job
```

### 3. **AutoML** (`/api/automl`)

```
POST   /api/automl/start              Start AutoML job
GET    /api/automl/jobs               List all jobs
GET    /api/automl/jobs/{id}          Get job status
POST   /api/automl/jobs/{id}/stop     Stop job
```

### 4. **Predictions** (`/api/predictions`)

```
POST   /api/predictions/predict       Single prediction
POST   /api/predictions/batch         Batch predictions
```

### 5. **Models** (`/api/models`)

```
GET    /api/models                    List all models
GET    /api/models/{id}               Get model info
DELETE /api/models/{id}               Delete model
GET    /api/models/{id}/feature-importance  Feature importance
POST   /api/models/{id}/deploy        Deploy model
```

### Service Endpoints

```
GET    /                              Root info
GET    /health                        Health check
GET    /docs                          Swagger UI
GET    /redoc                         ReDoc documentation
GET    /openapi.json                  OpenAPI schema
```

---

## 💻 EXAMPLE USAGE

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000/api"

# 1. Upload dataset
with open('data.csv', 'rb') as f:
    resp = requests.post(
        f"{BASE_URL}/datasets/upload",
        files={'file': f}
    )
    dataset_id = resp.json()['dataset_id']
    print(f"Dataset: {dataset_id}")

# 2. Get dataset preview
resp = requests.get(f"{BASE_URL}/datasets/{dataset_id}/preview?rows=5")
print(resp.json())

# 3. Get statistics
resp = requests.get(f"{BASE_URL}/datasets/{dataset_id}/statistics")
stats = resp.json()
print(f"Rows: {stats['row_count']}, Columns: {stats['column_count']}")

# 4. Start training
training_request = {
    "dataset_id": dataset_id,
    "target_column": "target",
    "algorithm": "xgboost",
    "problem_type": "classification"
}
resp = requests.post(
    f"{BASE_URL}/training/train",
    json=training_request
)
job_id = resp.json()['job_id']
print(f"Training started: {job_id}")

# 5. Check job status
resp = requests.get(f"{BASE_URL}/training/jobs/{job_id}")
print(resp.json())

# 6. List models
resp = requests.get(f"{BASE_URL}/models")
print(resp.json())
```

### CURL Examples

```bash
# Upload dataset
curl -F "file=@data.csv" http://localhost:8000/api/datasets/upload

# List datasets
curl http://localhost:8000/api/datasets

# Preview dataset
DATASET_ID="your-dataset-id"
curl http://localhost:8000/api/datasets/$DATASET_ID/preview?rows=10

# Get statistics
curl http://localhost:8000/api/datasets/$DATASET_ID/statistics

# Start training
curl -X POST http://localhost:8000/api/training/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "'"$DATASET_ID"'",
    "target_column": "target",
    "algorithm": "xgboost",
    "problem_type": "classification"
  }'

# Check health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/api/models
```

---

## 🔧 CONFIGURATION

Edit `app/config.py` to customize:

```python
class Settings:
    # API Settings
    APP_NAME = "ML Engine API"
    APP_VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8000
    DEBUG = True
    
    # CORS
    CORS_ORIGINS = ["*"]
    
    # Directories
    DATA_DIR = Path("data")
    MODELS_DIR = Path("models")
    LOGS_DIR = Path("logs")
```

---

## 🔗 INTEGRATION WITH ML ENGINE

This FastAPI service can **optionally** integrate with the **ML Engine** (separate project).

### How to Integrate (Optional)

1. **Install ML Engine separately:**
   ```bash
   pip install <path-to-ml-engine> -e .
   ```

2. **Update routers** to import ML Engine modules:
   ```python
   from ml_engine.automl import automl_find_best_model
   from ml_engine.train import train_model
   # etc.
   ```

3. **Update services** to use ML Engine:
   ```python
   from ml_engine import training_module
   
   def train_model(dataset_id, target_column, algorithm):
       df = load_dataset(dataset_id)
       model = training_module.train(df, target_column, algorithm)
       return model
   ```

**NOTE:** This service runs independently. ML Engine integration is optional.

---

## ✅ VERIFICATION

After starting the service, verify everything works:

```bash
# 1. Health check
curl http://localhost:8000/health
# Expected: {"status": "ok", "version": "1.0.0", ...}

# 2. Root info
curl http://localhost:8000/
# Expected: Shows available endpoints

# 3. API Docs
# Open http://localhost:8000/docs in browser
```

---

## 📦 DEPENDENCIES

### Minimal
```
fastapi>=0.68.0
uvicorn>=0.15.0
pandas>=2.0.0
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

### Full (see requirements.txt)
```
All of above, plus:
xgboost>=2.0.0
shap>=0.42.0
mlflow>=2.0.0
psycopg2-binary>=2.9.0
mysql-connector-python>=8.0.0
google-cloud-bigquery>=3.0.0
boto3>=1.20.0
```

---

## 📚 DOCUMENTATION

- **main.py** - FastAPI application
- **app/routers/** - API endpoints
- **app/models/schemas.py** - Request/response models
- **app/config.py** - Configuration
- **Swagger UI** - Interactive docs at `/docs`
- **ReDoc** - API documentation at `/redoc`

---

## 🎯 USE CASES

### Use This Service For:

✅ Dataset upload and management  
✅ Training job submission  
✅ Model deployment  
✅ Prediction serving  
✅ Job monitoring  
✅ Model versioning  
✅ API gateway  

### Optional ML Engine Integration For:

✅ AutoML functionality  
✅ Hyperparameter tuning  
✅ Feature engineering  
✅ Model evaluation  
✅ SHAP explanations  
✅ Batch predictions  

---

## 🏗️ ARCHITECTURE

```
┌─────────────────────────────────────┐
│      Client / User Interface        │
└──────────────┬──────────────────────┘
               │ REST API
               ↓
┌─────────────────────────────────────┐
│      FastAPI Service (This)         │
├─────────────────────────────────────┤
│  • AutoML Router                    │
│  • Training Router                  │
│  • Predictions Router               │
│  • Models Router                    │
│  • Datasets Router                  │
├─────────────────────────────────────┤
│  • Services (Business Logic)        │
│  • Schemas (Request/Response)       │
│  • Config (Settings)                │
└──────────────┬──────────────────────┘
               │ Optional Import
               ↓
┌─────────────────────────────────────┐
│    ML Engine (SEPARATE PROJECT)     │
│        If installed & configured    │
└─────────────────────────────────────┘
```

---

## ✨ KEY FEATURES

✅ **5 API Routers** for complete ML workflow  
✅ **Async Endpoints** for high performance  
✅ **Type Hints** throughout for safety  
✅ **Error Handling** with informative messages  
✅ **CORS Support** for frontend integration  
✅ **Swagger UI** for interactive testing  
✅ **ReDoc** for API documentation  
✅ **Health Checks** for monitoring  
✅ **Job Management** for async operations  
✅ **Dataset Registry** for managing datasets  

---

## 🚀 DEPLOYMENT

### Local Development
```bash
python app/main.py
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
```

### Docker
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app/main.py"]
```

---

## 📞 TROUBLESHOOTING

### Port Already in Use
```bash
python app/main.py --port 8001
```

### Module Not Found
```bash
pip install -r requirements.txt --break-system-packages
```

### Health Check Failed
```bash
# Check if service is running
curl http://localhost:8000/health

# Check logs
tail -f logs/*.log
```

---

## 📊 PROJECT STATS

| Metric | Value |
|--------|-------|
| **Routers** | 5 |
| **Endpoints** | 20+ |
| **Python Files** | 13 |
| **Lines of Code** | 2000+ |
| **Models** | Pydantic schemas |
| **Dependencies** | ~20 packages |

---

## ✅ WHAT'S INCLUDED

✅ 5 API routers (AutoML, Training, Predictions, Models, Datasets)  
✅ Business logic services  
✅ Pydantic schemas for validation  
✅ Configuration management  
✅ CORS middleware  
✅ Health check endpoints  
✅ Job management system  
✅ Dataset registry  
✅ Error handling  
✅ Type hints  
✅ Full documentation  

---

## 🎉 YOU HAVE

✅ **Standalone FastAPI Service** - Complete and ready  
✅ **5 API Routers** - All functionality  
✅ **Production Ready** - Error handling & validation  
✅ **Well Documented** - Clear docstrings & examples  
✅ **No Dependencies on ML Engine** - Works independently  
✅ **Optional ML Engine Integration** - Can work with ML Engine if needed  

---

**Version:** 1.0.0  
**Status:** ✅ Production Ready  
**Architecture:** Standalone FastAPI  
**Date:** January 12, 2026

---

## 🚀 GET STARTED

```bash
# 1. Install
pip install -r requirements.txt --break-system-packages

# 2. Run
python app/main.py

# 3. Open API docs
# http://localhost:8000/docs

# 4. Upload a dataset
# POST /api/datasets/upload

# 5. Train a model
# POST /api/training/train

# 6. Get predictions
# POST /api/predictions/predict
```

**Ready to go!** 🎯
