"""ML Service - Wraps ml_engine core functions for FastAPI."""
import sys
import os
import uuid
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Add ml_engine to path
ML_ENGINE_PATH = os.getenv("ML_ENGINE_PATH", "/home/claude/kedro-ml-engine/src")
if ML_ENGINE_PATH not in sys.path:
    sys.path.insert(0, ML_ENGINE_PATH)

# Now import ml_engine modules
from ml_engine.automl import automl_find_best_model, get_classification_algorithms, get_regression_algorithms
from ml_engine.train import train_model
from ml_engine.feature_engineering import FeatureEngineer
from ml_engine.evaluation import cross_validate_model, plot_confusion_matrix, plot_roc_curve
from ml_engine.explainability import explain_model_shap

from app.config import settings
from app.services.job_manager import Job, job_manager


class MLService:
    """
    Service that wraps ml_engine core functions.
    Provides methods for training, AutoML, prediction, evaluation.
    """
    
    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        self.models_registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load model registry from disk."""
        registry_path = self.models_dir / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, "r") as f:
                    self.models_registry = json.load(f)
                print(f"✅ Loaded {len(self.models_registry)} models from registry")
            except Exception as e:
                print(f"⚠️ Could not load registry: {e}")
                self.models_registry = {}
    
    def _save_registry(self):
        """Save model registry to disk."""
        registry_path = self.models_dir / "registry.json"
        with open(registry_path, "w") as f:
            json.dump(self.models_registry, f, indent=2, default=str)
    
    def _register_model(
        self, 
        model_id: str, 
        name: str,
        algorithm: str,
        problem_type: str,
        score: float,
        metric: str,
        model_path: str,
        feature_engineer_path: str = None,
        feature_names: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Register a trained model."""
        self.models_registry[model_id] = {
            "model_id": model_id,
            "name": name,
            "algorithm": algorithm,
            "problem_type": problem_type,
            "score": score,
            "metric": metric,
            "model_path": model_path,
            "feature_engineer_path": feature_engineer_path,
            "feature_names": feature_names or [],
            "created_at": datetime.now().isoformat(),
            "is_deployed": False,
            "metadata": metadata or {}
        }
        self._save_registry()
        return self.models_registry[model_id]
    
    # ============ AUTOML ============
    
    def run_automl(self, job: Job) -> Dict[str, Any]:
        """
        Run AutoML job - finds best algorithm automatically.
        
        This is called by JobManager in a background thread.
        """
        request = job.request_data
        dataset_path = request.get("dataset_path")
        target_column = request["target_column"]
        problem_type = request["problem_type"]
        cv_folds = request.get("cv_folds", 5)
        use_fe = request.get("use_feature_engineering", True)
        scaling_method = request.get("scaling_method", "standard")
        
        # Initialize phases
        phases = [
            ("data_loading", "Data Loading"),
            ("data_validation", "Data Validation"),
            ("feature_engineering", "Feature Engineering"),
            ("algorithm_selection", "Algorithm Selection"),
            ("model_training", "Model Training"),
            ("evaluation", "Evaluation")
        ]
        for name, label in phases:
            job_manager.update_phase(job, name, "pending")
        
        job.algorithms_total = 5 if problem_type == "classification" else 7
        
        # ===== PHASE 1: Data Loading =====
        job_manager.update_phase(job, "data_loading", "running")
        job_manager.update_progress(job, progress=5, phase="Data Loading")
        job_manager._add_log(job, "INFO", f"Loading dataset from {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        job_manager._add_log(job, "INFO", f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        job_manager.update_phase(job, "data_loading", "completed")
        
        if job_manager.is_stop_requested(job):
            return {"status": "stopped"}
        
        # ===== PHASE 2: Data Validation =====
        job_manager.update_phase(job, "data_validation", "running")
        job_manager.update_progress(job, progress=10, phase="Data Validation")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical columns
        X = pd.get_dummies(X, drop_first=True)
        job_manager._add_log(job, "INFO", f"Features after encoding: {X.shape[1]}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        job_manager._add_log(job, "INFO", f"Train: {len(X_train)}, Test: {len(X_test)}")
        job_manager.update_phase(job, "data_validation", "completed")
        
        if job_manager.is_stop_requested(job):
            return {"status": "stopped"}
        
        # ===== PHASE 3: Feature Engineering =====
        job_manager.update_phase(job, "feature_engineering", "running")
        job_manager.update_progress(job, progress=20, phase="Feature Engineering")
        
        feature_engineer = None
        if use_fe:
            job_manager._add_log(job, "INFO", f"Applying feature engineering (scaling: {scaling_method})")
            feature_engineer = FeatureEngineer(scaling_method=scaling_method)
            X_train = feature_engineer.fit_transform(X_train, y_train)
            X_test = feature_engineer.transform(X_test)
            job_manager._add_log(job, "INFO", f"Features after engineering: {X_train.shape[1]}")
        
        job_manager.update_phase(job, "feature_engineering", "completed")
        
        if job_manager.is_stop_requested(job):
            return {"status": "stopped"}
        
        # ===== PHASE 4: Algorithm Selection =====
        job_manager.update_phase(job, "algorithm_selection", "running")
        job_manager.update_progress(job, progress=30, phase="Algorithm Selection")
        
        # Get algorithms
        if problem_type == "classification":
            algorithms = get_classification_algorithms()
            scoring = "accuracy"
        else:
            algorithms = get_regression_algorithms()
            scoring = "r2"
        
        job.algorithms_total = len(algorithms)
        job_manager._add_log(job, "INFO", f"Testing {len(algorithms)} algorithms...")
        
        # Test each algorithm
        leaderboard = []
        best_model = None
        best_score = -np.inf
        best_algorithm = None
        
        from sklearn.model_selection import cross_val_score
        import time
        
        for idx, (algo_name, model) in enumerate(algorithms.items()):
            if job_manager.is_stop_requested(job):
                return {"status": "stopped", "partial_results": leaderboard}
            
            # Update progress
            progress = 30 + int((idx / len(algorithms)) * 50)
            job_manager.update_progress(job, progress=progress, algorithm=algo_name)
            job_manager._add_log(job, "INFO", f"Testing {algo_name}...")
            
            start_time = time.time()
            try:
                scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=-1)
                mean_score = scores.mean()
                std_score = scores.std()
                elapsed = time.time() - start_time
                
                result = {
                    "rank": 0,  # Will be set later
                    "algorithm": algo_name,
                    "score": round(mean_score, 4),
                    "std": round(std_score, 4),
                    "training_time_seconds": round(elapsed, 2)
                }
                
                # Add metrics based on problem type
                if problem_type == "classification":
                    result["accuracy"] = round(mean_score, 4)
                else:
                    result["r2"] = round(mean_score, 4)
                
                leaderboard.append(result)
                job_manager.add_algorithm_result(job, algo_name, mean_score, elapsed)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_algorithm = algo_name
                    best_model = model
                
            except Exception as e:
                job_manager._add_log(job, "ERROR", f"Failed {algo_name}: {str(e)}")
        
        # Sort leaderboard and assign ranks
        leaderboard.sort(key=lambda x: x["score"], reverse=True)
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1
        
        job_manager.update_phase(job, "algorithm_selection", "completed")
        
        # ===== PHASE 5: Model Training =====
        job_manager.update_phase(job, "model_training", "running")
        job_manager.update_progress(job, progress=85, phase="Model Training")
        job_manager._add_log(job, "INFO", f"Training best model: {best_algorithm}")
        
        # Train best model on full training data
        best_model.fit(X_train, y_train)
        
        # Generate model ID and save
        model_id = str(uuid.uuid4())
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "model.pkl"
        joblib.dump(best_model, model_path)
        job_manager._add_log(job, "INFO", f"Model saved: {model_path}")
        
        # Save feature engineer
        fe_path = None
        if feature_engineer:
            fe_path = model_dir / "feature_engineer.pkl"
            feature_engineer.save(str(fe_path))
        
        # Save feature names
        feature_names = list(X_train.columns)
        fn_path = model_dir / "feature_names.pkl"
        joblib.dump(feature_names, fn_path)
        
        job_manager.update_phase(job, "model_training", "completed")
        
        # ===== PHASE 6: Evaluation =====
        job_manager.update_phase(job, "evaluation", "running")
        job_manager.update_progress(job, progress=95, phase="Evaluation")
        
        # Get test predictions
        test_pred = best_model.predict(X_test)
        
        # Calculate feature importance
        feature_importance = self._get_feature_importance(best_model, feature_names)
        
        # Final metrics
        if problem_type == "classification":
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            final_metrics = {
                "accuracy": round(accuracy_score(y_test, test_pred), 4),
                "precision": round(precision_score(y_test, test_pred, average='weighted', zero_division=0), 4),
                "recall": round(recall_score(y_test, test_pred, average='weighted', zero_division=0), 4),
                "f1_score": round(f1_score(y_test, test_pred, average='weighted', zero_division=0), 4)
            }
            best_metric = "Accuracy"
        else:
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            final_metrics = {
                "r2": round(r2_score(y_test, test_pred), 4),
                "mae": round(mean_absolute_error(y_test, test_pred), 4),
                "mse": round(mean_squared_error(y_test, test_pred), 4),
                "rmse": round(np.sqrt(mean_squared_error(y_test, test_pred)), 4)
            }
            best_metric = "R²"
        
        job_manager._add_log(job, "INFO", f"Final test {best_metric}: {best_score:.4f}")
        job_manager.update_phase(job, "evaluation", "completed")
        job_manager.update_progress(job, progress=100)
        
        # Register model
        self._register_model(
            model_id=model_id,
            name=f"AutoML - {best_algorithm}",
            algorithm=best_algorithm,
            problem_type=problem_type,
            score=best_score,
            metric=best_metric,
            model_path=str(model_path),
            feature_engineer_path=str(fe_path) if fe_path else None,
            feature_names=feature_names,
            metadata={
                "cv_folds": cv_folds,
                "use_feature_engineering": use_fe,
                "dataset_path": dataset_path,
                "target_column": target_column
            }
        )
        
        # Build result
        result = {
            "job_id": job.id,
            "status": "completed",
            "problem_type": problem_type,
            "best_algorithm": best_algorithm,
            "best_score": best_score,
            "best_metric": best_metric,
            "leaderboard": leaderboard,
            "model_id": model_id,
            "model_path": str(model_path),
            "feature_engineer_path": str(fe_path) if fe_path else None,
            "dataset_id": request.get("dataset_id"),
            "target_column": target_column,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": len(feature_names),
            "feature_importance": feature_importance,
            "metrics": final_metrics,
            "completed_at": datetime.now().isoformat()
        }
        
        return result
    
    # ============ MANUAL TRAINING ============
    
    def run_training(self, job: Job) -> Dict[str, Any]:
        """
        Run manual training job with specific algorithm.
        """
        request = job.request_data
        dataset_path = request.get("dataset_path")
        target_column = request["target_column"]
        algorithm = request["algorithm"]
        problem_type = request["problem_type"]
        config = request.get("config", {})
        
        # Initialize phases
        for name in ["data_loading", "feature_engineering", "training", "evaluation"]:
            job_manager.update_phase(job, name, "pending")
        
        job.algorithms_total = 1
        
        # ===== DATA LOADING =====
        job_manager.update_phase(job, "data_loading", "running")
        job_manager.update_progress(job, progress=10, phase="Data Loading")
        
        df = pd.read_csv(dataset_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X = pd.get_dummies(X, drop_first=True)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        job_manager._add_log(job, "INFO", f"Data loaded: {len(df)} rows")
        job_manager.update_phase(job, "data_loading", "completed")
        
        # ===== FEATURE ENGINEERING =====
        job_manager.update_phase(job, "feature_engineering", "running")
        job_manager.update_progress(job, progress=25, phase="Feature Engineering")
        
        feature_engineer = None
        if config.get("use_feature_engineering", False):
            scaling = config.get("scaling_method", "standard")
            feature_engineer = FeatureEngineer(scaling_method=scaling)
            X_train = feature_engineer.fit_transform(X_train, y_train)
            X_test = feature_engineer.transform(X_test)
            job_manager._add_log(job, "INFO", f"Feature engineering applied")
        
        job_manager.update_phase(job, "feature_engineering", "completed")
        
        # ===== TRAINING =====
        job_manager.update_phase(job, "training", "running")
        job_manager.update_progress(job, progress=50, phase="Training", algorithm=algorithm)
        job_manager._add_log(job, "INFO", f"Training {algorithm}...")
        
        import time
        start_time = time.time()
        
        # Get model based on algorithm
        model = self._get_model(algorithm, problem_type)
        
        # Hyperparameter tuning if enabled
        if config.get("tune_hyperparameters", False):
            job_manager._add_log(job, "INFO", "Running hyperparameter tuning...")
            from ml_engine.hyperparameter_tuning import tune_model
            tuning_results = tune_model(
                algorithm=algorithm,
                X=X_train,
                y=y_train,
                search_type=config.get("tuning_search_type", "grid"),
                cv=config.get("cv_folds", 5)
            )
            if tuning_results:
                model = tuning_results["best_model"]
                job_manager._add_log(job, "INFO", f"Best params: {tuning_results['best_params']}")
        
        # Train model
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        job_manager.add_algorithm_result(job, algorithm, 0, training_time)
        job_manager.update_phase(job, "training", "completed")
        
        # ===== EVALUATION =====
        job_manager.update_phase(job, "evaluation", "running")
        job_manager.update_progress(job, progress=80, phase="Evaluation")
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        if problem_type == "classification":
            from sklearn.metrics import accuracy_score
            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            metric = "Accuracy"
        else:
            from sklearn.metrics import r2_score
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)
            metric = "R²"
        
        # Cross-validation
        cv_score = None
        cv_std = None
        if config.get("cv_folds", 0) > 0:
            from sklearn.model_selection import cross_val_score
            scoring = "accuracy" if problem_type == "classification" else "r2"
            cv_scores = cross_val_score(model, X_train, y_train, cv=config["cv_folds"], scoring=scoring)
            cv_score = cv_scores.mean()
            cv_std = cv_scores.std()
        
        job_manager._add_log(job, "INFO", f"Train {metric}: {train_score:.4f}, Test: {test_score:.4f}")
        
        # Save model
        model_id = str(uuid.uuid4())
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        fe_path = None
        if feature_engineer:
            fe_path = model_dir / "feature_engineer.pkl"
            feature_engineer.save(str(fe_path))
        
        feature_names = list(X_train.columns)
        joblib.dump(feature_names, model_dir / "feature_names.pkl")
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, feature_names)
        
        job_manager.update_phase(job, "evaluation", "completed")
        job_manager.update_progress(job, progress=100)
        
        # Register model
        self._register_model(
            model_id=model_id,
            name=f"{algorithm} - {target_column}",
            algorithm=algorithm,
            problem_type=problem_type,
            score=test_score,
            metric=metric,
            model_path=str(model_path),
            feature_engineer_path=str(fe_path) if fe_path else None,
            feature_names=feature_names
        )
        
        return {
            "job_id": job.id,
            "status": "completed",
            "problem_type": problem_type,
            "algorithm": algorithm,
            "train_score": round(train_score, 4),
            "test_score": round(test_score, 4),
            "cv_score": round(cv_score, 4) if cv_score else None,
            "cv_std": round(cv_std, 4) if cv_std else None,
            "model_id": model_id,
            "model_path": str(model_path),
            "feature_engineer_path": str(fe_path) if fe_path else None,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "dataset_id": request.get("dataset_id"),
            "target_column": target_column,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_importance": feature_importance,
            "training_time_seconds": round(training_time, 2),
            "completed_at": datetime.now().isoformat()
        }
    
    # ============ PREDICTIONS ============
    
    def predict(self, model_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make single prediction."""
        model_info = self.models_registry.get(model_id)
        if not model_info:
            raise ValueError(f"Model not found: {model_id}")
        
        # Load model
        model = joblib.load(model_info["model_path"])
        
        # Load feature names
        feature_names = model_info.get("feature_names", [])
        
        # Prepare features
        X = pd.DataFrame([features])
        X = pd.get_dummies(X, drop_first=True)
        
        # Ensure same features as training
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names] if feature_names else X
        
        # Load feature engineer if exists
        fe_path = model_info.get("feature_engineer_path")
        if fe_path and Path(fe_path).exists():
            fe = FeatureEngineer.load(fe_path)
            X = fe.transform(X)
        
        # Predict
        prediction = model.predict(X)[0]
        
        result = {"prediction": prediction}
        
        # Get probabilities for classification
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = model.classes_
            result["probability"] = {str(c): round(p, 4) for c, p in zip(classes, proba)}
            result["confidence"] = round(max(proba), 4)
        
        return result
    
    def predict_batch(self, model_id: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make batch predictions."""
        model_info = self.models_registry.get(model_id)
        if not model_info:
            raise ValueError(f"Model not found: {model_id}")
        
        model = joblib.load(model_info["model_path"])
        feature_names = model_info.get("feature_names", [])
        
        X = pd.DataFrame(data)
        X = pd.get_dummies(X, drop_first=True)
        
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names] if feature_names else X
        
        fe_path = model_info.get("feature_engineer_path")
        if fe_path and Path(fe_path).exists():
            fe = FeatureEngineer.load(fe_path)
            X = fe.transform(X)
        
        predictions = model.predict(X).tolist()
        
        result = {
            "predictions": predictions,
            "count": len(predictions)
        }
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            classes = model.classes_
            result["probabilities"] = [
                {str(c): round(p, 4) for c, p in zip(classes, row)}
                for row in proba
            ]
        
        return result
    
    # ============ MODEL MANAGEMENT ============
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model info by ID."""
        return self.models_registry.get(model_id)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models."""
        return list(self.models_registry.values())
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        if model_id not in self.models_registry:
            return False
        
        model_info = self.models_registry[model_id]
        
        # Delete files
        model_path = Path(model_info["model_path"])
        if model_path.exists():
            model_dir = model_path.parent
            import shutil
            shutil.rmtree(model_dir)
        
        del self.models_registry[model_id]
        self._save_registry()
        return True
    
    # ============ HELPERS ============
    
    def _get_model(self, algorithm: str, problem_type: str):
        """Get sklearn model by algorithm name."""
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.svm import SVC, SVR
        import xgboost as xgb
        
        if problem_type == "classification":
            models = {
                "logistic": LogisticRegression(max_iter=1000, random_state=42),
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                "xgboost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss'),
                "svm": SVC(kernel='rbf', probability=True, random_state=42),
                "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
            }
        else:
            models = {
                "linear": LinearRegression(),
                "ridge": Ridge(alpha=1.0, random_state=42),
                "lasso": Lasso(alpha=1.0, random_state=42),
                "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "xgboost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
                "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
                "svr": SVR(kernel='rbf')
            }
        
        return models.get(algorithm)
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Extract feature importance from model."""
        importance = []
        
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_).flatten()
            else:
                return []
            
            # Normalize
            total = sum(importances)
            if total > 0:
                importances = importances / total
            
            for i, (name, imp) in enumerate(sorted(zip(feature_names, importances), key=lambda x: -x[1])):
                importance.append({
                    "feature": name,
                    "importance": round(float(imp), 4),
                    "rank": i + 1
                })
        except Exception as e:
            print(f"Could not get feature importance: {e}")
        
        return importance[:20]  # Top 20


# Global instance
ml_service = MLService()
