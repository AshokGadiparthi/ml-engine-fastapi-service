"""
Layer 3 Evaluation Service Integration - FIXED VERSION
=======================================

Wraps Layer 3 Enhanced Evaluation functions for FastAPI.
Provides business-focused evaluation with real-time threshold adjustment.

FIX: Now accepts and passes feature_names parameter through the pipeline
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import Layer 3 functions
from ml_engine.evaluation_enhanced import (
    evaluate_with_threshold,
    calculate_business_impact,
    calculate_roc_pr_curves,
    calculate_learning_curve_data,
    calculate_feature_importance_with_correlation,
    get_optimal_threshold_for_business,
    assess_production_readiness,
    complete_model_evaluation
)

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for model evaluation using Layer 3."""
    
    def __init__(self):
        """Initialize evaluation service."""
        self.logger = logger
    
    def evaluate_with_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        target_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model at specific threshold.
        
        Args:
            y_true: True labels (0/1)
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            target_names: Class names (list, not tuple)
        
        Returns:
            Evaluation result with metrics and confusion matrix
        """
        try:
            self.logger.info(f"Evaluating at threshold {threshold}")
            result = evaluate_with_threshold(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                threshold=threshold,
                target_names=target_names
            )
            self.logger.info(f"✅ Evaluation complete: {result['metrics']['accuracy']:.3f} accuracy")
            return result
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
    
    def calculate_business_impact(
        self,
        evaluation_result: Dict[str, Any],
        cost_false_positive: float = 500,
        cost_false_negative: float = 2000,
        revenue_true_positive: float = 1000,
        volume: float = 10000
    ) -> Dict[str, Any]:
        """
        Calculate business impact of predictions.
        
        Args:
            evaluation_result: Result from evaluate_with_threshold
            cost_false_positive: Cost of false positive in dollars
            cost_false_negative: Cost of false negative in dollars
            revenue_true_positive: Revenue from true positive in dollars
            volume: Number of predictions per period
        
        Returns:
            Business impact metrics including costs and profit
        """
        try:
            self.logger.info("Calculating business impact...")
            result = calculate_business_impact(
                evaluation_result=evaluation_result,
                cost_false_positive=cost_false_positive,
                cost_false_negative=cost_false_negative,
                revenue_true_positive=revenue_true_positive,
                volume=int(volume)
            )
            self.logger.info(f"✅ Business impact: ${result['financial']['net_profit']:,} profit")
            return result
        except Exception as e:
            self.logger.error(f"Business impact calculation failed: {e}")
            raise
    
    def get_roc_pr_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get ROC and PR curve data.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
        
        Returns:
            ROC and PR curve data for plotting
        """
        try:
            self.logger.info("Calculating ROC and PR curves...")
            result = calculate_roc_pr_curves(
                y_true=y_true,
                y_pred_proba=y_pred_proba
            )
            self.logger.info(f"✅ Curves calculated: AUC={result['roc_curve']['auc']:.3f}")
            return result
        except Exception as e:
            self.logger.error(f"Curve calculation failed: {e}")
            raise
    
    def get_learning_curve(
        self,
        y_test: np.ndarray,
        y_pred_proba: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        y_pred_train: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Get learning curve data for overfitting analysis.
        
        Args:
            y_test: Test labels
            y_pred_proba: Test predictions
            y_train: Train labels (optional)
            y_pred_train: Train predictions (optional)
        
        Returns:
            Learning curve data
        """
        try:
            self.logger.info("Analyzing learning curve...")
            result = calculate_learning_curve_data(
                y_test=y_test,
                y_pred_proba=y_pred_proba,
                y_train=y_train,
                y_pred_train=y_pred_train
            )
            self.logger.info(f"✅ Learning curve: {result['overfitting_status']}")
            return result
        except Exception as e:
            self.logger.error(f"Learning curve analysis failed: {e}")
            raise
    
    def get_feature_importance(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Get feature importance with correlation analysis.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: Feature names
        
        Returns:
            Feature importance with correlation analysis
        """
        try:
            self.logger.info("Calculating feature importance...")
            result = calculate_feature_importance_with_correlation(
                model=model,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names
            )
            self.logger.info(f"✅ Importance calculated for {result['total_features']} features")
            return result
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {e}")
            raise
    
    def get_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        cost_false_positive: float = 500,
        cost_false_negative: float = 2000,
        revenue_true_positive: float = 1000
    ) -> Dict[str, Any]:
        """
        Find optimal threshold for business metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            cost_false_positive: Cost of FP
            cost_false_negative: Cost of FN
            revenue_true_positive: Revenue from TP
        
        Returns:
            Optimal threshold and expected profit
        """
        try:
            self.logger.info("Finding optimal threshold...")
            result = get_optimal_threshold_for_business(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                cost_false_positive=cost_false_positive,
                cost_false_negative=cost_false_negative,
                revenue_true_positive=revenue_true_positive
            )
            self.logger.info(f"✅ Optimal threshold: {result['optimal_threshold']:.2f}")
            return result
        except Exception as e:
            self.logger.error(f"Optimal threshold finding failed: {e}")
            raise
    
    def assess_production_readiness(
        self,
        evaluation_result: Dict[str, Any],
        learning_curve: Dict[str, Any],
        business_impact: Dict[str, Any],
        feature_importance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess production readiness with 18-point checklist.
        
        Args:
            evaluation_result: Evaluation result
            learning_curve: Learning curve analysis
            business_impact: Business impact metrics
            feature_importance: Feature importance data
        
        Returns:
            Production readiness assessment
        """
        try:
            self.logger.info("Assessing production readiness...")
            result = assess_production_readiness(
                evaluation_result=evaluation_result,
                learning_curve=learning_curve,
                business_impact=business_impact,
                feature_importance=feature_importance
            )
            self.logger.info(f"✅ Readiness: {result['overall_status']} ({result['summary']['passed']}/{result['summary']['total_criteria']} criteria)")
            return result
        except Exception as e:
            self.logger.error(f"Production readiness assessment failed: {e}")
            raise
    
    def complete_evaluation(
        self,
        model: Any,
        X_test,
        y_test: np.ndarray,
        y_pred_proba: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        y_pred_train_proba: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        cost_fp: float = 500,
        cost_fn: float = 2000,
        revenue_tp: float = 1000,
        feature_names: Optional[List[str]] = None  # ✅ ADDED
    ) -> Dict[str, Any]:
        """
        Complete model evaluation in one call.
        
        Returns all metrics: evaluation, business impact, curves,
        learning curve, feature importance, optimal threshold, readiness.
        
        FIXED: Now accepts and passes feature_names to ensure real feature
        names are returned instead of generic "feature_0" placeholders.
        
        Args:
            model: Trained model
            X_test: Test features (DataFrame or array)
            y_test: Test labels
            y_pred_proba: Test predictions (probabilities)
            y_train: Train labels (optional)
            y_pred_train_proba: Train predictions (optional)
            threshold: Evaluation threshold
            cost_fp: Cost of false positive
            cost_fn: Cost of false negative
            revenue_tp: Revenue from true positive
            feature_names: Feature names (optional, for real feature labels)
        
        Returns:
            Complete evaluation package with all metrics
        """
        try:
            self.logger.info("Running complete model evaluation...")
            
            # ✅ FIX: Pass feature_names to complete_model_evaluation
            result = complete_model_evaluation(
                model=model,
                X_test=X_test,
                y_test=y_test,
                y_pred_proba=y_pred_proba,
                y_train=y_train,
                y_pred_train_proba=y_pred_train_proba,
                threshold=threshold,
                cost_fp=cost_fp,
                cost_fn=cost_fn,
                revenue_tp=revenue_tp,
                feature_names=feature_names  # ✅ ADDED
            )
            self.logger.info(f"✅ Complete evaluation finished!")
            return result
        except Exception as e:
            self.logger.error(f"Complete evaluation failed: {e}")
            raise


# Create singleton instance
evaluation_service = EvaluationService()