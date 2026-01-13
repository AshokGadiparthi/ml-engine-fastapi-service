"""
EDA Analytics Service for FastAPI
Provides standalone EDA analysis without external dependencies
All functionality self-contained
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class EDAAnalyticsService:
    """
    Comprehensive EDA Analytics Service
    Standalone implementation, no external ML Engine required
    """
    
    def __init__(self):
        """Initialize EDA Analytics Service"""
        self.logger = logging.getLogger(__name__)
    
    def analyze_dataset(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive EDA on dataset
        
        Args:
            df: Input DataFrame
            target_column: Optional target column name
            
        Returns:
            Dictionary with complete EDA analysis
        """
        self.logger.info(f"Starting EDA analysis on dataset with shape {df.shape}")
        
        try:
            results = {
                'dataset_info': self._get_dataset_info(df),
                'histogram': self._histogram_analysis(df),
                'categorical': self._categorical_analysis(df),
                'missing': self._missing_pattern_analysis(df),
                'outliers': self._outlier_detection(df),
                'correlations': self._correlation_analysis(df),
                'quality': self._calculate_data_quality(df),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("EDA analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during EDA analysis: {str(e)}")
            raise
    
    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
    
    def _histogram_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze numeric feature distributions"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            try:
                data = df[col].dropna()
                if len(data) == 0:
                    continue
                
                results[col] = {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'q25': float(data.quantile(0.25)),
                    'q75': float(data.quantile(0.75)),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'count': int(len(data))
                }
            except Exception as e:
                self.logger.warning(f"Could not analyze column {col}: {str(e)}")
        
        return results
    
    def _categorical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical feature distributions"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        results = {}
        
        for col in categorical_cols:
            try:
                value_counts = df[col].value_counts()
                results[col] = {
                    'unique_values': len(value_counts),
                    'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'mode_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'top_10': value_counts.head(10).to_dict(),
                    'missing_count': int(df[col].isna().sum()),
                    'missing_percentage': float(df[col].isna().sum() / len(df) * 100)
                }
            except Exception as e:
                self.logger.warning(f"Could not analyze column {col}: {str(e)}")
        
        return results
    
    def _missing_pattern_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing_data = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_missing': int(df.isna().sum().sum()),
            'total_missing_percentage': float(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100),
            'missing_per_column': {},
            'rows_with_missing': int((df.isna().sum(axis=1) > 0).sum()),
            'fully_missing_columns': []
        }
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_data['missing_per_column'][col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(df) * 100)
                }
            if missing_count == len(df):
                missing_data['fully_missing_columns'].append(col)
        
        return missing_data
    
    def _outlier_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR == 0:
                    continue
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                results[col] = {
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_count': int(outlier_count),
                    'outlier_percentage': float(outlier_count / len(df) * 100),
                    'outlier_indices': df[outlier_mask].index.tolist()[:100]  # Limit to 100
                }
            except Exception as e:
                self.logger.warning(f"Could not detect outliers in column {col}: {str(e)}")
        
        return results
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric features"""
        numeric_data = df.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {'message': 'Not enough numeric columns for correlation analysis'}
        
        try:
            correlation_matrix = numeric_data.corr()
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Strong correlation threshold
                        strong_correlations.append({
                            'feature1': str(correlation_matrix.columns[i]),
                            'feature2': str(correlation_matrix.columns[j]),
                            'correlation': float(corr_value)
                        })
            
            # Sort by absolute correlation
            strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'total_strong_correlations': len(strong_correlations)
            }
        except Exception as e:
            self.logger.warning(f"Could not analyze correlations: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall data quality assessment"""
        missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
        
        # Count outliers
        numeric_data = df.select_dtypes(include=[np.number])
        outlier_count = 0
        for col in numeric_data.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                outlier_count += ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        
        # Calculate quality score
        completeness = max(0, 100 - missing_pct)
        uniqueness = 100 - (len(df[df.duplicated()]) / len(df) * 100)
        
        quality_score = (completeness + uniqueness) / 2
        
        # Assessment
        if quality_score >= 90:
            assessment = "Excellent"
        elif quality_score >= 75:
            assessment = "Good"
        elif quality_score >= 50:
            assessment = "Fair"
        else:
            assessment = "Poor"
        
        return {
            'overall_score': float(quality_score),
            'assessment': assessment,
            'completeness': float(completeness),
            'uniqueness': float(uniqueness),
            'missing_percentage': float(missing_pct),
            'duplicate_rows': int(len(df[df.duplicated()])),
            'duplicate_percentage': float(len(df[df.duplicated()]) / len(df) * 100),
            'outlier_count': int(outlier_count),
            'recommendations': self._get_recommendations(missing_pct, assessment)
        }
    
    def _get_recommendations(self, missing_pct: float, assessment: str) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        if missing_pct > 30:
            recommendations.append("High missing data: Consider imputation or removal")
        elif missing_pct > 10:
            recommendations.append("Moderate missing data: Review and handle missing values")
        
        if assessment in ["Poor", "Fair"]:
            recommendations.append("Data quality is below optimal: Consider data cleaning")
        
        if not recommendations:
            recommendations.append("Data quality is good - proceed with analysis")
        
        return recommendations


# Endpoint-specific functions for direct use

async def get_histogram_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Get histogram analysis"""
    service = EDAAnalyticsService()
    return service._histogram_analysis(df)


async def get_categorical_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Get categorical analysis"""
    service = EDAAnalyticsService()
    return service._categorical_analysis(df)


async def get_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Get missing pattern analysis"""
    service = EDAAnalyticsService()
    return service._missing_pattern_analysis(df)


async def get_outlier_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Get outlier detection"""
    service = EDAAnalyticsService()
    return service._outlier_detection(df)


async def get_correlation_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Get correlation analysis"""
    service = EDAAnalyticsService()
    return service._correlation_analysis(df)


async def get_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """Get data quality assessment"""
    service = EDAAnalyticsService()
    return service._calculate_data_quality(df)
