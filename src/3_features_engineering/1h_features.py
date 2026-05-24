import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================
# Configuration & Constants
# ===============================
DESKTOP_PATH = Path(os.path.expanduser("~/Desktop"))
PROJECT_ROOT = DESKTOP_PATH / "Equinox Quantitative Analytics and Predictive Platform/data"

OUTPUT = PROJECT_ROOT / "features"
INPUT = PROJECT_ROOT / "processed_data"
os.makedirs(OUTPUT, exist_ok=True)


class PipelineStage(Enum):
    """Pipeline execution stages for strict ordering"""
    DATA_LOAD = 1
    DATA_VALIDATION = 2
    DATA_PREPROCESSING = 3
    FEATURE_ENGINEERING = 4
    FEATURE_VALIDATION = 5
    DATA_EXPORT = 6


@dataclass
class FeatureValidationConfig:
    """Configuration for feature validation"""
    min_non_null_ratio: float = 0.95
    max_nan_threshold: float = 0.05
    feature_bounds: Dict[str, Tuple[float, float]] = None


@dataclass
class DataQualityMetrics:
    """Metrics for data quality validation"""
    total_rows: int
    null_counts: Dict[str, int]
    null_percentages: Dict[str, float]
    numeric_features: List[str]
    categorical_features: List[str]
    warnings: List[str]


class FeatureEngineeringPipeline:
    """Strict pipeline with ordered stages for feature engineering"""
    
    def __init__(self, validation_config: FeatureValidationConfig = None):
        self.validation_config = validation_config or FeatureValidationConfig()
        self.current_stage = None
        self.stage_history = []
        self.quality_metrics = None
        
    def _set_stage(self, stage: PipelineStage) -> None:
        """Enforce strict pipeline ordering"""
        if self.current_stage is not None:
            if stage.value <= self.current_stage.value:
                raise RuntimeError(
                    f"Pipeline order violation: Cannot move from {self.current_stage.name} "
                    f"to {stage.name}. Stages must proceed in order."
                )
        self.current_stage = stage
        self.stage_history.append(stage)
        logger.info(f"[STAGE {stage.value}] {stage.name}")
    
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load and validate raw data"""
        self._set_stage(PipelineStage.DATA_LOAD)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    def validate_input_data(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Validate data integrity and quality"""
        self._set_stage(PipelineStage.DATA_VALIDATION)
        
        required_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        metrics = DataQualityMetrics(
            total_rows=len(df),
            null_counts={col: int(df[col].isna().sum()) for col in df.columns},
            null_percentages={col: float(df[col].isna().sum() / len(df)) for col in df.columns},
            numeric_features=df.select_dtypes(include=[np.number]).columns.tolist(),
            categorical_features=df.select_dtypes(include=['object']).columns.tolist(),
            warnings=[]
        )
        
        for col, null_pct in metrics.null_percentages.items():
            if null_pct > self.validation_config.max_nan_threshold:
                metrics.warnings.append(
                    f"Column '{col}' has {null_pct:.2%} null values (threshold: "
                    f"{self.validation_config.max_nan_threshold:.2%})"
                )
        
        self.quality_metrics = metrics
        logger.info(f"Data validation complete. Warnings: {len(metrics.warnings)}")
        for warning in metrics.warnings:
            logger.warning(f"  - {warning}")
        
        return metrics
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        self._set_stage(PipelineStage.DATA_PREPROCESSING)
        
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], format='mixed', utc=False)
        df = df.sort_values("Date").reset_index(drop=True)
        
        if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
            logger.warning("Found non-positive price values, applying absolute value")
            df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].abs()
        
        invalid_highs = df["High"] < df["Low"]
        if invalid_highs.any():
            logger.warning(f"Found {invalid_highs.sum()} rows where High < Low, fixing...")
            df.loc[invalid_highs, ["High", "Low"]] = df.loc[
                invalid_highs, ["Low", "High"]
            ].values
        
        logger.info("Data preprocessing complete")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all technical features"""
        self._set_stage(PipelineStage.FEATURE_ENGINEERING)
        
        df = df.copy()
        features_created = []
        
        df["return"] = df["Close"].pct_change()
        features_created.append("return")
        
        adx = ta.adx(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            length=14
        )
        df["ADX"] = adx["ADX_14"]
        df["DI_plus"] = adx["DMP_14"]
        df["DI_minus"] = adx["DMN_14"]
        features_created.extend(["ADX", "DI_plus", "DI_minus"])
        
        df["ATR"] = ta.atr(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            length=14
        )
        features_created.append("ATR")
        
        ichimoku = ta.ichimoku(
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )
        df["Ichimoku_conversion"] = ichimoku[0]["ITS_9"]
        df["Ichimoku_base"] = ichimoku[0]["IKS_26"]
        df["Ichimoku_spanA"] = ichimoku[0]["ISA_9"]
        df["Ichimoku_spanB"] = ichimoku[0]["ISB_26"]
        df["Ichimoku_lagging"] = ichimoku[0]["ICS_26"]
        features_created.extend([
            "Ichimoku_conversion", "Ichimoku_base", "Ichimoku_spanA",
            "Ichimoku_spanB", "Ichimoku_lagging"
        ])
        
        df["price_vs_spanA"] = df["Close"] - df["Ichimoku_spanA"]
        df["price_vs_spanB"] = df["Close"] - df["Ichimoku_spanB"]
        df["conversion_base_diff"] = df["Ichimoku_conversion"] - df["Ichimoku_base"]
        df["lagging_diff"] = df["Close"] - df["Ichimoku_lagging"]
        features_created.extend([
            "price_vs_spanA", "price_vs_spanB", "conversion_base_diff", "lagging_diff"
        ])
        
        df["trend_strength"] = df["ADX"] * (df["DI_plus"] - df["DI_minus"])
        features_created.append("trend_strength")
        
        df["range"] = df["High"] - df["Low"]
        features_created.append("range")
        
        df["ATR_change"] = df["ATR"].diff()
        df["ADX_change"] = df["ADX"].diff()
        features_created.extend(["ATR_change", "ADX_change"])
        
        for lag in [1, 2, 3]:
            df[f"return_lag{lag}"] = df["return"].shift(lag)
            df[f"ADX_lag{lag}"] = df["ADX"].shift(lag)
            df[f"DI_plus_lag{lag}"] = df["DI_plus"].shift(lag)
            df[f"DI_minus_lag{lag}"] = df["DI_minus"].shift(lag)
            features_created.extend([
                f"return_lag{lag}", f"ADX_lag{lag}",
                f"DI_plus_lag{lag}", f"DI_minus_lag{lag}"
            ])
        
        df["volume_log"] = np.log1p(df["Volume"].values.astype(np.float64))
        # Safe volume_change: if Volume is all zeros (e.g. forex pairs), fill with 0
        if df["Volume"].sum() == 0:
            df["volume_change"] = 0.0
        else:
            df["volume_change"] = df["Volume"].pct_change()
        features_created.extend(["volume_log", "volume_change"])
        
        logger.info(f"Created {len(features_created)} features")
        return df
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate engineered features"""
        self._set_stage(PipelineStage.FEATURE_VALIDATION)
        
        validation_results = {
            "total_features": len(df.columns),
            "feature_checks": {},
            "passed": True,
            "errors": [],
            "warnings": []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].values
            
            inf_mask = np.isinf(col_data)
            if inf_mask.any():
                inf_count = int(inf_mask.sum())
                validation_results["warnings"].append(
                    f"Feature '{col}' contains {inf_count} infinite values"
                )
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            nan_ratio = df[col].isna().sum() / len(df)
            if nan_ratio > self.validation_config.max_nan_threshold:
                validation_results["errors"].append(
                    f"Feature '{col}' exceeds NaN threshold: {nan_ratio:.2%}"
                )
                validation_results["passed"] = False
        
        logger.info(f"Feature validation: {'PASSED' if validation_results['passed'] else 'FAILED'}")
        for warning in validation_results["warnings"]:
            logger.warning(f"  - {warning}")
        for error in validation_results["errors"]:
            logger.error(f"  - {error}")
        
        return validation_results
    
    def create_target_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable and handle NaN values"""
        df = df.copy()
        
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        
        rows_before = len(df)
        df = df.dropna()
        rows_removed = rows_before - len(df)
        
        logger.info(f"Removed {rows_removed} rows with NaN values")
        logger.info(f"Final dataset shape: {df.shape}")
        
        ichimoku_raw = [
            "Ichimoku_conversion", "Ichimoku_base", "Ichimoku_spanA",
            "Ichimoku_spanB", "Ichimoku_lagging"
        ]
        df = df.drop(columns=ichimoku_raw)
        
        return df
    
    def export_features(self, df: pd.DataFrame, file_name: str) -> Path:
        """Save engineered features"""
        self._set_stage(PipelineStage.DATA_EXPORT)
        
        output_path = OUTPUT / file_name.replace("cleaned.csv", "features.csv")
        df.to_csv(output_path, index=False)
        
        logger.info(f"Features saved to: {output_path}")
        logger.info(f"Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        
        return output_path
    
    def run(self, input_file_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute complete pipeline"""
        try:
            file_path = INPUT / input_file_name
            df = self.load_data(file_path)
            
            self.validate_input_data(df)
            df = self.preprocess_data(df)
            df = self.engineer_features(df)
            validation_results = self.validate_features(df)
            df = self.create_target_and_clean(df)
            self.export_features(df, input_file_name)
            
            logger.info("✓ Pipeline completed successfully")
            
            return df, validation_results
            
        except Exception as e:
            logger.error(f"✗ Pipeline failed at stage {self.current_stage}: {str(e)}")
            raise


class FeatureTestStrategy:
    """Comprehensive testing strategy for feature validation"""
    
    @staticmethod
    def test_feature_distributions(df: pd.DataFrame) -> Dict[str, Any]:
        """Test feature distributions for anomalies"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        test_results: Dict[str, Any] = {"passed": True, "issues": []}
        
        for col in numeric_cols:
            col_data = df[col]
            
            std_dev = float(col_data.std())
            if std_dev == 0:
                test_results["issues"].append(f"Feature '{col}' has zero variance")
                test_results["passed"] = False
            
            skewness = float(col_data.skew())
            if abs(skewness) > 3:
                test_results["issues"].append(
                    f"Feature '{col}' has extreme skewness: {skewness:.2f}"
                )
        
        return test_results
    
    @staticmethod
    def test_target_distribution(df: pd.DataFrame) -> Dict[str, Any]:
        """Test target variable balance"""
        if "target" not in df.columns:
            return {"passed": False, "error": "Target column not found"}
        
        target_counts = df["target"].value_counts()
        total = len(df)
        
        test_results: Dict[str, Any] = {
            "class_0_pct": float(target_counts.get(0, 0) / total),
            "class_1_pct": float(target_counts.get(1, 0) / total),
            "passed": True,
            "warning": ""
        }
        
        if test_results["class_0_pct"] < 0.3 or test_results["class_1_pct"] < 0.3:
            test_results["warning"] = "Imbalanced target classes detected"
        
        return test_results
    
    @staticmethod
    def run_all_tests(df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        results: Dict[str, Any] = {
            "distribution_tests": FeatureTestStrategy.test_feature_distributions(df),
            "target_tests": FeatureTestStrategy.test_target_distribution(df)
        }
        return results


def generate_ml_features(files: str):
    """Wrapper function for backward compatibility"""
    pipeline = FeatureEngineeringPipeline()
    try:
        df, validation_results = pipeline.run(files)
        
        logger.info("\n[TEST SUITE] Running feature validation tests...")
        test_results = FeatureTestStrategy.run_all_tests(df)
        
        logger.info("\n[TEST RESULTS]")
        logger.info(f"Distribution tests: {'✓ PASSED' if test_results['distribution_tests']['passed'] else '✗ FAILED'}")
        logger.info(f"Target tests: {'✓ PASSED' if test_results['target_tests']['passed'] else '✗ FAILED'}")
        
        if test_results['target_tests'].get('warning'):
            logger.warning(f"  Warning: {test_results['target_tests']['warning']}")
        
        return df, validation_results, test_results
        
    except Exception as e:
        logger.error(f"Feature generation failed for {files}: {str(e)}")
        raise


def main():
    """Main entry point with error handling"""
    TARGET_FILES_cleaned = [
        "BTC_1h_cleaned.csv",
        "AAPL_1h_cleaned.csv",
        "EURUSD_1h_cleaned.csv",
        "XAUUSD_1h_cleaned.csv",
    ]
    
    logger.info("=" * 80)
    logger.info("EQUINOX FEATURE ENGINEERING PIPELINE - 1H TIMEFRAME (numpy 2.2 compatible)")
    logger.info("=" * 80)
    
    successful = []
    failed = []
    
    for file_name in TARGET_FILES_cleaned:
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing: {file_name}")
            logger.info(f"{'='*80}")
            
            df, val_results, test_results = generate_ml_features(file_name)
            successful.append(file_name)
            
        except Exception as e:
            logger.error(f"Failed to process {file_name}: {str(e)}")
            failed.append(file_name)
    
    # Final report
    logger.info(f"\n{'='*80}")
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"✓ Successful: {len(successful)}/{len(TARGET_FILES_cleaned)}")
    for f in successful:
        logger.info(f"  - {f}")
    
    if failed:
        logger.error(f"✗ Failed: {len(failed)}/{len(TARGET_FILES_cleaned)}")
        for f in failed:
            logger.error(f"  - {f}")
    
    logger.info(f"\nAll outputs saved to: {OUTPUT}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()