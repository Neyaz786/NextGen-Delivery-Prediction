"""
Model Training Module for Predictive Delivery Optimizer.

This module trains Random Forest classifier and regressor models
using RandomizedSearchCV for hyperparameter tuning.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from scipy.stats import randint, uniform
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_preprocessor_columns(X: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """Align X columns to what the preprocessor expects by adding any missing
    columns with safe defaults. Numeric -> 0.0, Ordinal -> 0, Nominal -> 'Unknown'.
    """
    if preprocessor is None or not hasattr(preprocessor, "transformers_"):
        return X

    required_numeric: list[str] = []
    required_ordinal: list[str] = []
    required_nominal: list[str] = []

    for name, transformer, cols in preprocessor.transformers_:
        if cols is None:
            continue
        if name == 'num':
            required_numeric.extend(cols)
        elif name == 'ord':
            required_ordinal.extend(cols)
        elif name == 'nom':
            required_nominal.extend(cols)

    # Add missing numeric
    for col in required_numeric:
        if col not in X.columns:
            X[col] = 0.0
    # Add missing ordinal
    for col in required_ordinal:
        if col not in X.columns:
            X[col] = 0
    # Add missing nominal
    for col in required_nominal:
        if col not in X.columns:
            X[col] = 'Unknown'

    # Order columns to put required first (others can remain)
    required_all = required_numeric + required_ordinal + required_nominal
    ordered_cols = [c for c in required_all if c in X.columns] + [c for c in X.columns if c not in required_all]
    return X[ordered_cols]


def load_processed_data(processed_dir: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Any]:
    """
    Load processed data and preprocessor.
    
    Args:
        processed_dir: Directory containing processed data
        
    Returns:
        Tuple of (X, y_class, y_reg, preprocessor)
    """
    logger.info("Loading processed data and preprocessor...")
    
    # Load processed data
    processed_path = processed_dir / "merged_processed.csv"
    if not processed_path.exists():
        # Fallback to merged.csv if processed version doesn't exist
        processed_path = processed_dir / "merged.csv"
    
    df = pd.read_csv(processed_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Drop leakage columns if they exist (should not be in features)
    leakage_columns = [
        "Actual_Delivery_Days",  # Post-delivery, used in target calculation
        "Customer_Rating",  # Post-delivery rating, causes leakage
        "Traffic_Delay_Minutes",  # Post-delivery data, causes leakage
        "Delivery_Status",  # Post-delivery information
        "Route", "Origin", "Destination",
        "Order_Date", "Special_Handling", "Quality_Issue"
        # Note: Promised_Delivery_Days kept - available before delivery (promise at order time)
        # Note: Customer_Segment kept - available before delivery (customer information)
    ]
    columns_to_drop = [col for col in leakage_columns if col in df.columns]
    if columns_to_drop:
        logger.info(f"Dropping leakage columns from data: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
        logger.info(f"Data shape after dropping leakage: {df.shape}")
    
    # Identify feature columns (exclude targets and Order_ID)
    target_cols = ["delay_days", "is_delayed", "Order_ID"]
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    
    X = df[feature_cols]
    y_class = df["is_delayed"]
    y_reg = df["delay_days"]
    
    # Load preprocessor
    preprocessor_path = processed_dir / "preprocessor.joblib"
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        logger.info("Preprocessor loaded")
        # Align columns to preprocessor expectations, then transform
        X_aligned = ensure_preprocessor_columns(X.copy(), preprocessor)
        X_transformed = preprocessor.transform(X_aligned)
        X = pd.DataFrame(X_transformed)
    else:
        logger.warning("Preprocessor not found. Using raw features.")
    
    return X, y_class, y_reg, preprocessor if preprocessor_path.exists() else None


def get_classifier_param_space() -> Dict[str, Any]:
    """
    Define parameter space for Random Forest Classifier.
    
    Returns:
        Dictionary of parameter distributions for RandomizedSearchCV
    """
    return {
        'n_estimators': randint(50, 501),
        'max_depth': randint(5, 51),
        'min_samples_split': randint(2, 21),
        'min_samples_leaf': randint(1, 11),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }


def get_regressor_param_space() -> Dict[str, Any]:
    """
    Define parameter space for Random Forest Regressor.
    
    Returns:
        Dictionary of parameter distributions for RandomizedSearchCV
    """
    return {
        'n_estimators': randint(50, 501),
        'max_depth': randint(5, 51),
        'min_samples_split': randint(2, 21),
        'min_samples_leaf': randint(1, 11),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Train Random Forest Classifier with RandomizedSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target (binary)
        X_test: Test features
        y_test: Test target (binary)
        
    Returns:
        Tuple of (best_model, metrics_dict)
    """
    logger.info("Training Random Forest Classifier...")
    
    # Base classifier
    base_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Parameter space
    param_dist = get_classifier_param_space()
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_classifier,
        param_distributions=param_dist,
        n_iter=75,  # Medium-large parameter space search
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Fit
    logger.info("Performing hyperparameter search...")
    random_search.fit(X_train, y_train)
    
    # Get best model
    best_classifier = random_search.best_estimator_
    logger.info(f"Best parameters: {random_search.best_params_}")
    logger.info(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = best_classifier.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    logger.info("Classifier Test Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return best_classifier, metrics


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    Train Random Forest Regressor with RandomizedSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target (continuous)
        X_test: Test features
        y_test: Test target (continuous)
        
    Returns:
        Tuple of (best_model, metrics_dict)
    """
    logger.info("Training Random Forest Regressor...")
    
    # Base regressor
    base_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Parameter space
    param_dist = get_regressor_param_space()
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_regressor,
        param_distributions=param_dist,
        n_iter=75,  # Medium-large parameter space search
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Fit
    logger.info("Performing hyperparameter search...")
    random_search.fit(X_train, y_train)
    
    # Get best model
    best_regressor = random_search.best_estimator_
    logger.info(f"Best parameters: {random_search.best_params_}")
    logger.info(f"Best cross-validation score: {-random_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = best_regressor.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    logger.info("Regressor Test Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return best_regressor, metrics


def save_model(
    model: Any,
    model_path: Path,
    metadata: Dict[str, Any]
) -> None:
    """
    Save model and metadata.
    
    Args:
        model: Trained model
        model_path: Path to save model
        metadata: Model metadata dictionary
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save metadata
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main() -> None:
    """
    Main function to execute model training pipeline.
    """
    logger.info("Starting model training pipeline...")
    
    # Define paths
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "processed"
    models_dir = project_root / "models"
    
    # Load data
    X, y_class, y_reg, preprocessor = load_processed_data(processed_dir)
    
    # Train-test split (80/20)
    logger.info("Performing train-test split (80/20)...")
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Train classifier
    classifier, classifier_metrics = train_classifier(
        X_train, y_class_train, X_test, y_class_test
    )
    
    # Prepare classifier metadata
    classifier_metadata = {
        'model_type': 'RandomForestClassifier',
        'timestamp': datetime.now().isoformat(),
        'best_params': classifier.get_params(),
        'metrics': classifier_metrics,
        'random_state': 42
    }
    
    # Save classifier
    classifier_path = models_dir / "classifier.joblib"
    save_model(classifier, classifier_path, classifier_metadata)
    
    # Train regressor
    regressor, regressor_metrics = train_regressor(
        X_train, y_reg_train, X_test, y_reg_test
    )
    
    # Prepare regressor metadata
    regressor_metadata = {
        'model_type': 'RandomForestRegressor',
        'timestamp': datetime.now().isoformat(),
        'best_params': regressor.get_params(),
        'metrics': regressor_metrics,
        'random_state': 42
    }
    
    # Save regressor
    regressor_path = models_dir / "regressor.joblib"
    save_model(regressor, regressor_path, regressor_metadata)
    
    logger.info("Model training completed successfully!")


if __name__ == "__main__":
    main()

