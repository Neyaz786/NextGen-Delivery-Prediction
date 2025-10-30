"""
Feature Engineering Module for Predictive Delivery Optimizer.

This module creates derived features, applies ordinal encoding,
and builds a preprocessing pipeline for machine learning models.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features available before delivery.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with derived features added
    """
    logger.info("Creating derived features...")
    
    df = df.copy()
    
    # Derived feature 1: Order value per kilometer (if distance exists)
    # This indicates order density/value per unit distance
    if "Distance_KM" in df.columns and "Order_Value_INR" in df.columns:
        df["value_per_km"] = df["Order_Value_INR"] / (df["Distance_KM"] + 1)  # +1 to avoid division by zero
    
    # Derived feature 2: Fuel efficiency indicator
    # Higher fuel consumption relative to distance might indicate delays
    if "Fuel_Consumption_L" in df.columns and "Distance_KM" in df.columns:
        df["fuel_efficiency"] = df["Distance_KM"] / (df["Fuel_Consumption_L"] + 0.001)
    
    logger.info(f"Created 2 derived features")
    logger.info(f"Data shape: {df.shape}")
    
    return df


def get_feature_types(df: pd.DataFrame, target_cols: List[str] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Identify numeric, ordinal, and nominal features.
    
    Args:
        df: Input dataframe
        target_cols: List of target column names to exclude
        
    Returns:
        Tuple of (numeric_features, ordinal_features, nominal_features)
    """
    if target_cols is None:
        target_cols = ["delay_days", "is_delayed", "Order_ID"]
    
    # Exclude target columns and Order_ID
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    # Ordinal features
    ordinal_features = ["Priority", "Weather_Impact"]
    ordinal_features = [f for f in ordinal_features if f in feature_cols]
    
    # Numeric features (excluding ordinal and nominal)
    numeric_features = []
    for col in feature_cols:
        if col not in ordinal_features and df[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
    
    # Nominal features (categorical that are not ordinal)
    nominal_features = []
    for col in feature_cols:
        if col not in ordinal_features and col not in numeric_features:
            nominal_features.append(col)
    
    logger.info(f"Numeric features: {len(numeric_features)}")
    logger.info(f"Ordinal features: {ordinal_features}")
    logger.info(f"Nominal features: {len(nominal_features)}")
    
    return numeric_features, ordinal_features, nominal_features


def create_ordinal_encoder() -> Dict[str, Dict[str, int]]:
    """
    Create ordinal encoding mappings.
    
    Returns:
        Dictionary with ordinal mappings for each ordinal feature
    """
    ordinal_mappings = {
        "Priority": {
            "Economy": 0,
            "Standard": 1,
            "Express": 2,
            "Unknown": 3
        },
        "Weather_Impact": {
            "Unknown": 0,
            "None": 0,
            "Fog": 1,
            "Light_Rain": 2,
            "Heavy_Rain": 3
        }
    }
    
    return ordinal_mappings


def apply_ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ordinal encoding to categorical features.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with ordinal encoding applied
    """
    logger.info("Applying ordinal encoding...")
    
    df = df.copy()
    ordinal_mappings = create_ordinal_encoder()
    
    for feature, mapping in ordinal_mappings.items():
        if feature in df.columns:
            # Replace values according to mapping
            df[feature] = df[feature].map(mapping).fillna(
                mapping.get("Unknown", mapping.get("None", 0))
            ).astype(int)
            logger.info(f"Encoded {feature}")
    
    return df


def build_preprocessing_pipeline(
    numeric_features: List[str],
    ordinal_features: List[str],
    nominal_features: List[str],
    ordinal_mappings: Dict[str, Dict[str, int]]
) -> ColumnTransformer:
    """
    Build preprocessing pipeline for all feature types.
    
    Args:
        numeric_features: List of numeric feature names
        ordinal_features: List of ordinal feature names
        nominal_features: List of nominal feature names
        ordinal_mappings: Dictionary with ordinal mappings
        
    Returns:
        ColumnTransformer with preprocessing pipeline
    """
    logger.info("Building preprocessing pipeline...")
    
    transformers = []
    
    # Numeric features: Median imputation + Standard scaling
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_features))
    
    # Ordinal features: Already encoded, just pass through or scale
    if ordinal_features:
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0))
        ])
        transformers.append(('ord', ordinal_transformer, ordinal_features))
    
    # Nominal features: One-hot encoding
    if nominal_features:
        nominal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        transformers.append(('nom', nominal_transformer, nominal_features))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    logger.info("Preprocessing pipeline created")
    
    return preprocessor


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    Prepare features: create derived features and apply ordinal encoding.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (processed_df, numeric_features, ordinal_features, nominal_features)
    """
    # Create derived features
    df = create_derived_features(df)
    
    # Apply ordinal encoding
    df = apply_ordinal_encoding(df)
    
    # Get feature types
    numeric_features, ordinal_features, nominal_features = get_feature_types(df)
    
    return df, numeric_features, ordinal_features, nominal_features


def save_preprocessor(preprocessor: ColumnTransformer, output_path: Path) -> None:
    """
    Save preprocessor to disk.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        output_path: Path to save preprocessor
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)
    logger.info(f"Saved preprocessor to {output_path}")


def main() -> None:
    """
    Main function to execute feature engineering pipeline.
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Define paths
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "processed"
    merged_path = processed_dir / "merged.csv"
    
    # Check if merged data exists
    if not merged_path.exists():
        logger.error(f"Merged data not found at {merged_path}. Please run data_prep.py first.")
        return
    
    # Read merged data
    logger.info(f"Reading merged data from {merged_path}")
    df = pd.read_csv(merged_path)
    
    # Prepare features
    df_processed, numeric_features, ordinal_features, nominal_features = prepare_features(df)
    
    # Save processed data
    processed_output_path = processed_dir / "merged_processed.csv"
    df_processed.to_csv(processed_output_path, index=False)
    logger.info(f"Saved processed data to {processed_output_path}")
    
    # Get feature columns (excluding targets)
    feature_cols = numeric_features + ordinal_features + nominal_features
    
    # Separate features and targets
    X = df_processed[feature_cols]
    y_class = df_processed["is_delayed"]
    y_reg = df_processed["delay_days"]
    
    # Build preprocessing pipeline
    ordinal_mappings = create_ordinal_encoder()
    preprocessor = build_preprocessing_pipeline(
        numeric_features, ordinal_features, nominal_features, ordinal_mappings
    )
    
    # Fit preprocessor
    logger.info("Fitting preprocessor...")
    preprocessor.fit(X)
    
    # Save preprocessor
    preprocessor_path = processed_dir / "preprocessor.joblib"
    save_preprocessor(preprocessor, preprocessor_path)
    
    logger.info("Feature engineering completed successfully!")


if __name__ == "__main__":
    main()

