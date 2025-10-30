"""
Data Preparation Module for Predictive Delivery Optimizer.

This module reads and merges three CSV files, drops data leakage columns,
creates target variables, and saves processed data with statistics.
"""

import logging
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_data_files(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read the three CSV files required for merging.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Tuple of (orders_df, delivery_df, routes_df)
    """
    logger.info("Reading data files...")
    
    orders_path = data_dir / "orders.csv"
    delivery_path = data_dir / "delivery_performance.csv"
    routes_path = data_dir / "routes_distance.csv"
    
    orders_df = pd.read_csv(orders_path)
    delivery_df = pd.read_csv(delivery_path)
    routes_df = pd.read_csv(routes_path)
    
    logger.info(f"Orders data shape: {orders_df.shape}")
    logger.info(f"Delivery data shape: {delivery_df.shape}")
    logger.info(f"Routes data shape: {routes_df.shape}")
    
    return orders_df, delivery_df, routes_df


def merge_data(
    orders_df: pd.DataFrame,
    delivery_df: pd.DataFrame,
    routes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge three dataframes on Order_ID using inner join.
    
    Args:
        orders_df: Orders dataframe
        delivery_df: Delivery performance dataframe
        routes_df: Routes distance dataframe
        
    Returns:
        Merged dataframe
    """
    logger.info("Merging dataframes on Order_ID...")
    
    # Merge orders with delivery performance
    merged_df = pd.merge(orders_df, delivery_df, on="Order_ID", how="inner")
    
    # Merge with routes data
    merged_df = pd.merge(merged_df, routes_df, on="Order_ID", how="inner")
    
    logger.info(f"Merged data shape: {merged_df.shape}")
    
    return merged_df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that cause data leakage (available only after delivery).
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with leakage columns removed
    """
    leakage_columns = [
        "Route",
        "Origin",
        "Destination",
        "Order_Date",
        "Special_Handling",
        "Delivery_Status",
        "Quality_Issue",
        "Actual_Delivery_Days",  # Post-delivery, used in target calculation
        "Customer_Rating",  # Post-delivery rating, causes leakage
        "Traffic_Delay_Minutes"  # Post-delivery data, causes leakage
        # Note: Promised_Delivery_Days kept - available before delivery
        # Note: Customer_Segment kept - available before delivery
    ]
    
    logger.info(f"Dropping leakage columns: {leakage_columns}")
    
    # Only drop columns that exist
    columns_to_drop = [col for col in leakage_columns if col in df.columns]
    df_cleaned = df.drop(columns=columns_to_drop)
    
    logger.info(f"Data shape after dropping leakage columns: {df_cleaned.shape}")
    
    return df_cleaned


def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables: delay_days and is_delayed.
    
    Args:
        df: Input dataframe with Actual_Delivery_Days and Promised_Delivery_Days
        
    Returns:
        Dataframe with target variables added
    """
    logger.info("Creating target variables...")
    
    # Calculate delay_days
    df["delay_days"] = df["Actual_Delivery_Days"] - df["Promised_Delivery_Days"]
    
    # Create binary target: is_delayed
    df["is_delayed"] = (df["delay_days"] > 0).astype(int)
    
    logger.info(f"Delay statistics:")
    logger.info(f"  Mean delay: {df['delay_days'].mean():.2f} days")
    logger.info(f"  Delay rate: {df['is_delayed'].mean() * 100:.2f}%")
    
    return df


def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics about the processed dataset.
    
    Args:
        df: Processed dataframe
        
    Returns:
        Dictionary with statistics
    """
    # Get feature count (excluding targets)
    feature_columns = [col for col in df.columns 
                      if col not in ["delay_days", "is_delayed", "Order_ID"]]
    
    stats = {
        "total_orders": len(df),
        "features": len(feature_columns),
        "delay_rate": f"{df['is_delayed'].mean() * 100:.2f}%",
        "mean_delay": f"{df['delay_days'].mean():.2f} days"
    }
    
    return stats


def save_processed_data(
    df: pd.DataFrame,
    output_dir: Path,
    stats: Dict[str, Any]
) -> None:
    """
    Save processed data and statistics.
    
    Args:
        df: Processed dataframe
        output_dir: Directory to save files
        stats: Statistics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save merged CSV
    merged_path = output_dir / "merged.csv"
    df.to_csv(merged_path, index=False)
    logger.info(f"Saved merged data to {merged_path}")
    
    # Save statistics
    stats_path = output_dir / "data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")


def main() -> None:
    """
    Main function to execute data preparation pipeline.
    """
    logger.info("Starting data preparation pipeline...")
    
    # Define paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "processed"
    
    # Read data files
    orders_df, delivery_df, routes_df = read_data_files(data_dir)
    
    # Merge data
    merged_df = merge_data(orders_df, delivery_df, routes_df)
    
    # Drop leakage columns
    cleaned_df = drop_leakage_columns(merged_df)
    
    # Create target variables
    final_df = create_target_variables(cleaned_df)
    
    # Calculate statistics
    stats = calculate_statistics(final_df)
    
    # Save processed data
    save_processed_data(final_df, output_dir, stats)
    
    logger.info("Data preparation completed successfully!")
    logger.info(f"Final statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()

