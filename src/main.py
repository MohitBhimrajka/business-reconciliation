"""
Main application module for the reconciliation application.
"""
import os
import sys
import logging
import argparse
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from utils import ensure_directories_exist
from ingestion import ingest_data
from analysis import analyze_orders, get_order_analysis_summary
from reporting import save_analysis_results, generate_report, generate_visualizations, identify_anomalies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_reconciliation(data_directory: str, visualize: bool = False) -> None:
    """
    Run the reconciliation process.
    
    Args:
        data_directory: Directory containing the data files
        visualize: Whether to generate visualizations
    """
    logger.info(f"Starting reconciliation process. Data directory: {data_directory}")
    
    # Ensure all necessary directories exist
    ensure_directories_exist()
    
    # Step 1: Ingest data
    logger.info("Step 1: Ingesting data...")
    orders_df, returns_df, settlement_df = ingest_data(data_directory)
    
    if orders_df.empty:
        logger.error("No order data found. Exiting.")
        return
    
    logger.info(f"Loaded {len(orders_df)} orders, {len(returns_df)} returns, {len(settlement_df)} settlements")
    
    # Step 2: Analyze orders
    logger.info("Step 2: Analyzing orders...")
    analysis_df = analyze_orders(orders_df, returns_df, settlement_df)
    
    if analysis_df.empty:
        logger.error("Analysis produced no results. Exiting.")
        return
    
    # Save analysis results
    save_analysis_results(analysis_df)
    
    # Step 3: Generate summary statistics
    logger.info("Step 3: Generating summary statistics...")
    summary = get_order_analysis_summary(analysis_df)
    
    # Generate report
    generate_report(summary)
    
    # Identify anomalies
    logger.info("Step 4: Identifying anomalies...")
    anomalies_df = identify_anomalies(analysis_df, orders_df, returns_df, settlement_df)
    
    if not anomalies_df.empty:
        logger.info(f"Found {len(anomalies_df)} anomalies")
    
    # Generate visualizations if requested
    if visualize:
        logger.info("Step 5: Generating visualizations...")
        generate_visualizations(analysis_df, orders_df)
    
    logger.info("Reconciliation process completed successfully")

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Order Reconciliation Application")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="reconciliation/data",
        help="Directory containing the data files"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations"
    )
    
    return parser.parse_args(args)

def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        Exit code
    """
    try:
        args = parse_args()
        run_reconciliation(args.data_dir, args.visualize)
        return 0
    except Exception as e:
        logger.exception(f"Error running reconciliation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 