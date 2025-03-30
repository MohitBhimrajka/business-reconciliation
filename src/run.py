#!/usr/bin/env python3
"""
Offline reconciliation workflow script.
Processes files from data directory and generates analysis outputs.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from utils import (
    ensure_directories_exist, read_file,
    ORDERS_PATTERN, RETURNS_PATTERN, SETTLEMENT_PATTERN,
    ORDERS_MASTER, RETURNS_MASTER, SETTLEMENT_MASTER,
    ANALYSIS_OUTPUT, DATA_DIR
)
from ingestion import scan_directory, process_file
from analysis import analyze_orders, get_order_analysis_summary
from reporting import (
    save_analysis_results, generate_report, save_report,
    identify_anomalies, generate_visualizations
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('reconciliation.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process order reconciliation files and generate analysis.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing data files (default: data/)'
    )
    return parser.parse_args()

def load_previous_analysis() -> Optional[pd.DataFrame]:
    """
    Load previous analysis results if they exist.
    
    Returns:
        Previous analysis DataFrame or None if not found
    """
    try:
        if ANALYSIS_OUTPUT.exists():
            logger.info("Loading previous analysis results...")
            return read_file(ANALYSIS_OUTPUT)
    except Exception as e:
        logger.warning(f"Error loading previous analysis: {e}")
    return None

def process_data_files(data_dir: Path) -> bool:
    """
    Process all data files in the specified directory.
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        True if all critical files were processed successfully
    """
    # Scan directory for files
    logger.info(f"Scanning directory: {data_dir}")
    files = scan_directory(str(data_dir))
    
    if not any(files.values()):
        logger.error("No files found to process!")
        return False
    
    # Process each file type
    success = True
    
    # Process orders first
    for file_path in sorted(files['orders']):
        logger.info(f"Processing orders file: {file_path}")
        if not process_file(Path(file_path), 'orders'):
            logger.error(f"Failed to process orders file: {file_path}")
            success = False
    
    # Then process returns
    for file_path in sorted(files['returns']):
        logger.info(f"Processing returns file: {file_path}")
        if not process_file(Path(file_path), 'returns'):
            logger.warning(f"Failed to process returns file: {file_path}")
    
    # Finally process settlements
    for file_path in sorted(files['settlement']):
        logger.info(f"Processing settlement file: {file_path}")
        if not process_file(Path(file_path), 'settlement'):
            logger.warning(f"Failed to process settlement file: {file_path}")
    
    return success

def load_master_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from master files.
    
    Returns:
        Tuple of (orders_df, returns_df, settlement_df)
    """
    logger.info("Loading master files...")
    
    # Load orders (required)
    if not ORDERS_MASTER.exists():
        raise FileNotFoundError("Orders master file not found!")
    orders_df = read_file(ORDERS_MASTER)
    
    # Load returns (optional)
    returns_df = read_file(RETURNS_MASTER) if RETURNS_MASTER.exists() else pd.DataFrame()
    
    # Load settlements (optional)
    settlement_df = read_file(SETTLEMENT_MASTER) if SETTLEMENT_MASTER.exists() else pd.DataFrame()
    
    return orders_df, returns_df, settlement_df

def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_args()
        data_dir = Path(args.data_dir)
        
        # Ensure directories exist
        ensure_directories_exist()
        
        # Process all files
        if not process_data_files(data_dir):
            logger.error("Critical file processing failed!")
            return 1
        
        # Load master data
        try:
            orders_df, returns_df, settlement_df = load_master_data()
        except FileNotFoundError as e:
            logger.error(f"Error loading master data: {e}")
            return 1
        
        # Load previous analysis
        previous_analysis_df = load_previous_analysis()
        
        # Run analysis
        logger.info("Running order analysis...")
        analysis_df = analyze_orders(
            orders_df,
            returns_df,
            settlement_df,
            previous_analysis_df
        )
        
        # Get analysis summary
        logger.info("Generating analysis summary...")
        summary = get_order_analysis_summary(analysis_df)
        
        # Save analysis results
        logger.info("Saving analysis results...")
        save_analysis_results(analysis_df)
        
        # Generate and save report
        logger.info("Generating reconciliation report...")
        report = generate_report(summary)
        save_report(report)
        
        # Identify anomalies
        logger.info("Identifying anomalies...")
        anomalies_df = identify_anomalies(
            analysis_df,
            orders_df,
            returns_df,
            settlement_df
        )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        visualizations = generate_visualizations(analysis_df, summary)
        
        # Log completion summary
        logger.info("=" * 50)
        logger.info("Reconciliation completed successfully!")
        logger.info(f"Total orders processed: {len(orders_df):,}")
        logger.info(f"Total returns processed: {len(returns_df):,}")
        logger.info(f"Total settlements processed: {len(settlement_df):,}")
        logger.info(f"Anomalies identified: {len(anomalies_df):,}")
        logger.info("=" * 50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 