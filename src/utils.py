"""
Utility functions for the reconciliation application.
"""
import os
import re
import logging
from typing import List, Dict, Set, Tuple
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File patterns
ORDERS_PATTERN = r'orders-(\d{2})-(\d{4})\.csv'
RETURNS_PATTERN = r'returns-(\d{2})-(\d{4})\.csv'
SETTLEMENT_PATTERN = r'settlement-(\d{2})-(\d{4})\.csv'

# Column names for each file type
ORDERS_COLUMNS = [
    'order release id',  # Note: with spaces
    'order line id',
    'order status',
    'final amount',
    'total mrp',
    'discount',
    'coupon discount',
    'shipping charge',
    'gift charge',
    'tax recovery',
    'is_ship_rel'
]

RETURNS_COLUMNS = [
    'order_release_id',  # Note: with underscore
    'order_line_id',
    'return_type',
    'return_date',
    'packing_date',
    'delivery_date',
    'customer_paid_amount',
    'postpaid_amount',
    'prepaid_amount',
    'mrp',
    'total_discount_amount',
    'total_settlement',
    'total_actual_settlement',
    'amount_pending_settlement'
]

SETTLEMENT_COLUMNS = [
    'order_release_id',  # Note: with underscore
    'order_line_id',
    'return_type',
    'return_date',
    'packing_date',
    'delivery_date',
    'customer_paid_amount',
    'postpaid_amount',
    'prepaid_amount',
    'mrp',
    'total_discount_amount',
    'total_expected_settlement',
    'total_actual_settlement',
    'amount_pending_settlement'
]

# Column renaming mapping for standardization
COLUMN_RENAMES = {
    "orders": {
        "order release id": "order_release_id"  # Standardize to underscore
    },
    "returns": {},  # Already using underscores
    "settlement": {}  # Already using underscores
}

# Output directories
MASTER_DIR = os.path.join('reconciliation', 'output', 'master')
ANALYSIS_DIR = os.path.join('reconciliation', 'output', 'analysis')
REPORT_DIR = os.path.join('reconciliation', 'output', 'reports')
VISUALIZATION_DIR = os.path.join('reconciliation', 'output', 'visualizations')

# Master file paths
ORDERS_MASTER = os.path.join(MASTER_DIR, 'orders_master.csv')
RETURNS_MASTER = os.path.join(MASTER_DIR, 'returns_master.csv')
SETTLEMENT_MASTER = os.path.join(MASTER_DIR, 'settlement_master.csv')

# Analysis and report file paths
ANALYSIS_OUTPUT = os.path.join(ANALYSIS_DIR, 'order_analysis.csv')
REPORT_OUTPUT = os.path.join(REPORT_DIR, 'reconciliation_report.txt')
ANOMALIES_OUTPUT = os.path.join(REPORT_DIR, 'anomalies.csv')

def ensure_directories_exist() -> None:
    """Ensure all required directories exist."""
    directories = [
        MASTER_DIR,
        ANALYSIS_DIR,
        REPORT_DIR,
        VISUALIZATION_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def read_file(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file and return a DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the file contents
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read file: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()

def get_processed_files(master_file: str) -> List[str]:
    """
    Get list of already processed files from master file.
    
    Args:
        master_file: Path to the master file
        
    Returns:
        List of file identifiers
    """
    if not os.path.exists(master_file):
        return []
    
    try:
        df = pd.read_csv(master_file)
        if 'source_file' in df.columns:
            return df['source_file'].unique().tolist()
        return []
    except Exception as e:
        logger.error(f"Error reading processed files from {master_file}: {e}")
        return []

def extract_date_from_filename(filename: str, pattern: str) -> Tuple[str, str]:
    """
    Extract month and year from filename.
    
    Args:
        filename: Name of the file
        pattern: Regex pattern to match
        
    Returns:
        Tuple of (month, year)
    """
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return "", ""

def get_file_identifier(month: str, year: str) -> str:
    """
    Generate a unique identifier for a file.
    
    Args:
        month: Month from filename
        year: Year from filename
        
    Returns:
        String identifier
    """
    return f"{year}-{month}" 