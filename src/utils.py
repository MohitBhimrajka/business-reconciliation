"""
Utility functions for the reconciliation application.
"""
import os
import re
import logging
from typing import List, Dict, Set, Tuple, Optional
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
MASTER_DIR = OUTPUT_DIR / 'master'
ANALYSIS_DIR = OUTPUT_DIR / 'analysis'
REPORT_DIR = OUTPUT_DIR / 'reports'
VISUALIZATION_DIR = OUTPUT_DIR / 'visualizations'

# File paths
ORDERS_MASTER = MASTER_DIR / 'orders_master.csv'
RETURNS_MASTER = MASTER_DIR / 'returns_master.csv'
SETTLEMENT_MASTER = MASTER_DIR / 'settlement_master.csv'
ANALYSIS_OUTPUT = ANALYSIS_DIR / 'order_analysis.csv'
REPORT_OUTPUT = REPORT_DIR / 'reconciliation_report.txt'
ANOMALIES_OUTPUT = ANALYSIS_DIR / 'anomalies.csv'

# File patterns
ORDERS_PATTERN = r'orders-(\d{2})-(\d{4})\.(csv|xlsx)$'
RETURNS_PATTERN = r'returns-(\d{2})-(\d{4})\.(csv|xlsx)$'
SETTLEMENT_PATTERN = r'settlement-(\d{2})-(\d{4})\.(csv|xlsx)$'

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

def ensure_directories_exist() -> None:
    """Ensure all required directories exist."""
    for directory in [DATA_DIR, MASTER_DIR, ANALYSIS_DIR, REPORT_DIR, VISUALIZATION_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def read_file(file_path: Path) -> pd.DataFrame:
    """
    Read a CSV or Excel file into a DataFrame.
    
    Args:
        file_path: Path to the file
    
    Returns:
        DataFrame containing the file contents
    """
    if not file_path.exists():
        return pd.DataFrame()
    
    if file_path.suffix.lower() == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

def get_processed_files() -> List[Path]:
    """
    Get list of processed files in the data directory.
    
    Returns:
        List of Path objects for processed files
    """
    if not DATA_DIR.exists():
        return []
    
    processed_files = []
    for pattern in [ORDERS_PATTERN, RETURNS_PATTERN, SETTLEMENT_PATTERN]:
        for file in DATA_DIR.glob('*'):
            if re.match(pattern, file.name):
                processed_files.append(file)
    
    return processed_files

def extract_date_from_filename(filename: str) -> Optional[tuple]:
    """
    Extract month and year from filename.
    
    Args:
        filename: Name of the file
    
    Returns:
        Tuple of (month, year) if found, None otherwise
    """
    for pattern in [ORDERS_PATTERN, RETURNS_PATTERN, SETTLEMENT_PATTERN]:
        match = re.match(pattern, filename)
        if match:
            return match.groups()[:2]
    return None

def get_file_identifier(file_type: str, month: str, year: str) -> str:
    """
    Generate standard filename for a given file type, month, and year.
    
    Args:
        file_type: Type of file (orders, returns, settlement)
        month: Month (01-12)
        year: Year (YYYY)
    
    Returns:
        Standardized filename
    """
    return f"{file_type}-{month}-{year}.csv"

def validate_file_columns(df: pd.DataFrame, file_type: str) -> bool:
    """
    Validate if DataFrame has required columns for file type.
    
    Args:
        df: DataFrame to validate
        file_type: Type of file (orders, returns, settlement)
    
    Returns:
        True if valid, False otherwise
    """
    required_columns = {
        'orders': ['order_release_id', 'order_status', 'final_amount', 'total_mrp'],
        'returns': ['order_release_id', 'return_amount'],
        'settlement': ['order_release_id', 'settlement_amount']
    }
    
    if file_type not in required_columns:
        return False
    
    return all(col in df.columns for col in required_columns[file_type])

def format_currency(value: float) -> str:
    """
    Format a number as currency.
    
    Args:
        value: Number to format
    
    Returns:
        Formatted currency string
    """
    return f"₹{value:,.2f}"

def format_percentage(value: float) -> str:
    """
    Format a number as percentage.
    
    Args:
        value: Number to format
    
    Returns:
        Formatted percentage string
    """
    return f"{value:.2f}%" 