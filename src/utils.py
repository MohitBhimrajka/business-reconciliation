"""
Utility functions for the reconciliation application.
"""
import os
import re
import logging
from typing import List, Dict, Set, Optional
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory paths
BASE_DIR = Path('reconciliation')
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
ANALYSIS_DIR = OUTPUT_DIR / 'analysis'
REPORT_DIR = OUTPUT_DIR / 'reports'
VISUALIZATION_DIR = OUTPUT_DIR / 'visualizations'

# File paths
ORDERS_MASTER = OUTPUT_DIR / 'orders_master.csv'
RETURNS_MASTER = OUTPUT_DIR / 'returns_master.csv'
SETTLEMENT_MASTER = OUTPUT_DIR / 'settlement_master.csv'
ANALYSIS_OUTPUT = ANALYSIS_DIR / 'order_analysis.csv'
REPORT_OUTPUT = REPORT_DIR / 'reconciliation_report.txt'
ANOMALIES_OUTPUT = ANALYSIS_DIR / 'anomalies.csv'

# File patterns
ORDERS_PATTERN = r'orders-(\d{2})-(\d{4})\.(csv|xlsx)$'
RETURNS_PATTERN = r'returns-(\d{2})-(\d{4})\.(csv|xlsx)$'
SETTLEMENT_PATTERN = r'settlement-(\d{2})-(\d{4})\.(csv|xlsx)$'

# Required columns for each file type based on analysis.py logic
REQUIRED_COLUMNS = {
    'orders': {
        # Core columns required for status and financial calculations
        'order release id',     # Primary key for order identification
        'is_ship_rel',          # Required for determining fulfilled orders
        'final amount',         # Required for profit/loss calculation
        'total mrp'            # Required for profit/loss calculation
    },
    'returns': {
        # Core columns required for return processing
        'order_release_id',           # Primary key for order identification
        'total_actual_settlement'    # Required for return settlement calculation
    },
    'settlement': {
        # Core columns required for settlement processing
        'order_release_id',           # Primary key for order identification
        'total_actual_settlement'    # Required for settlement calculation
    }
}

# Column renaming mapping for standardization
COLUMN_RENAMES = {
    'orders': {
        'order release id': 'order_release_id',
        'final amount': 'final_amount',
        'total mrp': 'total_mrp'
    },
    'returns': {},  # Already using underscores
    'settlement': {}  # Already using underscores
}

def ensure_directories_exist() -> None:
    """Ensure all required directories exist."""
    for directory in [DATA_DIR, ANALYSIS_DIR, REPORT_DIR, VISUALIZATION_DIR]:
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

def validate_file_columns(df: pd.DataFrame, file_type: str) -> bool:
    """
    Validate that the DataFrame has all required columns for the given file type.
    
    Args:
        df: DataFrame to validate
        file_type: Type of file ('orders', 'returns', or 'settlement')
    
    Returns:
        True if all required columns are present, False otherwise
    
    Note:
        This function checks for the core columns required by analysis.py.
        Only columns that are absolutely necessary for the analysis logic
        are marked as required.
    """
    if file_type not in REQUIRED_COLUMNS:
        raise ValueError(f"Invalid file type: {file_type}")
    
    df_columns = set(df.columns)
    required_columns = REQUIRED_COLUMNS[file_type]
    
    # Check if all required columns are present
    missing_columns = required_columns - df_columns
    
    if missing_columns:
        logger.error(f"Missing required columns for {file_type}: {missing_columns}")
        return False
    
    return True

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