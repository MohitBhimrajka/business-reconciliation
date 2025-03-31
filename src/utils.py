"""
Utility functions for the reconciliation application.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging
import re

from schemas import (
    ORDERS_SCHEMA, RETURNS_SCHEMA, SETTLEMENT_SCHEMA,
    COLUMN_RENAMES, DATE_TYPE, STRING_TYPE, INTEGER_TYPE,
    FLOAT_TYPE, BOOLEAN_TYPE
)
from validation import (
    validate_dataframe, validate_master_file, validate_new_file,
    merge_master_files, ValidationResult, ValidationError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MASTER_DIR = OUTPUT_DIR / "master"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
REPORT_DIR = OUTPUT_DIR / "reports"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

# File paths
ORDERS_MASTER = MASTER_DIR / "orders_master.csv"
RETURNS_MASTER = MASTER_DIR / "returns_master.csv"
SETTLEMENT_MASTER = MASTER_DIR / "settlement_master.csv"
ANALYSIS_OUTPUT = ANALYSIS_DIR / "order_analysis.csv"
ANOMALIES_OUTPUT = ANALYSIS_DIR / "anomalies.csv"
REPORT_OUTPUT = REPORT_DIR / "reconciliation_report.csv"

# File patterns
ORDERS_PATTERN = "orders-*.csv"
RETURNS_PATTERN = "returns-*.csv"
SETTLEMENT_PATTERN = "settlement-*.csv"

def ensure_directories_exist() -> None:
    """Create necessary directories if they don't exist."""
    for directory in [DATA_DIR, OUTPUT_DIR, MASTER_DIR, ANALYSIS_DIR, REPORT_DIR, VISUALIZATION_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def convert_date(date_str: str) -> pd.Timestamp:
    """
    Convert date string to pandas Timestamp.
    Handles various date formats and invalid dates.
    """
    if pd.isna(date_str):
        return pd.NaT
    
    try:
        # Handle different date formats
        if isinstance(date_str, str):
            # Remove any extra quotes
            date_str = date_str.strip('"').strip()
            
            # Handle time-only format (e.g., "39:49.0")
            if re.match(r'^\d{2}:\d{2}\.0$', date_str):
                minutes, seconds = map(float, date_str.rstrip('.0').split(':'))
                total_minutes = int(minutes)
                total_seconds = int(seconds)
                return pd.Timestamp.now().replace(
                    hour=total_minutes // 60,
                    minute=total_minutes % 60,
                    second=total_seconds,
                    microsecond=0
                )
            
            # Try different date formats
            for fmt in ['%m/%d/%y %H:%M', '%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S']:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except ValueError:
                    continue
            # If no format matches, try default parsing
            return pd.to_datetime(date_str)
        return pd.to_datetime(date_str)
    except Exception as e:
        logger.warning(f"Failed to convert date: {date_str}. Error: {e}")
        return pd.NaT

def convert_numeric(value: Union[str, float, int], target_type: str) -> Union[float, int, None]:
    """
    Convert string to numeric value with proper type handling.
    """
    if pd.isna(value):
        return None
    
    try:
        if isinstance(value, str):
            # Remove any extra quotes and whitespace
            value = value.strip('"').strip()
            if value == '':
                return None
        
        # Convert to numeric value
        numeric_value = pd.to_numeric(value, errors='coerce')
        
        if target_type == INTEGER_TYPE:
            # For integers, convert to int64 and handle NaN
            if pd.isna(numeric_value):
                return None
            return int(numeric_value)
        else:
            # For floats, return as is
            return float(numeric_value) if not pd.isna(numeric_value) else None
            
    except Exception as e:
        logger.warning(f"Failed to convert numeric value: {value}. Error: {e}")
        return None

def convert_boolean(value: Union[str, bool]) -> bool:
    """
    Convert string to boolean value.
    """
    if pd.isna(value):
        return False
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        value = value.strip().upper()
        return value in ['TRUE', '1', 'YES', 'Y']
    
    return False

def validate_and_convert_dataframe(
    df: pd.DataFrame,
    schema: Dict,
    file_type: str
) -> pd.DataFrame:
    """
    Validate and convert DataFrame columns according to schema.
    """
    # Rename columns according to mapping
    rename_map = COLUMN_RENAMES.get(file_type, {})
    df = df.rename(columns=rename_map)
    
    # Create a copy to avoid modifying the original
    df_converted = df.copy()
    
    # Track validation errors
    validation_errors = []
    
    # Process each column according to schema
    for col_name, col_info in schema.items():
        if col_name not in df_converted.columns:
            if col_info['required']:
                validation_errors.append(f"Missing required column: {col_name}")
            continue
        
        target_type = col_info['type']
        
        try:
            if target_type == DATE_TYPE:
                df_converted[col_name] = df_converted[col_name].apply(convert_date)
            elif target_type in [INTEGER_TYPE, FLOAT_TYPE]:
                df_converted[col_name] = df_converted[col_name].apply(
                    lambda x: convert_numeric(x, target_type)
                )
            elif target_type == BOOLEAN_TYPE:
                df_converted[col_name] = df_converted[col_name].apply(convert_boolean)
            elif target_type == STRING_TYPE:
                df_converted[col_name] = df_converted[col_name].astype(str)
            
            # Check for required columns with null values
            if col_info['required'] and df_converted[col_name].isna().any():
                null_count = df_converted[col_name].isna().sum()
                validation_errors.append(
                    f"Column {col_name} has {null_count} null values but is required"
                )
        except Exception as e:
            validation_errors.append(f"Error processing column {col_name}: {str(e)}")
    
    if validation_errors:
        logger.warning(f"Validation errors in {file_type} file:")
        for error in validation_errors:
            logger.warning(error)
    
    return df_converted

def read_file(file_path: Union[str, Path], file_type: Optional[str] = None) -> pd.DataFrame:
    """
    Read a file and return a DataFrame.
    """
    try:
        # Convert Path to string if needed
        file_path_str = str(file_path)
        
        if not os.path.exists(file_path_str):
            raise FileNotFoundError(f"File not found: {file_path_str}")
        
        # Determine file type from filename if not provided
        if file_type is None:
            if "orders" in file_path_str.lower():
                file_type = "orders"
            elif "returns" in file_path_str.lower():
                file_type = "returns"
            elif "settlement" in file_path_str.lower():
                file_type = "settlement"
            else:
                raise ValueError("Could not determine file type from filename")
        
        # Get schema based on file type
        schema = {
            "orders": ORDERS_SCHEMA,
            "returns": RETURNS_SCHEMA,
            "settlement": SETTLEMENT_SCHEMA
        }.get(file_type)
        
        if not schema:
            raise ValueError(f"Invalid file type: {file_type}")
        
        # First try reading with standard settings
        try:
            df = pd.read_csv(
                file_path_str,
                sep='\t',
                encoding='utf-8',
                dtype=str
            )
        except Exception as e:
            logger.warning(f"Standard CSV reading failed: {str(e)}. Trying alternative approach...")
            
            # Try alternative approach with more lenient settings
            try:
                # Read the file in chunks to handle large files
                chunks = []
                chunk_size = 1000  # Adjust based on your needs
                
                for chunk in pd.read_csv(
                    file_path_str,
                    sep='\t',
                    encoding='utf-8',
                    dtype=str,
                    chunksize=chunk_size,
                    quoting=3,  # QUOTE_NONE
                    escapechar='\\',
                    on_bad_lines='warn'
                ):
                    chunks.append(chunk)
                
                df = pd.concat(chunks, ignore_index=True)
                
            except Exception as e2:
                logger.error(f"Alternative CSV reading failed: {str(e2)}")
                raise
        
        # Clean up the DataFrame
        # Remove any rows where all values are NaN
        df = df.dropna(how='all')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Validate DataFrame against schema
        df, result = validate_dataframe(df, schema, file_type)
        
        if not result.is_valid:
            logger.warning(f"Validation errors in {file_type} file:")
            for error in result.errors:
                logger.warning(f"- {error['message']}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    return f"₹{amount:,.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage string."""
    return f"{value:.2f}%"

def get_file_identifier(file_path: Union[str, Path]) -> str:
    """Extract month and year from filename."""
    file_path = Path(file_path)
    filename = file_path.name
    # Extract MM-YYYY from filename
    parts = filename.split('-')
    if len(parts) >= 3:
        return f"{parts[1]}-{parts[2].split('.')[0]}"
    return "unknown" 