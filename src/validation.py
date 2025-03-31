"""
Data validation module for the reconciliation application.
"""
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import re

from schemas import (
    ORDERS_SCHEMA, RETURNS_SCHEMA, SETTLEMENT_SCHEMA,
    DATE_TYPE, STRING_TYPE, INTEGER_TYPE, FLOAT_TYPE, BOOLEAN_TYPE,
    COLUMN_RENAMES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class ValidationResult:
    """Class to hold validation results and errors."""
    def __init__(self):
        self.is_valid = True
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {
            'total_rows': 0,
            'valid_rows': 0,
            'invalid_rows': 0,
            'missing_values': {},
            'invalid_values': {},
            'duplicate_records': []
        }
    
    def add_error(self, row_index: int, column: str, message: str, value: Any = None):
        """Add a validation error."""
        self.errors.append({
            'row_index': row_index,
            'column': column,
            'message': message,
            'value': value
        })
        self.is_valid = False
        logger.error(f"Validation error in row {row_index}, column {column}: {message}")
    
    def add_warning(self, row_index: int, column: str, message: str, value: Any = None):
        """Add a validation warning."""
        self.warnings.append({
            'row_index': row_index,
            'column': column,
            'message': message,
            'value': value
        })
        logger.warning(f"Validation warning in row {row_index}, column {column}: {message}")
    
    def update_stats(self, total_rows: int, valid_rows: int):
        """Update validation statistics."""
        self.stats['total_rows'] = total_rows
        self.stats['valid_rows'] = valid_rows
        self.stats['invalid_rows'] = total_rows - valid_rows
        logger.info(f"Validation stats: {self.stats}")

def validate_date(value: Any, column: str) -> Optional[datetime]:
    """Validate and convert date values."""
    if pd.isna(value) or value == '':
        # For required date columns, use a default date
        if column in ['return_date', 'created_on', 'upload_timestamp']:
            return pd.Timestamp.now()
        return pd.NaT
    
    try:
        if isinstance(value, str):
            # Clean the string value
            value = value.strip()
            
            # Handle time-only format (e.g., "39:49.0")
            if re.match(r'^\d{2}:\d{2}\.0$', value):
                # For time-only values, use a default date of today
                minutes, seconds = map(float, value.rstrip('.0').split(':'))
                total_minutes = int(minutes)
                total_seconds = int(seconds)
                return datetime.now().replace(
                    hour=total_minutes // 60,
                    minute=total_minutes % 60,
                    second=total_seconds,
                    microsecond=0
                )
            
            # Try multiple date formats
            date_formats = [
                '%Y-%m-%d',
                '%d-%m-%Y',
                '%Y/%m/%d',
                '%d/%m/%Y',
                '%Y-%m-%d %H:%M:%S',
                '%d-%m-%Y %H:%M:%S',
                '%Y/%m/%d %H:%M:%S',
                '%d/%m/%Y %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%d-%m-%Y %H:%M',
                '%Y/%m/%d %H:%M',
                '%d/%m/%Y %H:%M',
                '%Y%m%d',
                '%d%m%Y',
                '%m/%d/%y %H:%M',
                '%m/%d/%Y %H:%M'
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            
            # If no format matches, try pandas datetime parsing
            try:
                return pd.to_datetime(value).to_pydatetime()
            except:
                logger.warning(f"Invalid date format for {column}: {value}, using default date")
                return pd.Timestamp.now()
                
        elif isinstance(value, datetime):
            return value
        elif isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        else:
            logger.warning(f"Invalid date type for {column}: {type(value)}, using default date")
            return pd.Timestamp.now()
    except Exception as e:
        logger.warning(f"Date validation failed for {column}: {str(e)}, using default date")
        return pd.Timestamp.now()

def validate_numeric(value: Any, column: str, target_type: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> Optional[float]:
    """Validate and convert numeric values."""
    if pd.isna(value) or value == '':
        return 0 if target_type == INTEGER_TYPE else 0.0
    
    try:
        # Handle string values with currency symbols and commas
        if isinstance(value, str):
            # Remove currency symbols and commas
            value = value.replace('₹', '').replace(',', '').strip()
            # Handle percentage values
            if value.endswith('%'):
                value = value[:-1]
                value = float(value) / 100
            elif value == '':
                return 0 if target_type == INTEGER_TYPE else 0.0
            else:
                value = float(value)
        else:
            # Convert to float first
            value = float(value)
        
        # Check range if specified
        if min_value is not None and value < min_value:
            logger.warning(f"Value {value} in column {column} is below minimum {min_value}, setting to minimum")
            value = min_value
        if max_value is not None and value > max_value:
            logger.warning(f"Value {value} in column {column} is above maximum {max_value}, setting to maximum")
            value = max_value
        
        # Convert to target type if needed
        if target_type == INTEGER_TYPE:
            return int(value)
        return value
    except (ValueError, TypeError) as e:
        logger.warning(f"Numeric validation failed for {column}: {str(e)}, using default value")
        return 0 if target_type == INTEGER_TYPE else 0.0
    except Exception as e:
        logger.warning(f"Unexpected error in numeric validation for {column}: {str(e)}, using default value")
        return 0 if target_type == INTEGER_TYPE else 0.0

def validate_string(value: Any, column: str, max_length: Optional[int] = None, pattern: Optional[str] = None) -> Optional[str]:
    """Validate and convert string values."""
    if pd.isna(value) or value == '':
        return ''
    
    try:
        str_value = str(value).strip()
        
        # Check length if specified
        if max_length is not None and len(str_value) > max_length:
            logger.warning(f"String length {len(str_value)} exceeds maximum {max_length} for {column}, truncating")
            str_value = str_value[:max_length]
        
        # Check pattern if specified
        if pattern is not None:
            if not re.match(pattern, str_value):
                logger.warning(f"String does not match pattern {pattern} for {column}, using empty string")
                return ''
        
        return str_value
    except Exception as e:
        logger.warning(f"String validation failed for {column}: {str(e)}, using empty string")
        return ''

def validate_boolean(value: Any, column: str) -> Optional[bool]:
    """Validate and convert boolean values."""
    if pd.isna(value) or value == '':
        return False
    
    try:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value = value.lower().strip()
            if value in ['true', '1', 'yes', 'y']:
                return True
            if value in ['false', '0', 'no', 'n']:
                return False
            logger.warning(f"Invalid boolean string value for {column}: {value}, using False")
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        logger.warning(f"Invalid boolean value type for {column}: {type(value)}, using False")
        return False
    except Exception as e:
        logger.warning(f"Boolean validation failed for {column}: {str(e)}, using False")
        return False

def validate_gst_number(value: Any, column: str) -> Optional[str]:
    """Validate GST number format."""
    if pd.isna(value) or value == '':
        return ''
    
    try:
        str_value = str(value).strip()
        # GST number format: 2 digits state code + 10 digits PAN + 1 digit entity number + 1 digit Z
        if not (len(str_value) == 15 and str_value[:2].isdigit() and str_value[2:12].isalnum() and str_value[12:].isdigit()):
            logger.warning(f"Invalid GST number format for {column}: {value}, using empty string")
            return ''
        return str_value
    except Exception as e:
        logger.warning(f"GST number validation failed for {column}: {str(e)}, using empty string")
        return ''

def validate_pincode(value: Any, column: str) -> Optional[str]:
    """Validate Indian PIN code format."""
    if pd.isna(value) or value == '':
        return ''
    
    try:
        str_value = str(value).strip()
        if not (len(str_value) == 6 and str_value.isdigit()):
            logger.warning(f"Invalid PIN code format for {column}: {value}, using empty string")
            return ''
        return str_value
    except Exception as e:
        logger.warning(f"PIN code validation failed for {column}: {str(e)}, using empty string")
        return ''

def validate_dataframe(df: pd.DataFrame, schema: Dict, file_type: str) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Validate a DataFrame against a schema and return the validated DataFrame.
    """
    errors = []
    warnings = []
    
    # Create a copy to avoid modifying the original
    df_validated = df.copy()
    
    # Get column renames for this file type
    rename_map = COLUMN_RENAMES.get(file_type, {})
    
    # First, rename columns according to mapping
    df_validated = df_validated.rename(columns=rename_map)
    
    # Add upload_timestamp if missing
    if 'upload_timestamp' not in df_validated.columns:
        df_validated['upload_timestamp'] = pd.Timestamp.now()
        warnings.append({
            'row': 0,
            'column': 'upload_timestamp',
            'message': 'Added missing upload_timestamp column with current timestamp'
        })
    
    # Define essential columns for each file type
    essential_columns = {
        "orders": [
            'order_release_id', 'order_line_id', 'seller_order_id', 'order_status',
            'packet_id', 'seller_packe_id', 'created_on', 'core_item_id',
            'seller_sku_code', 'myntra_sku_code', 'brand', 'style_name',
            'order_tracking_number', 'is_ship_rel'
        ],
        "returns": [
            'order_release_id', 'return_type', 'return_date', 'customer_paid_amount',
            'return_id'
        ],
        "settlement": [
            'order_release_id', 'order_line_id', 'return_type', 'return_date',
            'packet_id', 'customer_paid_amount'
        ]
    }
    
    # Get essential columns for this file type
    required_columns = essential_columns.get(file_type, [])
    
    # Check for missing required columns
    missing_columns = [col for col in required_columns if col not in df_validated.columns]
    if missing_columns:
        # Try to find columns with similar names
        for col in missing_columns:
            similar_cols = [c for c in df_validated.columns if col.lower() in c.lower() or c.lower() in col.lower()]
            if similar_cols:
                warnings.append({
                    'row': 0,
                    'column': col,
                    'message': f'Found similar columns: {similar_cols}. Please check if these should be mapped to {col}.'
                })
            else:
                errors.append({
                    'row': 0,
                    'column': col,
                    'message': f'Missing required column: {col}'
                })
    
    # Process each column according to schema
    for col_name, col_info in schema.items():
        if col_name not in df_validated.columns:
            if col_info['required']:
                errors.append({
                    'row': 0,
                    'column': col_name,
                    'message': f'Missing required column: {col_name}'
                })
            continue
        
        target_type = col_info['type']
        
        try:
            if target_type == DATE_TYPE:
                df_validated[col_name] = df_validated[col_name].apply(validate_date)
            elif target_type in [INTEGER_TYPE, FLOAT_TYPE]:
                df_validated[col_name] = df_validated[col_name].apply(
                    lambda x: validate_numeric(x, target_type)
                )
            elif target_type == BOOLEAN_TYPE:
                df_validated[col_name] = df_validated[col_name].apply(validate_boolean)
            elif target_type == STRING_TYPE:
                df_validated[col_name] = df_validated[col_name].astype(str)
            
            # Check for required columns with null values
            if col_info['required'] and df_validated[col_name].isna().any():
                null_count = df_validated[col_name].isna().sum()
                errors.append({
                    'row': 0,
                    'column': col_name,
                    'message': f'Column {col_name} has {null_count} null values but is required'
                })
        except Exception as e:
            errors.append({
                'row': 0,
                'column': col_name,
                'message': f'Error processing column {col_name}: {str(e)}'
            })
    
    # Log warnings
    for warning in warnings:
        logger.warning(f"Validation warning in row {warning['row']}, column {warning['column']}: {warning['message']}")
    
    # Log errors
    for error in errors:
        logger.error(f"Validation error in row {error['row']}, column {error['column']}: {error['message']}")
    
    return df_validated, ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_master_file(file_path: Path, schema: Dict, file_type: str) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Validate master file and return validated DataFrame and results.
    """
    try:
        logger.info(f"Validating master file: {file_path}")
        df = pd.read_csv(file_path)
        return validate_dataframe(df, schema, file_type)
    except Exception as e:
        result = ValidationResult()
        result.add_error(0, 'file', f"Error reading master file: {str(e)}")
        logger.error(f"Error validating master file: {str(e)}")
        return pd.DataFrame(), result

def validate_new_file(file_path: Path, schema: Dict, file_type: str) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Validate new file and return validated DataFrame and results.
    """
    try:
        logger.info(f"Validating new file: {file_path}")
        df = pd.read_csv(file_path)
        return validate_dataframe(df, schema, file_type)
    except Exception as e:
        result = ValidationResult()
        result.add_error(0, 'file', f"Error reading new file: {str(e)}")
        logger.error(f"Error validating new file: {str(e)}")
        return pd.DataFrame(), result

def merge_master_files(
    master_df: pd.DataFrame,
    new_df: pd.DataFrame,
    file_type: str,
    key_columns: List[str]
) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Merge master and new DataFrames, handling duplicates and updates.
    """
    result = ValidationResult()
    
    try:
        logger.info(f"Merging master and new files for {file_type}")
        
        # Check for duplicate keys in new file
        duplicates = new_df[new_df.duplicated(subset=key_columns, keep=False)]
        if not duplicates.empty:
            for idx, row in duplicates.iterrows():
                result.add_warning(
                    idx,
                    'duplicate_key',
                    f"Duplicate key found: {', '.join(f'{k}={row[k]}' for k in key_columns)}"
                )
        
        # Merge DataFrames
        merged_df = pd.concat([master_df, new_df], ignore_index=True)
        
        # Keep latest record for each key combination
        merged_df = merged_df.sort_values('upload_timestamp', ascending=False)
        merged_df = merged_df.drop_duplicates(subset=key_columns, keep='first')
        
        # Update statistics
        result.stats['total_rows'] = len(merged_df)
        result.stats['valid_rows'] = len(merged_df)
        result.stats['new_records'] = len(new_df)
        result.stats['updated_records'] = len(merged_df) - len(master_df)
        
        logger.info(f"Successfully merged files. New total records: {len(merged_df)}")
        return merged_df, result
    
    except Exception as e:
        result.add_error(0, 'merge', f"Error merging files: {str(e)}")
        logger.error(f"Error merging files: {str(e)}")
        return pd.DataFrame(), result 