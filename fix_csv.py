import pandas as pd
import logging
import csv
from datetime import datetime
from pathlib import Path
import sys
import io

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from schemas import (
    ORDERS_SCHEMA, RETURNS_SCHEMA, SETTLEMENT_SCHEMA,
    DATE_TYPE, STRING_TYPE, INTEGER_TYPE, FLOAT_TYPE, BOOLEAN_TYPE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_schema_for_file(filename):
    """Get the appropriate schema based on filename."""
    if 'orders' in filename.lower():
        return ORDERS_SCHEMA
    elif 'returns' in filename.lower():
        return RETURNS_SCHEMA
    elif 'settlement' in filename.lower():
        return SETTLEMENT_SCHEMA
    raise ValueError(f"Unknown file type: {filename}")

def convert_column_type(series, col_type):
    """Convert column to appropriate type with error handling."""
    try:
        if col_type == DATE_TYPE:
            return pd.to_datetime(series, errors='coerce')
        elif col_type == INTEGER_TYPE:
            return pd.to_numeric(series, errors='coerce').astype('Int64')
        elif col_type == FLOAT_TYPE:
            return pd.to_numeric(series, errors='coerce').astype('float64')
        elif col_type == BOOLEAN_TYPE:
            return series.map({'TRUE': True, 'True': True, '1': True, 
                             'FALSE': False, 'False': False, '0': False}).fillna(False)
        else:  # STRING_TYPE
            return series.astype(str).replace('nan', '')
    except Exception as e:
        logger.warning(f"Error converting column type: {str(e)}")
        return series

def preprocess_csv(input_file):
    """Preprocess CSV file to handle line breaks within fields."""
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Get header line
    header = content.split('\n')[0]
    expected_columns = len(header.split(','))
    
    # Process the rest of the content
    lines = []
    current_line = []
    in_quotes = False
    buffer = ""
    
    for char in content:
        if char == '"':
            in_quotes = not in_quotes
            buffer += char
        elif char == '\n' and in_quotes:
            buffer += ' '  # Replace newline with space within quotes
        elif char == '\n' and not in_quotes:
            buffer += char
            if buffer.count(',') == expected_columns - 1:
                lines.append(buffer.strip())
                buffer = ""
        else:
            buffer += char
    
    if buffer:
        lines.append(buffer.strip())
    
    return '\n'.join(lines)

def fix_csv_file(input_file, output_file):
    try:
        logger.info(f"Reading {input_file}")
        schema = get_schema_for_file(input_file)
        
        # Preprocess the CSV file
        processed_content = preprocess_csv(input_file)
        
        # Read the preprocessed content
        df = pd.read_csv(
            io.StringIO(processed_content),
            dtype=str,
            quoting=csv.QUOTE_MINIMAL,
            escapechar='\\'
        )
        
        # Clean up column names
        df.columns = df.columns.str.strip()
        
        # Add missing required columns with default values
        for col_name, col_info in schema.items():
            if col_name not in df.columns:
                if col_info['type'] == DATE_TYPE:
                    df[col_name] = pd.Timestamp.now()
                elif col_info['type'] == INTEGER_TYPE:
                    df[col_name] = 0
                elif col_info['type'] == FLOAT_TYPE:
                    df[col_name] = 0.0
                elif col_info['type'] == BOOLEAN_TYPE:
                    df[col_name] = False
                else:  # STRING_TYPE
                    df[col_name] = ''
                logger.info(f"Added missing column: {col_name}")
        
        # Convert each column to its proper type
        for col_name, col_info in schema.items():
            if col_name in df.columns:
                df[col_name] = convert_column_type(df[col_name], col_info['type'])
        
        # Add upload_timestamp if missing
        if 'upload_timestamp' not in df.columns:
            df['upload_timestamp'] = pd.Timestamp.now()
        
        # Remove any completely empty rows
        df = df.dropna(how='all')
        
        # Save with proper formatting
        logger.info(f"Saving to {output_file}")
        df.to_csv(
            output_file,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            escapechar='\\',
            date_format='%Y-%m-%d %H:%M:%S'
        )
        logger.info("File fixed successfully")
        
    except Exception as e:
        logger.error(f"Error fixing CSV file: {str(e)}")
        raise

if __name__ == "__main__":
    fix_csv_file('output/master/orders_master.csv', 'output/master/orders_master_fixed.csv')
    fix_csv_file('output/master/returns_master.csv', 'output/master/returns_master_fixed.csv')
    fix_csv_file('output/master/settlement_master.csv', 'output/master/settlement_master_fixed.csv') 