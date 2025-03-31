"""
Data ingestion module for the reconciliation application.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (
    ensure_directories_exist, DATA_DIR, ORDERS_MASTER, RETURNS_MASTER,
    SETTLEMENT_MASTER, ORDERS_PATTERN, RETURNS_PATTERN, SETTLEMENT_PATTERN,
    COLUMN_RENAMES, read_file
)
from schemas import ORDERS_SCHEMA, RETURNS_SCHEMA, SETTLEMENT_SCHEMA
from validation import (
    validate_master_file, validate_new_file, merge_master_files,
    ValidationResult, ValidationError, validate_dataframe
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for batch processing
BATCH_SIZE = 1000  # Number of rows to process in each batch
MAX_WORKERS = 4  # Maximum number of parallel workers

def process_file_in_batches(file_path: Path, schema: Dict, file_type: str) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Process a file in batches to handle large files efficiently.
    """
    try:
        logger.info(f"Processing {file_path} in batches")
        
        # Read the file in chunks
        chunks = pd.read_csv(file_path, chunksize=BATCH_SIZE)
        
        # Process each chunk
        processed_chunks = []
        validation_result = ValidationResult()
        
        for chunk in chunks:
            # Validate and convert the chunk
            processed_chunk, chunk_result = validate_dataframe(chunk, schema, file_type)
            
            # Merge validation results
            validation_result.errors.extend(chunk_result.errors)
            validation_result.warnings.extend(chunk_result.warnings)
            validation_result.stats['total_rows'] += chunk_result.stats['total_rows']
            validation_result.stats['valid_rows'] += chunk_result.stats['valid_rows']
            
            processed_chunks.append(processed_chunk)
        
        # Combine processed chunks
        if processed_chunks:
            final_df = pd.concat(processed_chunks, ignore_index=True)
            validation_result.stats['invalid_rows'] = validation_result.stats['total_rows'] - validation_result.stats['valid_rows']
            return final_df, validation_result
        else:
            validation_result.add_error(0, 'process', "No valid data found in file")
            return pd.DataFrame(), validation_result
            
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        validation_result = ValidationResult()
        validation_result.add_error(0, 'process', str(e))
        return pd.DataFrame(), validation_result

def process_orders_file(file_path: Path) -> Tuple[bool, ValidationResult]:
    """
    Process orders file with validation and master file update.
    """
    try:
        logger.info(f"Processing orders file: {file_path}")
        
        # Validate new file
        new_df, validation_result = process_file_in_batches(
            file_path, ORDERS_SCHEMA, 'orders'
        )
        
        if not validation_result.is_valid:
            logger.error(f"Validation failed for {file_path}")
            return False, validation_result
        
        # Load and validate master file if exists
        master_df = pd.DataFrame()
        if ORDERS_MASTER.exists():
            master_df, master_result = validate_master_file(
                ORDERS_MASTER, ORDERS_SCHEMA, 'orders'
            )
            if not master_result.is_valid:
                logger.error("Master file validation failed")
                return False, master_result
        
        # Merge files
        merged_df, merge_result = merge_master_files(
            master_df, new_df, 'orders',
            key_columns=['order_release_id', 'order_line_id']
        )
        
        if not merge_result.is_valid:
            logger.error("File merge failed")
            return False, merge_result
        
        # Save updated master file
        merged_df.to_csv(ORDERS_MASTER, index=False)
        logger.info(f"Successfully processed {file_path}")
        
        return True, merge_result
    
    except Exception as e:
        logger.error(f"Error processing orders file: {e}")
        result = ValidationResult()
        result.add_error(0, 'process', str(e))
        return False, result

def process_returns_file(file_path: Path) -> Tuple[bool, ValidationResult]:
    """
    Process returns file with validation and master file update.
    """
    try:
        logger.info(f"Processing returns file: {file_path}")
        
        # Validate new file
        new_df, validation_result = process_file_in_batches(
            file_path, RETURNS_SCHEMA, 'returns'
        )
        
        if not validation_result.is_valid:
            logger.error(f"Validation failed for {file_path}")
            return False, validation_result
        
        # Load and validate master file if exists
        master_df = pd.DataFrame()
        if RETURNS_MASTER.exists():
            master_df, master_result = validate_master_file(
                RETURNS_MASTER, RETURNS_SCHEMA, 'returns'
            )
            if not master_result.is_valid:
                logger.error("Master file validation failed")
                return False, master_result
        
        # Merge files
        merged_df, merge_result = merge_master_files(
            master_df, new_df, 'returns',
            key_columns=['order_release_id', 'order_line_id']
        )
        
        if not merge_result.is_valid:
            logger.error("File merge failed")
            return False, merge_result
        
        # Save updated master file
        merged_df.to_csv(RETURNS_MASTER, index=False)
        logger.info(f"Successfully processed {file_path}")
        
        return True, merge_result
    
    except Exception as e:
        logger.error(f"Error processing returns file: {e}")
        result = ValidationResult()
        result.add_error(0, 'process', str(e))
        return False, result

def process_settlement_file(file_path: Path) -> Tuple[bool, ValidationResult]:
    """
    Process settlement file with validation and master file update.
    """
    try:
        logger.info(f"Processing settlement file: {file_path}")
        
        # Validate new file
        new_df, validation_result = process_file_in_batches(
            file_path, SETTLEMENT_SCHEMA, 'settlement'
        )
        
        if not validation_result.is_valid:
            logger.error(f"Validation failed for {file_path}")
            return False, validation_result
        
        # Load and validate master file if exists
        master_df = pd.DataFrame()
        if SETTLEMENT_MASTER.exists():
            master_df, master_result = validate_master_file(
                SETTLEMENT_MASTER, SETTLEMENT_SCHEMA, 'settlement'
            )
            if not master_result.is_valid:
                logger.error("Master file validation failed")
                return False, master_result
        
        # Merge files
        merged_df, merge_result = merge_master_files(
            master_df, new_df, 'settlement',
            key_columns=['order_release_id', 'order_line_id']
        )
        
        if not merge_result.is_valid:
            logger.error("File merge failed")
            return False, merge_result
        
        # Save updated master file
        merged_df.to_csv(SETTLEMENT_MASTER, index=False)
        logger.info(f"Successfully processed {file_path}")
        
        return True, merge_result
    
    except Exception as e:
        logger.error(f"Error processing settlement file: {e}")
        result = ValidationResult()
        result.add_error(0, 'process', str(e))
        return False, result

def process_new_files() -> Dict[str, List[ValidationResult]]:
    """
    Process all new files in the data directory.
    Returns a dictionary of results for each file type.
    """
    ensure_directories_exist()
    
    results = {
        'orders': [],
        'returns': [],
        'settlement': []
    }
    
    try:
        # Process each file in the data directory
        for file_path in DATA_DIR.glob('*.csv'):
            file_name = file_path.name.lower()
            
            if ORDERS_PATTERN.match(file_name):
                success, result = process_orders_file(file_path)
                results['orders'].append(result)
            elif RETURNS_PATTERN.match(file_name):
                success, result = process_returns_file(file_path)
                results['returns'].append(result)
            elif SETTLEMENT_PATTERN.match(file_name):
                success, result = process_settlement_file(file_path)
                results['settlement'].append(result)
            else:
                logger.warning(f"Unknown file type: {file_name}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing new files: {e}")
        return results 