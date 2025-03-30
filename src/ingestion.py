"""
Data ingestion module for reconciliation application.
"""
import os
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path

from utils import (
    ensure_directories_exist, read_file, get_processed_files,
    ORDERS_PATTERN, RETURNS_PATTERN, SETTLEMENT_PATTERN,
    ORDERS_COLUMNS, RETURNS_COLUMNS, SETTLEMENT_COLUMNS,
    COLUMN_RENAMES, ORDERS_MASTER, RETURNS_MASTER, SETTLEMENT_MASTER,
    extract_date_from_filename, get_file_identifier, validate_file_columns
)

logger = logging.getLogger(__name__)

def scan_directory(directory: str) -> Dict[str, List[str]]:
    """
    Scan a directory for files matching the expected patterns.
    
    Args:
        directory: Directory to scan
        
    Returns:
        Dictionary with file types as keys and lists of file paths as values
    """
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return {"orders": [], "returns": [], "settlement": []}
    
    files = os.listdir(directory)
    
    import re
    
    orders_files = [os.path.join(directory, f) for f in files if re.match(ORDERS_PATTERN, f)]
    returns_files = [os.path.join(directory, f) for f in files if re.match(RETURNS_PATTERN, f)]
    settlement_files = [os.path.join(directory, f) for f in files if re.match(SETTLEMENT_PATTERN, f)]
    
    return {
        "orders": orders_files,
        "returns": returns_files,
        "settlement": settlement_files
    }

def process_orders_file(file_path: Path) -> bool:
    """
    Process an orders file and append to master file.
    
    Args:
        file_path: Path to the orders file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the file
        df = read_file(file_path)
        if df.empty:
            logger.error(f"Empty file: {file_path}")
            return False
        
        # Validate columns
        if not validate_file_columns(df, 'orders'):
            logger.error(f"Invalid columns in orders file: {file_path}")
            return False
        
        # Add source file information
        df['source_file'] = file_path.name
        
        # Append to master file
        if ORDERS_MASTER.exists():
            master_df = read_file(ORDERS_MASTER)
            # Remove duplicates based on order_release_id and source_file
            master_df = master_df[
                ~(master_df['order_release_id'].isin(df['order_release_id']) &
                  master_df['source_file'] == file_path.name)
            ]
            df = pd.concat([master_df, df], ignore_index=True)
        
        # Save to master file
        df.to_csv(ORDERS_MASTER, index=False)
        logger.info(f"Successfully processed orders file: {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing orders file {file_path}: {e}")
        return False

def process_returns_file(file_path: Path) -> bool:
    """
    Process a returns file and append to master file.
    
    Args:
        file_path: Path to the returns file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the file
        df = read_file(file_path)
        if df.empty:
            logger.error(f"Empty file: {file_path}")
            return False
        
        # Validate columns
        if not validate_file_columns(df, 'returns'):
            logger.error(f"Invalid columns in returns file: {file_path}")
            return False
        
        # Add source file information
        df['source_file'] = file_path.name
        
        # Append to master file
        if RETURNS_MASTER.exists():
            master_df = read_file(RETURNS_MASTER)
            # Remove duplicates based on order_release_id and source_file
            master_df = master_df[
                ~(master_df['order_release_id'].isin(df['order_release_id']) &
                  master_df['source_file'] == file_path.name)
            ]
            df = pd.concat([master_df, df], ignore_index=True)
        
        # Save to master file
        df.to_csv(RETURNS_MASTER, index=False)
        logger.info(f"Successfully processed returns file: {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing returns file {file_path}: {e}")
        return False

def process_settlement_file(file_path: Path) -> bool:
    """
    Process a settlement file and append to master file.
    
    Args:
        file_path: Path to the settlement file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the file
        df = read_file(file_path)
        if df.empty:
            logger.error(f"Empty file: {file_path}")
            return False
        
        # Validate columns
        if not validate_file_columns(df, 'settlement'):
            logger.error(f"Invalid columns in settlement file: {file_path}")
            return False
        
        # Add source file information
        df['source_file'] = file_path.name
        
        # Append to master file
        if SETTLEMENT_MASTER.exists():
            master_df = read_file(SETTLEMENT_MASTER)
            # Remove duplicates based on order_release_id and source_file
            master_df = master_df[
                ~(master_df['order_release_id'].isin(df['order_release_id']) &
                  master_df['source_file'] == file_path.name)
            ]
            df = pd.concat([master_df, df], ignore_index=True)
        
        # Save to master file
        df.to_csv(SETTLEMENT_MASTER, index=False)
        logger.info(f"Successfully processed settlement file: {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing settlement file {file_path}: {e}")
        return False

def process_file(file_path: Path, file_type: str) -> bool:
    """
    Process a file based on its type.
    
    Args:
        file_path: Path to the file
        file_type: Type of file (orders, returns, settlement)
    
    Returns:
        True if successful, False otherwise
    """
    if file_type == 'orders':
        return process_orders_file(file_path)
    elif file_type == 'returns':
        return process_returns_file(file_path)
    elif file_type == 'settlement':
        return process_settlement_file(file_path)
    else:
        logger.error(f"Invalid file type: {file_type}")
        return False

def ingest_data(data_directory: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ingest data from the specified directory and consolidate into master files.
    
    Args:
        data_directory: Directory containing the data files
        
    Returns:
        Tuple of DataFrames (orders, returns, settlement)
    """
    ensure_directories_exist()
    
    # Scan directory for files
    files = scan_directory(data_directory)
    
    # Get already processed files
    processed_orders = get_processed_files(ORDERS_MASTER)
    processed_returns = get_processed_files(RETURNS_MASTER)
    processed_settlements = get_processed_files(SETTLEMENT_MASTER)
    
    # Process orders files
    for file_path in files["orders"]:
        filename = os.path.basename(file_path)
        month, year = extract_date_from_filename(filename, ORDERS_PATTERN)
        file_id = get_file_identifier(month, year)
        
        if file_id in processed_orders:
            logger.info(f"Orders file already processed: {filename}")
            continue
        
        logger.info(f"Processing orders file: {filename}")
        process_file(Path(file_path), 'orders')
    
    # Process returns files
    for file_path in files["returns"]:
        filename = os.path.basename(file_path)
        month, year = extract_date_from_filename(filename, RETURNS_PATTERN)
        file_id = get_file_identifier(month, year)
        
        if file_id in processed_returns:
            logger.info(f"Returns file already processed: {filename}")
            continue
        
        logger.info(f"Processing returns file: {filename}")
        process_file(Path(file_path), 'returns')
    
    # Process settlement files
    for file_path in files["settlement"]:
        filename = os.path.basename(file_path)
        month, year = extract_date_from_filename(filename, SETTLEMENT_PATTERN)
        file_id = get_file_identifier(month, year)
        
        if file_id in processed_settlements:
            logger.info(f"Settlement file already processed: {filename}")
            continue
        
        logger.info(f"Processing settlement file: {filename}")
        process_file(Path(file_path), 'settlement')
    
    # Load and return the latest master data
    orders_df = pd.DataFrame()
    returns_df = pd.DataFrame()
    settlement_df = pd.DataFrame()
    
    if os.path.exists(ORDERS_MASTER):
        orders_df = pd.read_csv(ORDERS_MASTER)
    
    if os.path.exists(RETURNS_MASTER):
        returns_df = pd.read_csv(RETURNS_MASTER)
    
    if os.path.exists(SETTLEMENT_MASTER):
        settlement_df = pd.read_csv(SETTLEMENT_MASTER)
    
    return orders_df, returns_df, settlement_df 