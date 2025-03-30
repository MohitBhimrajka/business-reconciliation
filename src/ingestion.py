"""
Data ingestion and consolidation module.
"""
import os
import logging
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path

from utils import (
    ensure_directories_exist, read_file, get_processed_files,
    ORDERS_PATTERN, RETURNS_PATTERN, SETTLEMENT_PATTERN,
    ORDERS_COLUMNS, RETURNS_COLUMNS, SETTLEMENT_COLUMNS,
    COLUMN_RENAMES, ORDERS_MASTER, RETURNS_MASTER, SETTLEMENT_MASTER,
    extract_date_from_filename, get_file_identifier
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

def process_orders_file(file_path: str) -> pd.DataFrame:
    """
    Process an orders file and extract relevant columns.
    
    Args:
        file_path: Path to the orders file
        
    Returns:
        DataFrame with processed orders data
    """
    try:
        df = read_file(file_path)
        
        # Extract only the specified columns if they exist
        available_columns = [col for col in ORDERS_COLUMNS if col in df.columns]
        df = df[available_columns].copy()
        
        # Rename order release id column to standardize
        if 'order release id' in df.columns:
            df.rename(columns={'order release id': 'order_release_id'}, inplace=True)
        
        # Add source file information
        filename = os.path.basename(file_path)
        month, year = extract_date_from_filename(filename, ORDERS_PATTERN)
        df['source_file'] = get_file_identifier(month, year)
        
        # Ensure proper data types
        if 'order_release_id' in df.columns:
            df['order_release_id'] = df['order_release_id'].astype(str)
        if 'is_ship_rel' in df.columns:
            df['is_ship_rel'] = df['is_ship_rel'].astype(int)
        
        return df
    except Exception as e:
        logger.error(f"Error processing orders file {file_path}: {e}")
        return pd.DataFrame()

def process_returns_file(file_path: str) -> pd.DataFrame:
    """
    Process a returns file and extract relevant columns.
    
    Args:
        file_path: Path to the returns file
        
    Returns:
        DataFrame with processed returns data
    """
    try:
        df = read_file(file_path)
        
        # Extract only the specified columns if they exist
        available_columns = [col for col in RETURNS_COLUMNS if col in df.columns]
        df = df[available_columns].copy()
        
        # Rename columns according to standardization mapping
        for old_col, new_col in COLUMN_RENAMES["returns"].items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Add source file information
        filename = os.path.basename(file_path)
        month, year = extract_date_from_filename(filename, RETURNS_PATTERN)
        df['source_file'] = get_file_identifier(month, year)
        
        # Ensure proper data types
        if 'order_release_id' in df.columns:
            df['order_release_id'] = df['order_release_id'].astype(str)
        
        return df
    except Exception as e:
        logger.error(f"Error processing returns file {file_path}: {e}")
        return pd.DataFrame()

def process_settlement_file(file_path: str) -> pd.DataFrame:
    """
    Process a settlement file and extract relevant columns.
    
    Args:
        file_path: Path to the settlement file
        
    Returns:
        DataFrame with processed settlement data
    """
    try:
        df = read_file(file_path)
        
        # Extract only the specified columns if they exist
        available_columns = [col for col in SETTLEMENT_COLUMNS if col in df.columns]
        df = df[available_columns].copy()
        
        # Rename columns according to standardization mapping
        for old_col, new_col in COLUMN_RENAMES["settlement"].items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Add source file information
        filename = os.path.basename(file_path)
        month, year = extract_date_from_filename(filename, SETTLEMENT_PATTERN)
        df['source_file'] = get_file_identifier(month, year)
        
        # Ensure proper data types
        if 'order_release_id' in df.columns:
            df['order_release_id'] = df['order_release_id'].astype(str)
        
        return df
    except Exception as e:
        logger.error(f"Error processing settlement file {file_path}: {e}")
        return pd.DataFrame()

def append_to_master(df: pd.DataFrame, master_file: str) -> bool:
    """
    Append new data to a master file, avoiding duplicates.
    
    Args:
        df: DataFrame with new data
        master_file: Path to the master file
        
    Returns:
        True if successful, False otherwise
    """
    if df.empty:
        return True
    
    try:
        # Create or append to the master file
        if not os.path.exists(master_file):
            df.to_csv(master_file, index=False)
            logger.info(f"Created new master file: {master_file}")
            return True
        
        # Read existing data
        master_df = pd.read_csv(master_file)
        
        # Identify duplicates
        if 'order_release_id' in df.columns and 'source_file' in df.columns:
            # For each order_release_id and source_file combination, check if it already exists
            merged = pd.merge(
                df, master_df,
                on=['order_release_id', 'source_file'],
                how='left',
                indicator=True
            )
            
            # Keep only new records
            new_records = merged[merged['_merge'] == 'left_only']
            new_records = new_records.drop(columns=['_merge'])
            
            # Extract only the columns that should be in the new_records DataFrame
            cols_to_keep = [col for col in df.columns if col in new_records.columns]
            new_records = new_records[cols_to_keep]
            
            if new_records.empty:
                logger.info(f"No new records to add to {master_file}")
                return True
            
            # Append new records to the master file
            new_records.to_csv(master_file, mode='a', header=False, index=False)
            logger.info(f"Added {len(new_records)} new records to {master_file}")
        else:
            # If necessary columns are missing, just append all
            df.to_csv(master_file, mode='a', header=False, index=False)
            logger.info(f"Added {len(df)} records to {master_file} (without deduplication)")
        
        return True
    except Exception as e:
        logger.error(f"Error appending to master file {master_file}: {e}")
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
        df = process_orders_file(file_path)
        if not df.empty:
            append_to_master(df, ORDERS_MASTER)
    
    # Process returns files
    for file_path in files["returns"]:
        filename = os.path.basename(file_path)
        month, year = extract_date_from_filename(filename, RETURNS_PATTERN)
        file_id = get_file_identifier(month, year)
        
        if file_id in processed_returns:
            logger.info(f"Returns file already processed: {filename}")
            continue
        
        logger.info(f"Processing returns file: {filename}")
        df = process_returns_file(file_path)
        if not df.empty:
            append_to_master(df, RETURNS_MASTER)
    
    # Process settlement files
    for file_path in files["settlement"]:
        filename = os.path.basename(file_path)
        month, year = extract_date_from_filename(filename, SETTLEMENT_PATTERN)
        file_id = get_file_identifier(month, year)
        
        if file_id in processed_settlements:
            logger.info(f"Settlement file already processed: {filename}")
            continue
        
        logger.info(f"Processing settlement file: {filename}")
        df = process_settlement_file(file_path)
        if not df.empty:
            append_to_master(df, SETTLEMENT_MASTER)
    
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