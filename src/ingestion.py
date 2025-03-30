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

def process_orders_file(file_path: Path) -> None:
    """
    Process an orders file and update the orders master file.
    
    Args:
        file_path: Path to the orders file
    """
    # Read and validate the file
    orders_df = read_file(file_path)
    if not validate_file_columns(orders_df, 'orders'):
        raise ValueError(f"Invalid columns in orders file: {file_path}")
    
    # Standardize column names
    orders_df = orders_df.rename(columns={
        'order release id': 'order_release_id',
        'order line id': 'order_line_id',
        'order status': 'order_status',
        'final amount': 'final_amount',
        'total mrp': 'total_mrp',
        'coupon discount': 'coupon_discount',
        'shipping charge': 'shipping_charge',
        'gift charge': 'gift_charge',
        'tax recovery': 'tax_recovery'
    })
    
    # Convert numeric columns
    numeric_columns = [
        'final_amount', 'total_mrp', 'discount', 'coupon_discount',
        'shipping_charge', 'gift_charge', 'tax_recovery'
    ]
    for col in numeric_columns:
        if col in orders_df.columns:
            orders_df[col] = pd.to_numeric(orders_df[col], errors='coerce')
    
    # Convert date columns
    if 'order_date' in orders_df.columns:
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    
    # Update master file
    if ORDERS_MASTER.exists():
        master_df = read_file(ORDERS_MASTER)
        # Remove existing entries for these orders
        master_df = master_df[~master_df['order_release_id'].isin(orders_df['order_release_id'])]
        # Append new data
        master_df = pd.concat([master_df, orders_df], ignore_index=True)
    else:
        master_df = orders_df
    
    # Save updated master file
    master_df.to_csv(ORDERS_MASTER, index=False)

def process_returns_file(file_path: Path) -> None:
    """
    Process a returns file and update the returns master file.
    
    Args:
        file_path: Path to the returns file
    """
    # Read and validate the file
    returns_df = read_file(file_path)
    if not validate_file_columns(returns_df, 'returns'):
        raise ValueError(f"Invalid columns in returns file: {file_path}")
    
    # Convert numeric columns
    if 'return_amount' in returns_df.columns:
        returns_df['return_amount'] = pd.to_numeric(returns_df['return_amount'], errors='coerce')
    
    # Convert date columns
    if 'return_creation_date' in returns_df.columns:
        returns_df['return_creation_date'] = pd.to_datetime(returns_df['return_creation_date'])
    
    # Update master file
    if RETURNS_MASTER.exists():
        master_df = read_file(RETURNS_MASTER)
        # Remove existing entries for these returns
        master_df = master_df[~master_df['order_release_id'].isin(returns_df['order_release_id'])]
        # Append new data
        master_df = pd.concat([master_df, returns_df], ignore_index=True)
    else:
        master_df = returns_df
    
    # Save updated master file
    master_df.to_csv(RETURNS_MASTER, index=False)

def process_settlement_file(file_path: Path) -> None:
    """
    Process a settlement file and update the settlement master file.
    
    Args:
        file_path: Path to the settlement file
    """
    # Read and validate the file
    settlement_df = read_file(file_path)
    if not validate_file_columns(settlement_df, 'settlement'):
        raise ValueError(f"Invalid columns in settlement file: {file_path}")
    
    # Convert numeric columns
    numeric_columns = ['settlement_amount', 'total_actual_settlement']
    for col in numeric_columns:
        if col in settlement_df.columns:
            settlement_df[col] = pd.to_numeric(settlement_df[col], errors='coerce')
    
    # Convert date columns
    if 'settlement_date' in settlement_df.columns:
        settlement_df['settlement_date'] = pd.to_datetime(settlement_df['settlement_date'])
    
    # Update master file
    if SETTLEMENT_MASTER.exists():
        master_df = read_file(SETTLEMENT_MASTER)
        # Remove existing entries for these settlements
        master_df = master_df[~master_df['order_release_id'].isin(settlement_df['order_release_id'])]
        # Append new data
        master_df = pd.concat([master_df, settlement_df], ignore_index=True)
    else:
        master_df = settlement_df
    
    # Save updated master file
    master_df.to_csv(SETTLEMENT_MASTER, index=False)

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
        process_orders_file(file_path)
        return True
    elif file_type == 'returns':
        process_returns_file(file_path)
        return True
    elif file_type == 'settlement':
        process_settlement_file(file_path)
        return True
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