"""
Order analysis module for reconciliation application.
"""
import pandas as pd
import logging
import numpy as np
from typing import Dict, Tuple, List, Optional
from datetime import datetime

from utils import ensure_directories_exist, read_file, ANALYSIS_OUTPUT

logger = logging.getLogger(__name__)

def analyze_orders(
    orders_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    settlement_df: pd.DataFrame,
    previous_analysis_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Analyze orders and determine their status and financials.
    
    Args:
        orders_df: Orders DataFrame
        returns_df: Returns DataFrame
        settlement_df: Settlement DataFrame
        previous_analysis_df: Previous analysis results (if available)
    
    Returns:
        DataFrame with analysis results
    """
    if orders_df.empty:
        logger.warning("No orders data available for analysis")
        return pd.DataFrame()
    
    # Create a copy of orders DataFrame for analysis
    analysis_df = orders_df.copy()
    
    # Initialize new columns
    analysis_df['status'] = None
    analysis_df['profit_loss'] = None
    analysis_df['return_settlement'] = None
    analysis_df['order_settlement'] = None
    analysis_df['status_changed_this_run'] = False
    analysis_df['settlement_update_run_timestamp'] = None
    
    # Create previous status mapping if available
    previous_status_map = {}
    if previous_analysis_df is not None:
        previous_status_map = dict(zip(
            previous_analysis_df['order_release_id'],
            previous_analysis_df['status']
        ))
    
    # Analyze each order
    for idx, order in analysis_df.iterrows():
        previous_status = previous_status_map.get(order['order_release_id'])
        results = determine_order_status_and_financials(
            order, returns_df, settlement_df, previous_status
        )
        
        # Update analysis DataFrame
        for key, value in results.items():
            analysis_df.at[idx, key] = value
    
    return analysis_df

def determine_order_status_and_financials(
    order: pd.Series,
    returns_df: pd.DataFrame,
    settlement_df: pd.DataFrame,
    previous_status: Optional[str] = None
) -> Dict:
    """
    Determine order status and calculate financials based on original logic.
    
    Args:
        order: Order row from orders DataFrame
        returns_df: Returns DataFrame
        settlement_df: Settlement DataFrame
        previous_status: Status from previous analysis (if available)
    
    Returns:
        Dict containing status and financial calculations
    """
    order_id = order['order_release_id']
    
    # Initialize financial amounts
    return_settlement = 0.0  # Amount we owe back for returns
    order_settlement = 0.0   # Amount Myntra owes us
    profit_loss = 0.0
    
    # Check if the order was cancelled
    if 'is_ship_rel' in order and order['is_ship_rel'] == 0:
        status = "Cancelled"
        logger.debug(f"Order {order_id} marked as Cancelled (is_ship_rel=0)")
    else:
        # Check for returns by looking up order_id in returns DataFrame
        order_returns = returns_df[returns_df['order_release_id'] == order_id]
        has_returns = not order_returns.empty
        
        # Check for settlement
        order_settlement_data = settlement_df[settlement_df['order_release_id'] == order_id]
        has_settlement = not order_settlement_data.empty
        
        # Calculate settlements if available
        if has_returns and 'total_actual_settlement' in order_returns.columns:
            return_settlement = order_returns['total_actual_settlement'].sum()
            logger.debug(f"Order {order_id} has return settlement: {return_settlement}")
        
        if has_settlement and 'total_actual_settlement' in order_settlement_data.columns:
            order_settlement = order_settlement_data['total_actual_settlement'].sum()
            logger.debug(f"Order {order_id} has order settlement: {order_settlement}")
        
        # Determine status and calculate financials
        if has_returns:
            status = "Returned"
            profit_loss = order_settlement - return_settlement  # return_settlement is negative
            logger.debug(f"Order {order_id} marked as Returned, profit_loss: {profit_loss}")
        elif has_settlement:
            status = "Completed - Settled"
            profit_loss = order_settlement
            logger.debug(f"Order {order_id} marked as Completed - Settled, profit_loss: {profit_loss}")
        else:
            status = "Completed - Pending Settlement"
            profit_loss = 0.0
            logger.debug(f"Order {order_id} marked as Completed - Pending Settlement")
    
    # Track status changes
    status_changed = False
    if previous_status is not None and previous_status != status:
        status_changed = True
        logger.info(f"Order {order_id} status changed from {previous_status} to {status}")
    
    return {
        'status': status,
        'profit_loss': profit_loss,
        'return_settlement': return_settlement,
        'order_settlement': order_settlement,
        'status_changed_this_run': status_changed,
        'settlement_update_run_timestamp': datetime.now().isoformat() if status_changed else None
    }

def get_order_analysis_summary(analysis_df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics from order analysis.
    
    Args:
        analysis_df: Analysis results DataFrame
    
    Returns:
        Dict containing summary statistics
    """
    total_orders = len(analysis_df)
    total_profit_loss = analysis_df['profit_loss'].sum()
    
    # Calculate status counts
    status_counts = analysis_df['status'].value_counts()
    completed_orders = status_counts.get('Completed - Settled', 0)
    pending_orders = status_counts.get('Completed - Pending Settlement', 0)
    returned_orders = status_counts.get('Returned', 0)
    cancelled_orders = status_counts.get('Cancelled', 0)
    
    # Calculate rates
    settlement_rate = (completed_orders / total_orders * 100) if total_orders > 0 else 0
    return_rate = (returned_orders / total_orders * 100) if total_orders > 0 else 0
    
    # Calculate settlement amounts
    total_return_settlement = analysis_df['return_settlement'].sum()
    total_order_settlement = analysis_df['order_settlement'].sum()
    
    # Calculate pending settlement value
    pending_orders_df = analysis_df[analysis_df['status'] == 'Completed - Pending Settlement']
    pending_settlement_value = pending_orders_df['final_amount'].sum() if 'final_amount' in analysis_df.columns else 0
    
    # Count status changes
    status_changes = analysis_df['status_changed_this_run'].sum()
    
    # Calculate settlement changes
    settlement_changes = analysis_df[
        (analysis_df['status_changed_this_run']) &
        (analysis_df['status'] == 'Completed - Settled')
    ]
    
    pending_changes = analysis_df[
        (analysis_df['status_changed_this_run']) &
        (analysis_df['status'] == 'Completed - Pending Settlement')
    ]
    
    return {
        'total_orders': total_orders,
        'net_profit_loss': total_profit_loss,
        'settlement_rate': settlement_rate,
        'return_rate': return_rate,
        'total_return_settlement': total_return_settlement,
        'total_order_settlement': total_order_settlement,
        'pending_settlement_value': pending_settlement_value,
        'status_changes': status_changes,
        'settlement_changes': len(settlement_changes),
        'pending_changes': len(pending_changes),
        'status_counts': status_counts.to_dict()
    } 