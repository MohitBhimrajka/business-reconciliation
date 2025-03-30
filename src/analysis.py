"""
Order-level analysis module.
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
    Analyze orders to determine their status and financial outcome.
    
    Args:
        orders_df: DataFrame containing order data
        returns_df: DataFrame containing return data
        settlement_df: DataFrame containing settlement data
        previous_analysis_df: Previous analysis results (if available)
        
    Returns:
        DataFrame with order analysis results
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
    Determine order status and calculate financials.
    
    Args:
        order: Order row from orders DataFrame
        returns_df: Returns DataFrame
        settlement_df: Settlement DataFrame
        previous_status: Status from previous analysis (if available)
    
    Returns:
        Dict containing status and financial calculations
    """
    order_id = order['order_release_id']
    
    # Check for returns
    order_returns = returns_df[returns_df['order_release_id'] == order_id]
    has_returns = not order_returns.empty
    
    # Check for settlement
    order_settlement = settlement_df[settlement_df['order_release_id'] == order_id]
    has_settlement = not order_settlement.empty
    
    # Determine status
    if order['order_status'] == 'Cancelled':
        status = "Cancelled"
    elif has_returns:
        status = "Returned"
    elif has_settlement:
        status = "Completed - Settled"
    else:
        status = "Completed - Pending Settlement"
    
    # Calculate financials
    if status == "Cancelled":
        profit_loss = 0
        return_settlement = 0
        order_settlement = 0
    else:
        # Calculate profit/loss based on status
        if status == "Returned":
            profit_loss = -order['final_amount']
            return_settlement = order_returns['return_amount'].sum()
            order_settlement = 0
        elif status == "Completed - Settled":
            profit_loss = order['final_amount'] - order['total_mrp']
            return_settlement = 0
            order_settlement = order_settlement['settlement_amount'].sum()
        else:  # Pending Settlement
            profit_loss = order['final_amount'] - order['total_mrp']
            return_settlement = 0
            order_settlement = 0
    
    # Track status changes
    status_changed = False
    if previous_status is not None and previous_status != status:
        status_changed = True
    
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
    Generate a summary of the order analysis.
    
    Args:
        analysis_df: DataFrame containing order analysis results
        
    Returns:
        Dictionary with summary statistics
    """
    if analysis_df.empty:
        return {
            'total_orders': 0,
            'cancelled_orders': 0,
            'returned_orders': 0,
            'completed_settled_orders': 0,
            'completed_pending_settlement_orders': 0,
            'cancelled_percentage': 0,
            'returned_percentage': 0,
            'completed_settled_percentage': 0,
            'completed_pending_settlement_percentage': 0,
            'total_profit': 0,
            'total_loss': 0,
            'net_profit_loss': 0,
            'settlement_rate': 0,
            'return_rate': 0,
            'avg_profit_per_settled_order': 0,
            'avg_loss_per_returned_order': 0
        }
    
    # Calculate counts
    total_orders = len(analysis_df)
    cancelled_orders = len(analysis_df[analysis_df['status'] == 'Cancelled'])
    returned_orders = len(analysis_df[analysis_df['status'] == 'Returned'])
    completed_settled_orders = len(analysis_df[analysis_df['status'] == 'Completed - Settled'])
    completed_pending_settlement_orders = len(analysis_df[analysis_df['status'] == 'Completed - Pending Settlement'])
    
    # Calculate percentages
    cancelled_percentage = (cancelled_orders / total_orders) * 100 if total_orders > 0 else 0
    returned_percentage = (returned_orders / total_orders) * 100 if total_orders > 0 else 0
    completed_settled_percentage = (completed_settled_orders / total_orders) * 100 if total_orders > 0 else 0
    completed_pending_settlement_percentage = (completed_pending_settlement_orders / total_orders) * 100 if total_orders > 0 else 0
    
    # Calculate financial metrics
    total_profit = analysis_df[analysis_df['profit_loss'] > 0]['profit_loss'].sum()
    total_loss = abs(analysis_df[analysis_df['profit_loss'] < 0]['profit_loss'].sum())
    net_profit_loss = total_profit - total_loss
    
    # Calculate settlement and return rates
    shipped_orders = completed_settled_orders + completed_pending_settlement_orders + returned_orders
    settlement_rate = (completed_settled_orders / shipped_orders) * 100 if shipped_orders > 0 else 0
    return_rate = (returned_orders / shipped_orders) * 100 if shipped_orders > 0 else 0
    
    # Calculate averages
    avg_profit_per_settled_order = total_profit / completed_settled_orders if completed_settled_orders > 0 else 0
    avg_loss_per_returned_order = total_loss / returned_orders if returned_orders > 0 else 0
    
    return {
        'total_orders': total_orders,
        'cancelled_orders': cancelled_orders,
        'returned_orders': returned_orders,
        'completed_settled_orders': completed_settled_orders,
        'completed_pending_settlement_orders': completed_pending_settlement_orders,
        'cancelled_percentage': cancelled_percentage,
        'returned_percentage': returned_percentage,
        'completed_settled_percentage': completed_settled_percentage,
        'completed_pending_settlement_percentage': completed_pending_settlement_percentage,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'net_profit_loss': net_profit_loss,
        'settlement_rate': settlement_rate,
        'return_rate': return_rate,
        'avg_profit_per_settled_order': avg_profit_per_settled_order,
        'avg_loss_per_returned_order': avg_loss_per_returned_order
    } 