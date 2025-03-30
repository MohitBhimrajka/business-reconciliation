"""
Order-level analysis module.
"""
import pandas as pd
import logging
import numpy as np
from typing import Dict, Tuple, List, Optional

from utils import ensure_directories_exist, read_file

logger = logging.getLogger(__name__)

def analyze_orders(
    orders_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    settlement_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze orders to determine their status and financial outcome.
    
    Args:
        orders_df: DataFrame containing order data
        returns_df: DataFrame containing return data
        settlement_df: DataFrame containing settlement data
        
    Returns:
        DataFrame with order analysis results
    """
    if orders_df.empty:
        logger.warning("No orders data available for analysis")
        return pd.DataFrame()
    
    # Initialize the analysis results DataFrame with more columns
    results = orders_df[['order_release_id']].copy()
    
    # Add default values
    results['status'] = 'Unknown'
    results['profit_loss'] = np.nan
    results['return_settlement'] = 0.0  # Amount we owe back for returns
    results['order_settlement'] = 0.0   # Amount Myntra owes us
    results['has_return_data'] = False
    results['has_settlement_data'] = False
    
    # Check for return data
    if not returns_df.empty:
        results['has_return_data'] = results['order_release_id'].isin(returns_df['order_release_id'])
    
    # Check for settlement data
    if not settlement_df.empty:
        results['has_settlement_data'] = results['order_release_id'].isin(settlement_df['order_release_id'])
    
    # Process each order
    for index, order in results.iterrows():
        order_id = order['order_release_id']
        
        # Get the full order record
        order_record = orders_df[orders_df['order_release_id'] == order_id].iloc[0]
        
        # Get return and settlement data for this order
        order_returns = returns_df[returns_df['order_release_id'] == order_id] if not returns_df.empty else pd.DataFrame()
        order_settlement = settlement_df[settlement_df['order_release_id'] == order_id] if not settlement_df.empty else pd.DataFrame()
        
        # Determine the order status and financial outcome
        status, profit_loss, return_settlement, order_settlement = determine_order_status_and_financials(
            order_record,
            order_returns,
            order_settlement
        )
        
        # Update the results
        results.at[index, 'status'] = status
        results.at[index, 'profit_loss'] = profit_loss
        results.at[index, 'return_settlement'] = return_settlement
        results.at[index, 'order_settlement'] = order_settlement
    
    # Clean up the results
    if 'has_return_data' in results.columns:
        results.drop(columns=['has_return_data'], inplace=True)
    if 'has_settlement_data' in results.columns:
        results.drop(columns=['has_settlement_data'], inplace=True)
    
    return results

def determine_order_status_and_financials(
    order: pd.Series,
    returns: pd.DataFrame,
    settlement: pd.DataFrame
) -> Tuple[str, float, float, float]:
    """
    Determine the status and financial outcome of an order.
    
    Args:
        order: Series containing order data
        returns: DataFrame containing return data for the order
        settlement: DataFrame containing settlement data for the order
        
    Returns:
        Tuple of (status, profit_loss, return_settlement, order_settlement)
    """
    order_id = order['order_release_id']
    
    # Check if the order was cancelled
    if 'is_ship_rel' in order and order['is_ship_rel'] == 0:
        return "Cancelled", 0.0, 0.0, 0.0
    
    # Initialize financial amounts
    return_settlement = 0.0  # Amount we owe back for returns
    order_settlement = 0.0   # Amount Myntra owes us
    profit_loss = 0.0
    
    # First check if the order appears in returns.csv
    if not returns.empty:
        # Get the return settlement amount (negative as it's what we owe back)
        return_settlement = returns['total_actual_settlement'].sum()
        if return_settlement < 0:  # If there's a return settlement
            # Get the settlement amount (positive as it's what Myntra owes us)
            if not settlement.empty and 'total_actual_settlement' in settlement.columns:
                order_settlement = settlement['total_actual_settlement'].sum()
            
            # Calculate the final profit/loss
            profit_loss = order_settlement + return_settlement  # return_settlement is already negative
            
            return "Returned", profit_loss, return_settlement, order_settlement
    
    # If not returned, check if it was shipped
    if 'is_ship_rel' in order and order['is_ship_rel'] == 1:
        # Get the settlement amount (positive as it's what Myntra owes us)
        if not settlement.empty and 'total_actual_settlement' in settlement.columns:
            order_settlement = settlement['total_actual_settlement'].sum()
        
        if order_settlement > 0:
            return "Completed - Settled", order_settlement, return_settlement, order_settlement
        else:
            return "Completed - Pending Settlement", 0.0, return_settlement, order_settlement
    
    # If the order was shipped but no return or settlement data
    return "Completed - Pending Settlement", 0.0, return_settlement, order_settlement

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