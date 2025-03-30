"""
Reporting and analytics module.
"""
import os
import logging
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
from pathlib import Path
from datetime import datetime

from utils import (
    ensure_directories_exist, read_file, ANALYSIS_OUTPUT, REPORT_OUTPUT,
    VISUALIZATION_DIR, ANOMALIES_OUTPUT
)

logger = logging.getLogger(__name__)

def save_analysis_results(analysis_df: pd.DataFrame) -> bool:
    """
    Save the order analysis results to a CSV file.
    
    Args:
        analysis_df: DataFrame containing order analysis results
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Save the analysis results
        analysis_df.to_csv(ANALYSIS_OUTPUT, index=False)
        logger.info(f"Saved analysis results to {ANALYSIS_OUTPUT}")
        return True
    except Exception as e:
        logger.error(f"Error saving analysis results: {e}")
        return False

def generate_report(summary: Dict) -> bool:
    """
    Generate a report with aggregate statistics.
    
    Args:
        summary: Dictionary with summary statistics
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Format the report
        report = [
            "=== Order Reconciliation Report ===",
            "",
            "=== Order Counts ===",
            f"Total Orders: {summary['total_orders']}",
            f"Cancelled Orders: {summary['cancelled_orders']} ({summary['cancelled_percentage']:.2f}%)",
            f"Returned Orders: {summary['returned_orders']} ({summary['returned_percentage']:.2f}%)",
            f"Completed and Settled Orders: {summary['completed_settled_orders']} ({summary['completed_settled_percentage']:.2f}%)",
            f"Completed but Pending Settlement Orders: {summary['completed_pending_settlement_orders']} ({summary['completed_pending_settlement_percentage']:.2f}%)",
            "",
            "=== Financial Analysis ===",
            f"Total Profit from Settled Orders: {summary['total_profit']:.2f}",
            f"Total Loss from Returned Orders: {summary['total_loss']:.2f}",
            f"Net Profit/Loss: {summary['net_profit_loss']:.2f}",
            "",
            "=== Key Metrics ===",
            f"Settlement Rate: {summary['settlement_rate']:.2f}%",
            f"Return Rate: {summary['return_rate']:.2f}%",
            f"Average Profit per Settled Order: {summary['avg_profit_per_settled_order']:.2f}",
            f"Average Loss per Returned Order: {summary['avg_loss_per_returned_order']:.2f}",
            "",
            "=== Recommendations ===",
            "1. Monitor orders with status 'Completed - Pending Settlement' to ensure they get settled.",
            "2. Analyze orders with high losses to identify patterns and potential issues.",
            "3. Investigate return patterns to reduce return rates.",
            "4. Consider strategies to increase settlement rates for shipped orders."
        ]
        
        # Write the report to a file
        with open(REPORT_OUTPUT, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Generated report at {REPORT_OUTPUT}")
        return True
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return False

def generate_visualizations(analysis_df: pd.DataFrame, orders_df: pd.DataFrame) -> None:
    """
    Generate visualizations for the analysis results.
    
    Args:
        analysis_df: DataFrame containing order analysis results
        orders_df: DataFrame containing order data for additional context
    """
    try:
        # Set the style
        sns.set(style="whitegrid")
        
        # 1. Order Status Distribution
        plt.figure(figsize=(10, 6))
        status_counts = analysis_df['status'].value_counts()
        ax = status_counts.plot(kind='bar', color='skyblue')
        plt.title('Order Status Distribution')
        plt.xlabel('Status')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add count labels on top of bars
        for i, count in enumerate(status_counts):
            ax.text(i, count + 0.1, str(count), ha='center')
        
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'order_status_distribution.png'))
        plt.close()
        
        # 2. Profit/Loss Distribution
        # Filter out NaN values and very large/small values for better visualization
        profit_loss_data = analysis_df[analysis_df['profit_loss'].notna()]
        profit_loss_data = profit_loss_data[
            (profit_loss_data['profit_loss'] > profit_loss_data['profit_loss'].quantile(0.05)) &
            (profit_loss_data['profit_loss'] < profit_loss_data['profit_loss'].quantile(0.95))
        ]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(profit_loss_data['profit_loss'], kde=True)
        plt.title('Profit/Loss Distribution')
        plt.xlabel('Profit/Loss')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'profit_loss_distribution.png'))
        plt.close()
        
        # 3. Monthly Analysis
        if 'source_file' in orders_df.columns:
            # Extract month and year from source_file
            orders_df['month_year'] = orders_df['source_file']
            
            # Merge with analysis results
            monthly_data = pd.merge(
                orders_df[['order_release_id', 'month_year']],
                analysis_df[['order_release_id', 'status', 'profit_loss']],
                on='order_release_id'
            )
            
            # Aggregate by month_year and status
            monthly_status = monthly_data.groupby(['month_year', 'status']).size().unstack(fill_value=0)
            
            # Plot monthly status distribution
            plt.figure(figsize=(12, 6))
            monthly_status.plot(kind='bar', stacked=True, colormap='viridis')
            plt.title('Monthly Order Status Distribution')
            plt.xlabel('Month-Year')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.legend(title='Status')
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATION_DIR, 'monthly_status_distribution.png'))
            plt.close()
            
            # Aggregate profit/loss by month
            monthly_profit_loss = monthly_data.groupby('month_year')['profit_loss'].agg(['sum', 'mean']).reset_index()
            monthly_profit_loss.columns = ['month_year', 'total_profit_loss', 'avg_profit_loss']
            
            # Plot monthly profit/loss
            plt.figure(figsize=(12, 6))
            ax = monthly_profit_loss.plot(x='month_year', y='total_profit_loss', kind='bar', color='green')
            plt.title('Monthly Total Profit/Loss')
            plt.xlabel('Month-Year')
            plt.ylabel('Total Profit/Loss')
            plt.xticks(rotation=45)
            
            # Add labels on bars
            for i, value in enumerate(monthly_profit_loss['total_profit_loss']):
                ax.text(i, value + (0.1 if value >= 0 else -0.1), f'{value:.0f}', ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATION_DIR, 'monthly_profit_loss.png'))
            plt.close()
        
        logger.info(f"Generated visualizations in {VISUALIZATION_DIR}")
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

def identify_anomalies(analysis_df: pd.DataFrame, orders_df: pd.DataFrame, returns_df: pd.DataFrame, settlement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify anomalies in the data, such as:
    - Orders returned but not in returns.csv
    - Orders shipped but not settled after a certain timeframe
    
    Args:
        analysis_df: DataFrame containing order analysis results
        orders_df: DataFrame containing order data
        returns_df: DataFrame containing return data
        settlement_df: DataFrame containing settlement data
        
    Returns:
        DataFrame with anomalies
    """
    anomalies = []
    
    # Check for returned orders not in returns.csv
    if 'return_creation_date' in orders_df.columns:
        returned_orders = orders_df[pd.notna(orders_df['return_creation_date']) & (orders_df['return_creation_date'] != '')]
        
        for _, order in returned_orders.iterrows():
            order_id = order['order_release_id']
            if returns_df.empty or order_id not in returns_df['order_release_id'].values:
                anomalies.append({
                    'order_release_id': order_id,
                    'anomaly_type': 'Returned order not in returns data',
                    'description': f"Order {order_id} has return_creation_date but is not found in returns data"
                })
    
    # Check for shipped orders not settled
    for _, order in orders_df[orders_df['is_ship_rel'] == 1].iterrows():
        order_id = order['order_release_id']
        # Skip returned orders for this check
        if 'return_creation_date' in order and pd.notna(order['return_creation_date']) and order['return_creation_date'] != '':
            continue
        
        if settlement_df.empty or order_id not in settlement_df['order_release_id'].values:
            # Add the source_file (month-year) if available
            month_year = order.get('source_file', 'Unknown')
            anomalies.append({
                'order_release_id': order_id,
                'anomaly_type': 'Shipped order not settled',
                'description': f"Order {order_id} from {month_year} was shipped but not settled"
            })
    
    # Create DataFrame from anomalies
    anomalies_df = pd.DataFrame(anomalies)
    
    # Save anomalies to a file
    if not anomalies_df.empty:
        anomalies_df.to_csv(ANOMALIES_OUTPUT, index=False)
        logger.info(f"Identified {len(anomalies_df)} anomalies, saved to {ANOMALIES_OUTPUT}")
    
    return anomalies_df 