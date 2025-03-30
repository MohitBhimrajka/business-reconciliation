"""
Reporting and visualization module for reconciliation application.
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
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    ensure_directories_exist, read_file, ANALYSIS_OUTPUT, REPORT_OUTPUT,
    VISUALIZATION_DIR, ANOMALIES_OUTPUT
)

logger = logging.getLogger(__name__)

def save_analysis_results(analysis_df: pd.DataFrame) -> None:
    """
    Save analysis results to CSV file.
    
    Args:
        analysis_df: Analysis results DataFrame
    """
    ensure_directories_exist()
    analysis_df.to_csv(ANALYSIS_OUTPUT, index=False)

def generate_report(summary: Dict) -> str:
    """
    Generate a text report from analysis summary.
    
    Args:
        summary: Analysis summary dictionary
    
    Returns:
        Formatted report string
    """
    report = [
        "Order Reconciliation Report",
        "=" * 50,
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Summary Statistics",
        "-" * 50,
        f"Total Orders: {summary['total_orders']:,}",
        f"Net Profit/Loss: ₹{summary['net_profit_loss']:,.2f}",
        f"Settlement Rate: {summary['settlement_rate']:.2f}%",
        f"Return Rate: {summary['return_rate']:.2f}%",
        "",
        "Status Distribution",
        "-" * 50
    ]
    
    for status, count in summary['status_counts'].items():
        report.append(f"{status}: {count:,}")
    
    report.extend([
        "",
        "Settlement Information",
        "-" * 50,
        f"Total Return Settlement: ₹{summary['total_return_settlement']:,.2f}",
        f"Total Order Settlement: ₹{summary['total_order_settlement']:,.2f}",
        f"Status Changes This Run: {summary['status_changes']:,}"
    ])
    
    return "\n".join(report)

def save_report(report: str) -> None:
    """
    Save report to text file.
    
    Args:
        report: Report text to save
    """
    ensure_directories_exist()
    with open(REPORT_OUTPUT, 'w') as f:
        f.write(report)

def generate_visualizations(analysis_df: pd.DataFrame, summary: Dict) -> None:
    """
    Generate interactive visualizations using Plotly.
    
    Args:
        analysis_df: Analysis results DataFrame
        summary: Analysis summary dictionary
    """
    ensure_directories_exist()
    
    # Order Status Distribution
    status_counts = analysis_df['status'].value_counts()
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Order Status Distribution"
    )
    fig.write_html(VISUALIZATION_DIR / "status_distribution.html")
    
    # Profit/Loss Distribution
    profit_loss_data = analysis_df[analysis_df['profit_loss'].notna()]
    fig = px.histogram(
        profit_loss_data,
        x='profit_loss',
        title="Profit/Loss Distribution",
        nbins=50
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.write_html(VISUALIZATION_DIR / "profit_loss_distribution.html")
    
    # Monthly Trends
    if 'source_file' in analysis_df.columns:
        analysis_df['month_year'] = analysis_df['source_file'].apply(
            lambda x: pd.to_datetime(x.split('-')[1:]).strftime('%Y-%m')
        )
        
        monthly_stats = analysis_df.groupby('month_year').agg({
            'order_release_id': 'count',
            'profit_loss': 'sum',
            'status': lambda x: (x == 'Completed - Settled').mean() * 100
        }).reset_index()
        
        monthly_stats.columns = ['Month', 'Total Orders', 'Net Profit/Loss', 'Settlement Rate']
        
        # Orders Trend
        fig = px.line(
            monthly_stats,
            x='Month',
            y='Total Orders',
            title="Monthly Orders Trend"
        )
        fig.write_html(VISUALIZATION_DIR / "monthly_orders_trend.html")
        
        # Profit/Loss Trend
        fig = px.line(
            monthly_stats,
            x='Month',
            y='Net Profit/Loss',
            title="Monthly Profit/Loss Trend"
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.write_html(VISUALIZATION_DIR / "monthly_profit_loss_trend.html")
        
        # Settlement Rate Trend
        fig = px.line(
            monthly_stats,
            x='Month',
            y='Settlement Rate',
            title="Monthly Settlement Rate Trend"
        )
        fig.write_html(VISUALIZATION_DIR / "monthly_settlement_rate_trend.html")

def identify_anomalies(
    analysis_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    settlement_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify anomalies in the data.
    
    Args:
        analysis_df: Analysis results DataFrame
        orders_df: Orders DataFrame
        returns_df: Returns DataFrame
        settlement_df: Settlement DataFrame
    
    Returns:
        DataFrame containing identified anomalies
    """
    anomalies = []
    
    # Check for orders with negative profit/loss
    negative_profit = analysis_df[analysis_df['profit_loss'] < 0]
    if not negative_profit.empty:
        anomalies.extend([
            {
                'type': 'Negative Profit',
                'order_release_id': row['order_release_id'],
                'details': f"Profit/Loss: ₹{row['profit_loss']:,.2f}"
            }
            for _, row in negative_profit.iterrows()
        ])
    
    # Check for orders with missing settlement data
    pending_settlement = analysis_df[
        analysis_df['status'] == 'Completed - Pending Settlement'
    ]
    if not pending_settlement.empty:
        anomalies.extend([
            {
                'type': 'Missing Settlement',
                'order_release_id': row['order_release_id'],
                'details': f"Order Amount: ₹{row['final_amount']:,.2f}"
            }
            for _, row in pending_settlement.iterrows()
        ])
    
    # Check for orders with both return and settlement data
    conflict_orders = analysis_df[
        (analysis_df['return_settlement'] > 0) &
        (analysis_df['order_settlement'] > 0)
    ]
    if not conflict_orders.empty:
        anomalies.extend([
            {
                'type': 'Return/Settlement Conflict',
                'order_release_id': row['order_release_id'],
                'details': f"Return: ₹{row['return_settlement']:,.2f}, Settlement: ₹{row['order_settlement']:,.2f}"
            }
            for _, row in conflict_orders.iterrows()
        ])
    
    # Create anomalies DataFrame
    anomalies_df = pd.DataFrame(anomalies)
    
    # Save anomalies to file
    if not anomalies_df.empty:
        anomalies_df.to_csv(ANOMALIES_OUTPUT, index=False)
    
    return anomalies_df 