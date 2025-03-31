"""
Reporting and visualization module for reconciliation application.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from utils import (
    ensure_directories_exist, read_file, ANALYSIS_OUTPUT, REPORT_OUTPUT,
    VISUALIZATION_DIR, ANOMALIES_OUTPUT, REPORT_DIR
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
    # Calculate percentages
    total_orders = summary['total_orders']
    cancelled_pct = (summary['status_counts'].get('Cancelled', 0) / total_orders * 100) if total_orders > 0 else 0
    returned_pct = (summary['status_counts'].get('Returned', 0) / total_orders * 100) if total_orders > 0 else 0
    settled_pct = (summary['status_counts'].get('Completed - Settled', 0) / total_orders * 100) if total_orders > 0 else 0
    pending_pct = (summary['status_counts'].get('Completed - Pending Settlement', 0) / total_orders * 100) if total_orders > 0 else 0
    
    # Calculate average values
    avg_profit = (summary['total_order_settlement'] / summary['status_counts'].get('Completed - Settled', 1)) if summary['status_counts'].get('Completed - Settled', 0) > 0 else 0
    avg_loss = (summary['total_return_settlement'] / summary['status_counts'].get('Returned', 1)) if summary['status_counts'].get('Returned', 0) > 0 else 0
    
    # Get settlement metrics
    settlement_metrics = summary.get('settlement_metrics', {})
    
    report = [
        "=== Order Reconciliation Report ===",
        "",
        "=== Order Counts ===",
        f"Total Orders: {total_orders}",
        f"Cancelled Orders: {summary['status_counts'].get('Cancelled', 0)} ({cancelled_pct:.2f}%)",
        f"Returned Orders: {summary['status_counts'].get('Returned', 0)} ({returned_pct:.2f}%)",
        f"Completed and Settled Orders: {summary['status_counts'].get('Completed - Settled', 0)} ({settled_pct:.2f}%)",
        f"Completed but Pending Settlement Orders: {summary['status_counts'].get('Completed - Pending Settlement', 0)} ({pending_pct:.2f}%)",
        "",
        "=== Financial Analysis ===",
        f"Total Profit from Settled Orders: {summary['total_order_settlement']:.2f}",
        f"Total Loss from Returned Orders: {abs(summary['total_return_settlement']):.2f}",
        f"Net Profit/Loss: {summary['net_profit_loss']:.2f}",
        "",
        "=== Settlement Information ===",
        f"Total Return Settlement Amount: ₹{abs(summary['total_return_settlement']):,.2f}",
        f"Total Order Settlement Amount: ₹{summary['total_order_settlement']:,.2f}",
        f"Potential Settlement Value (Pending): ₹{summary.get('pending_settlement_value', 0):,.2f}",
        f"Status Changes in This Run: {summary['status_changes']}",
        f"Orders Newly Settled in This Run: {summary['settlement_changes']}",
        f"Orders Newly Pending in This Run: {summary['pending_changes']}",
        "",
        "=== Key Metrics ===",
        f"Settlement Rate: {summary['settlement_rate']:.2f}%",
        f"Return Rate: {summary['return_rate']:.2f}%",
        f"Average Profit per Settled Order: {avg_profit:.2f}",
        f"Average Loss per Returned Order: {abs(avg_loss):.2f}",
    ]
    
    # Add settlement tracking information
    if settlement_metrics:
        report.extend([
            "",
            "=== Settlement Tracking ===",
            f"Current Month: {settlement_metrics.get('current_month', 'N/A')}",
            "",
            "New Pending Settlements:",
            f"Count: {len(settlement_metrics.get('new_pending_settlements', []))}",
            f"Total Amount: ₹{sum(s['settlement_amount'] for s in settlement_metrics.get('new_pending_settlements', [])):,.2f}",
            "",
            "Resolved Settlements:",
            f"Count: {len(settlement_metrics.get('newly_resolved_settlements', []))}",
            f"Total Amount: ₹{sum(s['settlement_amount'] for s in settlement_metrics.get('newly_resolved_settlements', [])):,.2f}",
            "",
            "Settlement Resolution Rates by Month:"
        ])
        
        for month, rate in settlement_metrics.get('settlement_resolution_rates', {}).items():
            report.append(f"{month}: {rate:.2f}%")
    
    report.extend([
        "",
        "=== Recommendations ===",
        "1. Monitor orders with status 'Completed - Pending Settlement' to ensure they get settled.",
        "2. Analyze orders with high losses to identify patterns and potential issues.",
        "3. Investigate return patterns to reduce return rates.",
        "4. Consider strategies to increase settlement rates for shipped orders."
    ])
    
    return "\n".join(report)

def save_report(report: str) -> None:
    """
    Save report to text file.
    
    Args:
        report: Report text to save
    """
    ensure_directories_exist()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = REPORT_DIR / f'reconciliation_report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Also save to the standard location
    with open(REPORT_OUTPUT, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_file}")
    logger.info(f"Report also saved to {REPORT_OUTPUT}")

def generate_visualizations(analysis_df: pd.DataFrame, summary: Dict) -> Dict[str, go.Figure]:
    """
    Generate visualizations for the analysis results.
    
    Args:
        analysis_df: Analysis results DataFrame
        summary: Analysis summary dictionary
    
    Returns:
        Dictionary of Plotly figures
    """
    figures = {}
    
    # Status Distribution
    status_counts = analysis_df['status'].value_counts()
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Order Status Distribution"
    )
    figures['status_distribution'] = fig
    
    # Profit/Loss Distribution
    fig = px.histogram(
        analysis_df,
        x='profit_loss',
        title="Profit/Loss Distribution",
        labels={'profit_loss': 'Profit/Loss Amount'}
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    figures['profit_loss_distribution'] = fig
    
    # Monthly Trends
    if 'created_on' in analysis_df.columns:
        analysis_df['month'] = pd.to_datetime(analysis_df['created_on']).dt.strftime('%Y-%m')
        
        # Orders by Month
        monthly_orders = analysis_df.groupby('month').size().reset_index(name='count')
        fig = px.line(
            monthly_orders,
            x='month',
            y='count',
            title="Monthly Orders Trend"
        )
        figures['monthly_orders_trend'] = fig
        
        # Profit/Loss by Month
        monthly_profit = analysis_df.groupby('month')['profit_loss'].sum().reset_index()
        fig = px.line(
            monthly_profit,
            x='month',
            y='profit_loss',
            title="Monthly Profit/Loss Trend"
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        figures['monthly_profit_loss_trend'] = fig
    
    # Settlement Tracking Visualizations
    settlement_metrics = summary.get('settlement_metrics', {})
    if settlement_metrics:
        # Settlement Resolution Rate Trend
        resolution_rates = pd.DataFrame([
            {'month': month, 'rate': rate}
            for month, rate in settlement_metrics.get('settlement_resolution_rates', {}).items()
        ])
        if not resolution_rates.empty:
            fig = px.line(
                resolution_rates,
                x='month',
                y='rate',
                title="Monthly Settlement Resolution Rate Trend"
            )
            figures['settlement_resolution_rate_trend'] = fig
        
        # Pending vs Resolved Settlements
        pending_by_month = defaultdict(int)
        resolved_by_month = defaultdict(int)
        
        for settlement in settlement_metrics.get('new_pending_settlements', []):
            pending_by_month[settlement['pending_month']] += 1
        
        for settlement in settlement_metrics.get('newly_resolved_settlements', []):
            resolved_by_month[settlement['resolution_month']] += 1
        
        comparison_df = pd.DataFrame([
            {'month': month, 'pending': count, 'resolved': resolved_by_month[month]}
            for month, count in pending_by_month.items()
        ])
        
        if not comparison_df.empty:
            fig = px.bar(
                comparison_df,
                x='month',
                y=['pending', 'resolved'],
                title="Pending vs Resolved Settlements by Month",
                barmode='group'
            )
            figures['settlement_comparison'] = fig
    
    return figures

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

def generate_pdf_report(summary: Dict, analysis_df: pd.DataFrame, report_type: str = 'reconciliation') -> str:
    """
    Generate a PDF report from analysis summary.
    
    Args:
        summary: Analysis summary dictionary
        analysis_df: Analysis results DataFrame
        report_type: Type of report to generate ('reconciliation', 'financial', 'data_quality')
    
    Returns:
        Path to the generated PDF file
    """
    ensure_directories_exist()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_file = REPORT_DIR / f'{report_type}_report_{timestamp}.pdf'
    
    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create custom styles
    styles.add(ParagraphStyle(
        name='Metric',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=30
    ))
    
    # Build the report content
    story = []
    
    # Title
    title = "Order Reconciliation Report" if report_type == 'reconciliation' else \
            "Financial Summary Report" if report_type == 'financial' else \
            "Data Quality Report"
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 12))
    
    # Add report type specific content
    if report_type == 'reconciliation':
        story.extend(generate_reconciliation_report_content(summary, styles))
    elif report_type == 'financial':
        story.extend(generate_financial_report_content(summary, analysis_df, styles))
    else:  # data_quality
        story.extend(generate_data_quality_report_content(summary, analysis_df, styles))
    
    # Build the PDF
    doc.build(story)
    return str(pdf_file)

def generate_reconciliation_report_content(summary: Dict, styles: Dict) -> List:
    """Generate content for reconciliation report."""
    content = []
    
    # Order Counts Section
    content.append(Paragraph("Order Counts", styles['Heading2']))
    content.append(Spacer(1, 12))
    
    # Create table for order counts
    order_data = [
        ["Total Orders", str(summary['total_orders'])],
        ["Cancelled Orders", f"{summary['status_counts'].get('Cancelled', 0)} ({(summary['status_counts'].get('Cancelled', 0) / summary['total_orders'] * 100):.2f}%)"],
        ["Returned Orders", f"{summary['status_counts'].get('Returned', 0)} ({(summary['status_counts'].get('Returned', 0) / summary['total_orders'] * 100):.2f}%)"],
        ["Completed and Settled", f"{summary['status_counts'].get('Completed - Settled', 0)} ({(summary['status_counts'].get('Completed - Settled', 0) / summary['total_orders'] * 100):.2f}%)"],
        ["Pending Settlement", f"{summary['status_counts'].get('Completed - Pending Settlement', 0)} ({(summary['status_counts'].get('Completed - Pending Settlement', 0) / summary['total_orders'] * 100):.2f}%)"]
    ]
    
    t = Table(order_data, colWidths=[2*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(t)
    
    # Financial Analysis Section
    content.append(Spacer(1, 20))
    content.append(Paragraph("Financial Analysis", styles['Heading2']))
    content.append(Spacer(1, 12))
    
    financial_data = [
        ["Total Profit from Settled Orders", f"₹{summary['total_order_settlement']:,.2f}"],
        ["Total Loss from Returned Orders", f"₹{abs(summary['total_return_settlement']):,.2f}"],
        ["Net Profit/Loss", f"₹{summary['net_profit_loss']:,.2f}"]
    ]
    
    t = Table(financial_data, colWidths=[2*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(t)
    
    return content

def generate_financial_report_content(summary: Dict, analysis_df: pd.DataFrame, styles: Dict) -> List:
    """Generate content for financial report."""
    content = []
    
    # Revenue Analysis
    content.append(Paragraph("Revenue Analysis", styles['Heading2']))
    content.append(Spacer(1, 12))
    
    revenue_data = [
        ["Total Revenue", f"₹{summary['total_order_settlement']:,.2f}"],
        ["Return Revenue", f"₹{abs(summary['total_return_settlement']):,.2f}"],
        ["Net Revenue", f"₹{summary['net_profit_loss']:,.2f}"]
    ]
    
    t = Table(revenue_data, colWidths=[2*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(t)
    
    # Cost Analysis
    content.append(Spacer(1, 20))
    content.append(Paragraph("Cost Analysis", styles['Heading2']))
    content.append(Spacer(1, 12))
    
    # Calculate costs from analysis_df
    commission_costs = analysis_df['commission'].sum()
    logistics_costs = analysis_df['logistics_cost'].sum()
    total_costs = commission_costs + logistics_costs
    
    cost_data = [
        ["Commission Costs", f"₹{commission_costs:,.2f}"],
        ["Logistics Costs", f"₹{logistics_costs:,.2f}"],
        ["Total Costs", f"₹{total_costs:,.2f}"]
    ]
    
    t = Table(cost_data, colWidths=[2*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(t)
    
    return content

def generate_data_quality_report_content(summary: Dict, analysis_df: pd.DataFrame, styles: Dict) -> List:
    """Generate content for data quality report."""
    content = []
    
    # Data Completeness
    content.append(Paragraph("Data Completeness", styles['Heading2']))
    content.append(Spacer(1, 12))
    
    # Calculate completeness for key fields
    total_rows = len(analysis_df)
    completeness_data = []
    
    for column in ['order_release_id', 'status', 'profit_loss', 'created_on']:
        non_null_count = analysis_df[column].notna().sum()
        completeness = (non_null_count / total_rows) * 100
        completeness_data.append([
            column.replace('_', ' ').title(),
            f"{completeness:.2f}%"
        ])
    
    t = Table(completeness_data, colWidths=[2*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(t)
    
    # Validation Errors
    content.append(Spacer(1, 20))
    content.append(Paragraph("Validation Errors", styles['Heading2']))
    content.append(Spacer(1, 12))
    
    # Get validation errors from summary
    validation_errors = summary.get('validation_errors', [])
    if validation_errors:
        error_data = [[error['field'], error['message']] for error in validation_errors]
        t = Table(error_data, colWidths=[2*inch, 3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(t)
    else:
        content.append(Paragraph("No validation errors found.", styles['Normal']))
    
    return content

def calculate_data_quality_metrics(
    orders_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    settlement_df: pd.DataFrame
) -> Dict:
    """
    Calculate data quality metrics for all master files.
    
    Args:
        orders_df: Orders DataFrame
        returns_df: Returns DataFrame
        settlement_df: Settlement DataFrame
    
    Returns:
        Dictionary containing data quality metrics
    """
    metrics = {
        'completeness': {},
        'orphaned_records': {},
        'validation_errors': []
    }
    
    # Calculate completeness for each DataFrame
    for df_name, df in [('orders', orders_df), ('returns', returns_df), ('settlements', settlement_df)]:
        total_rows = len(df)
        completeness = {}
        
        for column in df.columns:
            non_null_count = df[column].notna().sum()
            completeness[column] = (non_null_count / total_rows) * 100 if total_rows > 0 else 0
        
        metrics['completeness'][df_name] = completeness
    
    # Check for orphaned records
    # Returns without matching orders
    orphaned_returns = returns_df[~returns_df['order_release_id'].isin(orders_df['order_release_id'])]
    metrics['orphaned_records']['returns'] = len(orphaned_returns)
    
    # Settlements without matching orders
    orphaned_settlements = settlement_df[~settlement_df['order_release_id'].isin(orders_df['order_release_id'])]
    metrics['orphaned_records']['settlements'] = len(orphaned_settlements)
    
    # Validate data types and formats
    validation_errors = []
    
    # Check order_release_id format
    for df_name, df in [('orders', orders_df), ('returns', returns_df), ('settlements', settlement_df)]:
        invalid_ids = df[~df['order_release_id'].str.match(r'^[A-Z0-9]+$', na=False)]
        if not invalid_ids.empty:
            validation_errors.append({
                'file': df_name,
                'field': 'order_release_id',
                'message': f"Found {len(invalid_ids)} invalid order IDs"
            })
    
    # Check numeric fields
    for df_name, df in [('orders', orders_df), ('returns', returns_df), ('settlements', settlement_df)]:
        for field in ['profit_loss', 'commission', 'logistics_cost']:
            if field in df.columns:
                invalid_values = df[~df[field].apply(lambda x: isinstance(x, (int, float)) or pd.isna(x))]
                if not invalid_values.empty:
                    validation_errors.append({
                        'file': df_name,
                        'field': field,
                        'message': f"Found {len(invalid_values)} invalid numeric values"
                    })
    
    metrics['validation_errors'] = validation_errors
    
    return metrics 