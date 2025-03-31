"""
Streamlit application for Order Reconciliation.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Any
import functools
import logging
from cachetools import TTLCache, cached, keys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gc
import json

from utils import (
    ensure_directories_exist, read_file, ANALYSIS_OUTPUT, REPORT_OUTPUT,
    VISUALIZATION_DIR, ANOMALIES_OUTPUT, ORDERS_MASTER, RETURNS_MASTER,
    SETTLEMENT_MASTER, ORDERS_PATTERN, RETURNS_PATTERN, SETTLEMENT_PATTERN,
    validate_and_convert_dataframe, get_file_identifier, format_currency,
    format_percentage, DATA_DIR
)
from ingestion import process_orders_file, process_returns_file, process_settlement_file
from analysis import analyze_orders, get_order_analysis_summary
from reporting import (
    generate_pdf_report,
    calculate_data_quality_metrics,
    generate_visualizations,
    generate_report,
    identify_anomalies
)
from schemas import ORDERS_SCHEMA, RETURNS_SCHEMA, SETTLEMENT_SCHEMA
from validation import ValidationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure the page
st.set_page_config(
    page_title="Order Reconciliation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'analysis_df' not in st.session_state:
    st.session_state.analysis_df = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'anomalies_df' not in st.session_state:
    st.session_state.anomalies_df = None
if 'newly_added_files' not in st.session_state:
    st.session_state.newly_added_files = []
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = None
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None
if 'orders_df' not in st.session_state:
    st.session_state.orders_df = None
if 'returns_df' not in st.session_state:
    st.session_state.returns_df = None
if 'settlement_df' not in st.session_state:
    st.session_state.settlement_df = None

# Initialize caches
master_file_cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour cache
analysis_cache = TTLCache(maxsize=100, ttl=1800)  # 30 minutes cache
visualization_cache = TTLCache(maxsize=100, ttl=1800)  # 30 minutes cache

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric columns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Optimized DataFrame
    """
    try:
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        
        return df
    except Exception as e:
        logger.error(f"Error optimizing DataFrame: {str(e)}")
        return df

def process_dataframe_batch(df: pd.DataFrame, batch_size: int = 10000) -> pd.DataFrame:
    """
    Process DataFrame in batches to manage memory usage.
    
    Args:
        df: Input DataFrame
        batch_size: Size of each batch
    
    Returns:
        Processed DataFrame
    """
    try:
        result_dfs = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size].copy()
            batch = optimize_dataframe(batch)
            result_dfs.append(batch)
            gc.collect()  # Force garbage collection
        
        return pd.concat(result_dfs, ignore_index=True)
    except Exception as e:
        logger.error(f"Error processing DataFrame batch: {str(e)}")
        return df

def parallel_process_dataframes(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Process multiple DataFrames in parallel.
    
    Args:
        dfs: List of DataFrames to process
    
    Returns:
        List of processed DataFrames
    """
    try:
        with ThreadPoolExecutor() as executor:
            return list(executor.map(process_dataframe_batch, dfs))
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        return dfs

def clear_caches():
    """Clear all caches when new data is uploaded."""
    try:
        master_file_cache.clear()
        analysis_cache.clear()
        visualization_cache.clear()
        gc.collect()  # Force garbage collection
        logger.info("All caches cleared")
    except Exception as e:
        logger.error(f"Error clearing caches: {str(e)}")

@cached(cache=master_file_cache)
def load_master_file(file_path: Path) -> pd.DataFrame:
    """Load master file with caching."""
    try:
        df = read_file(file_path)
        return optimize_dataframe(df)
    except Exception as e:
        logger.error(f"Error loading master file {file_path}: {str(e)}")
        raise

@cached(cache=analysis_cache)
def analyze_orders_cached_v3(orders_mtime, returns_mtime, settlement_mtime) -> tuple:
    """Cached version of analyze_orders using file modification times as cache keys."""
    logger.info("Running cached analysis function analyze_orders_cached_v3")
    
    # Load data inside the cached function
    orders_df = load_master_file(ORDERS_MASTER)
    returns_df = load_master_file(RETURNS_MASTER)
    settlement_df = load_master_file(SETTLEMENT_MASTER)

    # Store loaded DFs in session state
    st.session_state.orders_df = orders_df
    st.session_state.returns_df = returns_df
    st.session_state.settlement_df = settlement_df

    # Load previous analysis if available
    previous_analysis_df = None
    if ANALYSIS_OUTPUT.exists():
        previous_analysis_df = load_master_file(ANALYSIS_OUTPUT)

    # Run analysis
    logger.info("Calling analyze_orders from cached function")
    analysis_df, summary = analyze_orders(orders_df, returns_df, settlement_df, previous_analysis_df)

    # Optimize result DataFrame
    logger.info("Optimizing analysis_df in cached function")
    analysis_df = optimize_dataframe(analysis_df)

    return analysis_df, summary

def process_uploaded_files():
    """Process uploaded files and update master files."""
    try:
        # Process each file type
        for file_type, file_path in st.session_state.uploaded_files.items():
            if file_path:
                logger.info(f"Processing {file_type} file: {file_path}")
                process_file(file_path, file_type)
        
        # Ensure master files exist before getting mtime
        if not ORDERS_MASTER.exists() or not RETURNS_MASTER.exists() or not SETTLEMENT_MASTER.exists():
            st.error("Master files are missing after processing. Cannot proceed with analysis.")
            return

        # Get modification times
        orders_mtime = os.path.getmtime(ORDERS_MASTER)
        returns_mtime = os.path.getmtime(RETURNS_MASTER)
        settlement_mtime = os.path.getmtime(SETTLEMENT_MASTER)

        # Call the cached function with timestamps
        logger.info("Calling analyze_orders_cached_v3")
        st.session_state.analysis_df, st.session_state.summary = analyze_orders_cached_v3(
            orders_mtime, returns_mtime, settlement_mtime
        )
        logger.info("Analysis complete. Analysis DF rows: %d", len(st.session_state.analysis_df))

        # Generate visualizations
        logger.info("Generating visualizations")
        st.session_state.visualizations = generate_visualizations(
            st.session_state.analysis_df, st.session_state.summary
        )

        # Identify anomalies
        logger.info("Identifying anomalies")
        st.session_state.anomalies_df = identify_anomalies(
            st.session_state.analysis_df,
            st.session_state.orders_df,
            st.session_state.returns_df,
            st.session_state.settlement_df
        )

        # Clear newly added files list
        st.session_state.newly_added_files = []

        st.success("Files processed and analyzed successfully!")

    except Exception as e:
        logger.error(f"Error during analysis or post-processing: {str(e)}")
        st.error(f"Error during analysis: {str(e)}")
        st.exception(e)

def display_schema_info():
    """Display information about required columns and data types."""
    st.subheader("Data Schema Information")
    
    # Create tabs for each file type
    tab1, tab2, tab3 = st.tabs(["Orders", "Returns", "Settlement"])
    
    with tab1:
        st.write("### Orders Schema")
        orders_schema_df = pd.DataFrame([
            {"Column": col, "Type": info["type"], "Required": info["required"]}
            for col, info in ORDERS_SCHEMA.items()
        ])
        st.dataframe(orders_schema_df, use_container_width=True)
    
    with tab2:
        st.write("### Returns Schema")
        returns_schema_df = pd.DataFrame([
            {"Column": col, "Type": info["type"], "Required": info["required"]}
            for col, info in RETURNS_SCHEMA.items()
        ])
        st.dataframe(returns_schema_df, use_container_width=True)
    
    with tab3:
        st.write("### Settlement Schema")
        settlement_schema_df = pd.DataFrame([
            {"Column": col, "Type": info["type"], "Required": info["required"]}
            for col, info in SETTLEMENT_SCHEMA.items()
        ])
        st.dataframe(settlement_schema_df, use_container_width=True)

def display_validation_results(results: Dict[str, List[ValidationResult]]):
    """Display validation results in a user-friendly format."""
    try:
        st.subheader("Validation Results")
        
        for file_type, file_results in results.items():
            if not file_results:
                continue
            
            st.write(f"### {file_type.title()} Files")
            
            # Aggregate statistics
            total_rows = sum(r.stats['total_rows'] for r in file_results)
            valid_rows = sum(r.stats['valid_rows'] for r in file_results)
            invalid_rows = sum(r.stats['invalid_rows'] for r in file_results)
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", total_rows)
            with col2:
                st.metric("Valid Rows", valid_rows)
            with col3:
                st.metric("Invalid Rows", invalid_rows)
            
            # Display errors if any
            all_errors = []
            for result in file_results:
                all_errors.extend(result.errors)
            
            if all_errors:
                st.error(f"Found {len(all_errors)} validation errors")
                
                # Group errors by type
                error_types = {}
                for error in all_errors:
                    error_type = error.get('type', 'Unknown')
                    if error_type not in error_types:
                        error_types[error_type] = []
                    error_types[error_type].append(error)
                
                # Display errors by type
                for error_type, errors in error_types.items():
                    with st.expander(f"{error_type} ({len(errors)} errors)"):
                        errors_df = pd.DataFrame(errors)
                        st.dataframe(
                            errors_df,
                            column_config={
                                "row_index": st.column_config.NumberColumn("Row"),
                                "column": st.column_config.TextColumn("Column"),
                                "message": st.column_config.TextColumn("Error Message"),
                                "value": st.column_config.TextColumn("Invalid Value")
                            },
                            use_container_width=True
                        )
            
            # Display warnings if any
            all_warnings = []
            for result in file_results:
                all_warnings.extend(result.warnings)
            
            if all_warnings:
                st.warning(f"Found {len(all_warnings)} validation warnings")
                
                # Group warnings by type
                warning_types = {}
                for warning in all_warnings:
                    warning_type = warning.get('type', 'Unknown')
                    if warning_type not in warning_types:
                        warning_types[warning_type] = []
                    warning_types[warning_type].append(warning)
                
                # Display warnings by type
                for warning_type, warnings in warning_types.items():
                    with st.expander(f"{warning_type} ({len(warnings)} warnings)"):
                        warnings_df = pd.DataFrame(warnings)
                        st.dataframe(
                            warnings_df,
                            column_config={
                                "row_index": st.column_config.NumberColumn("Row"),
                                "column": st.column_config.TextColumn("Column"),
                                "message": st.column_config.TextColumn("Warning Message"),
                                "value": st.column_config.TextColumn("Value")
                            },
                            use_container_width=True
                        )
            
            # Display data quality metrics
            if any(r.stats for r in file_results):
                st.write("#### Data Quality Metrics")
                
                # Calculate completeness
                completeness = {}
                for result in file_results:
                    if 'completeness' in result.stats:
                        completeness.update(result.stats['completeness'])
                
                if completeness:
                    completeness_df = pd.DataFrame([
                        {"Column": col, "Completeness (%)": round(comp, 2)}
                        for col, comp in completeness.items()
                    ])
                    st.dataframe(
                        completeness_df,
                        column_config={
                            "Column": st.column_config.TextColumn("Column"),
                            "Completeness (%)": st.column_config.NumberColumn(
                                "Completeness (%)",
                                format="%.2f%%"
                            )
                        },
                        use_container_width=True
                    )
                
                # Display orphaned records
                orphaned_records = {}
                for result in file_results:
                    if 'orphaned_records' in result.stats:
                        orphaned_records.update(result.stats['orphaned_records'])
                
                if orphaned_records:
                    st.warning("Found orphaned records:")
                    for record_type, count in orphaned_records.items():
                        st.write(f"- {record_type}: {count} records")
            
            # Add download button for validation report
            if all_errors or all_warnings:
                try:
                    report_data = {
                        'errors': all_errors,
                        'warnings': all_warnings,
                        'stats': {
                            'total_rows': total_rows,
                            'valid_rows': valid_rows,
                            'invalid_rows': invalid_rows,
                            'completeness': completeness,
                            'orphaned_records': orphaned_records
                        }
                    }
                    
                    report_json = json.dumps(report_data, indent=2)
                    st.download_button(
                        label="Download Validation Report",
                        data=report_json,
                        file_name=f"{file_type}_validation_report.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Failed to generate validation report: {str(e)}")
                    logger.error(f"Error generating validation report: {str(e)}")
    except Exception as e:
        st.error(f"Error displaying validation results: {str(e)}")
        logger.error(f"Error in display_validation_results: {str(e)}")

def display_dashboard():
    """Display the main dashboard with metrics and charts."""
    if st.session_state.analysis_df is None:
        st.warning("No analysis data available. Please upload and process files first.")
        return
    
    # Add a date range selector at the top
    if 'created_on' in st.session_state.analysis_df.columns:
        min_date = pd.to_datetime(st.session_state.analysis_df['created_on']).min()
        max_date = pd.to_datetime(st.session_state.analysis_df['created_on']).max()
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data based on selected date range
        filtered_df = st.session_state.analysis_df[
            (pd.to_datetime(st.session_state.analysis_df['created_on']).dt.date >= date_range[0]) &
            (pd.to_datetime(st.session_state.analysis_df['created_on']).dt.date <= date_range[1])
        ]
    else:
        filtered_df = st.session_state.analysis_df
    
    # Display key metrics with enhanced styling and tooltips
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_orders = len(filtered_df)
        st.metric(
            "Total Orders",
            total_orders,
            f"{total_orders:,}",
            help="Total number of orders in the selected period"
        )
    with col2:
        net_profit_loss = filtered_df['profit_loss'].sum()
        profit_color = "green" if net_profit_loss >= 0 else "red"
        st.metric(
            "Net Profit/Loss",
            f"₹{net_profit_loss:,.2f}",
            delta_color=profit_color,
            help="Total profit or loss in the selected period"
        )
    with col3:
        settlement_rate = (len(filtered_df[filtered_df['status'].str.contains('Settled', na=False)]) / total_orders * 100) if total_orders > 0 else 0
        st.metric(
            "Settlement Rate",
            f"{settlement_rate:.2f}%",
            f"{settlement_rate:.1f}%",
            help="Percentage of orders that have been settled"
        )
    with col4:
        return_rate = (len(filtered_df[filtered_df['status'].str.contains('Returned', na=False)]) / total_orders * 100) if total_orders > 0 else 0
        st.metric(
            "Return Rate",
            f"{return_rate:.2f}%",
            f"{return_rate:.1f}%",
            help="Percentage of orders that have been returned"
        )
    
    # Settlement Tracking Dashboard with enhanced visuals
    settlement_metrics = st.session_state.summary.get('settlement_metrics', {})
    if settlement_metrics:
        st.subheader("Settlement Tracking")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_pending = settlement_metrics.get('new_pending_settlements', [])
            new_pending_count = len(new_pending)
            new_pending_amount = sum(s['settlement_amount'] for s in new_pending)
            st.metric(
                "New Pending Settlements",
                new_pending_count,
                f"₹{new_pending_amount:,.2f}",
                help="Number and value of new pending settlements"
            )
        with col2:
            resolved = settlement_metrics.get('newly_resolved_settlements', [])
            resolved_count = len(resolved)
            resolved_amount = sum(s['settlement_amount'] for s in resolved)
            st.metric(
                "Resolved Settlements",
                resolved_count,
                f"₹{resolved_amount:,.2f}",
                help="Number and value of newly resolved settlements"
            )
        with col3:
            current_month = settlement_metrics.get('current_month', 'N/A')
            resolution_rate = settlement_metrics.get('settlement_resolution_rates', {}).get(current_month, 0)
            st.metric(
                "Current Month Resolution Rate",
                f"{resolution_rate:.2f}%",
                f"{resolution_rate:.1f}%",
                help="Percentage of settlements resolved in the current month"
            )
    
    # Main Analytics Section with interactive charts
    st.subheader("Order Analytics")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Order Status Distribution with enhanced styling
        status_counts = filtered_df['status'].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Order Status Distribution",
            hole=0.4,  # Create a donut chart
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_status.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_status, use_container_width=True)
        
        # Monthly Orders Trend with enhanced styling
        if 'monthly_orders_trend' in st.session_state.visualizations:
            fig_monthly = st.session_state.visualizations['monthly_orders_trend']
            fig_monthly.update_layout(
                title="Monthly Orders Trend",
                xaxis_title="Month",
                yaxis_title="Number of Orders",
                hovermode='x unified'
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        # Profit/Loss Distribution with enhanced styling
        fig_profit = px.histogram(
            filtered_df,
            x='profit_loss',
            title="Profit/Loss Distribution",
            labels={'profit_loss': 'Profit/Loss Amount'},
            nbins=50,
            color_discrete_sequence=['#2ecc71']
        )
        fig_profit.add_vline(x=0, line_dash="dash", line_color="red")
        fig_profit.update_layout(
            xaxis_title="Profit/Loss Amount",
            yaxis_title="Count",
            hovermode='x unified'
        )
        st.plotly_chart(fig_profit, use_container_width=True)
        
        # Monthly Profit/Loss Trend with enhanced styling
        if 'monthly_profit_loss_trend' in st.session_state.visualizations:
            fig_profit_trend = st.session_state.visualizations['monthly_profit_loss_trend']
            fig_profit_trend.update_layout(
                title="Monthly Profit/Loss Trend",
                xaxis_title="Month",
                yaxis_title="Profit/Loss Amount",
                hovermode='x unified'
            )
            fig_profit_trend.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_profit_trend, use_container_width=True)
    
    # Settlement Analytics Section with enhanced visuals
    if settlement_metrics:
        st.subheader("Settlement Analytics")
        
        # Create two columns for settlement charts
        col1, col2 = st.columns(2)
            
        with col1:
            # Settlement Resolution Rate Trend with enhanced styling
            if 'settlement_resolution_rate_trend' in st.session_state.visualizations:
                fig_resolution = st.session_state.visualizations['settlement_resolution_rate_trend']
                fig_resolution.update_layout(
                    title="Settlement Resolution Rate Trend",
                    xaxis_title="Month",
                    yaxis_title="Resolution Rate (%)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_resolution, use_container_width=True)
        
        with col2:
            # Settlement Comparison with enhanced styling
            if 'settlement_comparison' in st.session_state.visualizations:
                fig_comparison = st.session_state.visualizations['settlement_comparison']
                fig_comparison.update_layout(
                    title="Settlement Comparison",
                    xaxis_title="Month",
                    yaxis_title="Amount",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Anomalies Section with enhanced visuals
    if st.session_state.anomalies_df is not None and not st.session_state.anomalies_df.empty:
        st.subheader("Anomalies")
        
        # Display anomaly metrics with tooltips
        col1, col2, col3 = st.columns(3)
        
        with col1:
            anomaly_count = len(st.session_state.anomalies_df)
            st.metric(
                "Total Anomalies",
                anomaly_count,
                help="Total number of anomalies detected"
            )
            with col2:
                anomaly_value = st.session_state.anomalies_df['profit_loss'].sum()
                st.metric(
                    "Anomaly Value",
                    f"₹{anomaly_value:,.2f}",
                    help="Total value of anomalies detected"
                )
        with col3:
            anomaly_rate = (anomaly_count / len(filtered_df)) * 100
            st.metric(
                "Anomaly Rate",
                f"{anomaly_rate:.2f}%",
                help="Percentage of orders with anomalies"
            )
        
        # Display anomaly details with enhanced styling
        st.dataframe(
            st.session_state.anomalies_df,
            column_config={
                "order_release_id": st.column_config.TextColumn("Order ID"),
                "status": st.column_config.TextColumn("Status"),
                "profit_loss": st.column_config.NumberColumn("Profit/Loss", format="₹%.2f"),
                "anomaly_type": st.column_config.TextColumn("Anomaly Type")
            },
            use_container_width=True,
            hide_index=True
            )

def display_detailed_analysis():
    """Display detailed order analysis in an interactive table."""
    if st.session_state.analysis_df is None:
        st.warning("No analysis data available. Please upload and process files first.")
        return
    
    st.dataframe(
        st.session_state.analysis_df,
        column_config={
            "order_release_id": st.column_config.TextColumn("Order ID"),
            "status": st.column_config.TextColumn("Status"),
            "profit_loss": st.column_config.NumberColumn("Profit/Loss", format="₹%.2f"),
            "return_settlement": st.column_config.NumberColumn("Return Settlement", format="₹%.2f"),
            "order_settlement": st.column_config.NumberColumn("Order Settlement", format="₹%.2f"),
            "status_changed_this_run": st.column_config.CheckboxColumn("Status Changed"),
            "settlement_update_run_timestamp": st.column_config.DatetimeColumn("Last Update")
        },
        use_container_width=True
    )

def display_master_data():
    """Display master data files in interactive tables."""
    col1, col2, col3 = st.tabs(["Orders", "Returns", "Settlement"])
    
    with col1:
        if os.path.exists(ORDERS_MASTER):
            orders_df = read_file(ORDERS_MASTER)
            st.dataframe(
                orders_df,
                column_config={
                    "order_release_id": st.column_config.TextColumn("Order ID"),
                    "order_status": st.column_config.TextColumn("Status"),
                    "final_amount": st.column_config.NumberColumn("Final Amount", format="₹%.2f"),
                    "total_mrp": st.column_config.NumberColumn("Total MRP", format="₹%.2f")
                },
                use_container_width=True
            )
        else:
            st.warning("No orders master data available.")
    
    with col2:
        if os.path.exists(RETURNS_MASTER):
            returns_df = read_file(RETURNS_MASTER)
            st.dataframe(
                returns_df,
                column_config={
                    "order_release_id": st.column_config.TextColumn("Order ID"),
                    "return_amount": st.column_config.NumberColumn("Return Amount", format="₹%.2f")
                },
                use_container_width=True
            )
        else:
            st.warning("No returns master data available.")
    
    with col3:
        if os.path.exists(SETTLEMENT_MASTER):
            settlement_df = read_file(SETTLEMENT_MASTER)
            st.dataframe(
                settlement_df,
                column_config={
                    "order_release_id": st.column_config.TextColumn("Order ID"),
                    "settlement_amount": st.column_config.NumberColumn("Settlement Amount", format="₹%.2f")
                },
                use_container_width=True
            )
        else:
            st.warning("No settlement master data available.")

def display_settlements_management():
    """Display enhanced settlement management interface."""
    if st.session_state.analysis_df is None:
        st.warning("No analysis data available. Please upload and process files first.")
        return
    
    settlement_metrics = st.session_state.summary.get('settlement_metrics', {})
    settlement_trends = st.session_state.summary.get('settlement_trends', {})
    settlement_report = st.session_state.summary.get('settlement_report', {})
    
    if not settlement_metrics:
        st.warning("No settlement tracking data available.")
        return
    
    # Settlement Tracking Dashboard
    st.subheader("Settlement Tracking Dashboard")
    
    # Current Month Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "New Pending Settlements",
            len(settlement_metrics.get('new_pending_settlements', [])),
            f"₹{sum(s['settlement_amount'] for s in settlement_metrics.get('new_pending_settlements', [])):,.2f}"
        )
    with col2:
        st.metric(
            "Resolved Settlements",
            len(settlement_metrics.get('newly_resolved_settlements', [])),
            f"₹{sum(s['settlement_amount'] for s in settlement_metrics.get('newly_resolved_settlements', [])):,.2f}"
        )
    with col3:
        current_month = settlement_metrics.get('current_month', 'N/A')
        resolution_rate = settlement_metrics.get('settlement_resolution_rates', {}).get(current_month, 0)
        st.metric(
            "Current Month Resolution Rate",
            f"{resolution_rate:.2f}%"
        )
    
    # Settlement Analytics
    st.subheader("Settlement Analytics")
    
    # Settlement Status Distribution
    status_counts = st.session_state.analysis_df['status'].value_counts()
    fig_status = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Settlement Status Distribution"
    )
    st.plotly_chart(fig_status, use_container_width=True)
    
    # Brand-wise Settlement Analysis
    if 'brand_wise_metrics' in settlement_metrics:
        brand_data = []
        for brand, metrics in settlement_metrics['brand_wise_metrics'].items():
            brand_data.append({
                'brand': brand,
                'settlement_rate': (
                    metrics['settled_orders'] / metrics['total_orders'] * 100
                ) if metrics['total_orders'] > 0 else 0,
                'pending_value_ratio': (
                    metrics['pending_settlement_value'] / metrics['total_settlement_value'] * 100
                ) if metrics['total_settlement_value'] > 0 else 0
            })
        
        brand_df = pd.DataFrame(brand_data)
        fig_brand = px.bar(
            brand_df,
            x='brand',
            y=['settlement_rate', 'pending_value_ratio'],
            title="Brand-wise Settlement Analysis",
            barmode='group'
        )
        st.plotly_chart(fig_brand, use_container_width=True)
    
    # Settlement Resolution Tracking
    st.subheader("Settlement Resolution Tracking")
    
    # Resolution Time Analysis
    if 'resolution_patterns' in settlement_metrics:
        resolution_patterns = settlement_metrics['resolution_patterns']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Average Resolution Time",
                f"{resolution_patterns['average_resolution_time']:.1f} days"
            )
        with col2:
            st.metric(
                "Min Resolution Time",
                f"{resolution_patterns['min_resolution_time']} days"
            )
        with col3:
            st.metric(
                "Max Resolution Time",
                f"{resolution_patterns['max_resolution_time']} days"
            )
        
        # Resolution Time Distribution
        distribution = resolution_patterns['resolution_time_distribution']
        fig_dist = px.bar(
            x=list(distribution.keys()),
            y=list(distribution.values()),
            title="Resolution Time Distribution"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Settlement History
    st.subheader("Settlement History")
    
    # Monthly Trends
    if 'monthly_trends' in settlement_trends:
        monthly_data = []
        for month, data in settlement_trends['monthly_trends'].items():
            monthly_data.append({
                'month': month,
                'resolution_rate': data['resolution_rate'],
                'pending_count': data['pending_count']
            })
        
        monthly_df = pd.DataFrame(monthly_data)
        fig_monthly = px.line(
            monthly_df,
            x='month',
            y=['resolution_rate', 'pending_count'],
            title="Monthly Settlement Trends"
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Detailed Settlement Information
    st.subheader("Detailed Settlement Information")
    
    # New Pending Settlements
    st.write("New Pending Settlements")
    new_pending = settlement_metrics.get('new_pending_settlements', [])
    if new_pending:
        pending_df = pd.DataFrame(new_pending)
        st.dataframe(
            pending_df,
            column_config={
                "order_id": st.column_config.TextColumn("Order ID"),
                "pending_month": st.column_config.TextColumn("Pending Month"),
                "settlement_amount": st.column_config.NumberColumn("Amount", format="₹%.2f")
            },
            use_container_width=True
        )
    else:
        st.info("No new pending settlements in this run.")
    
    # Resolved Settlements
    st.write("Resolved Settlements")
    resolved = settlement_metrics.get('newly_resolved_settlements', [])
    if resolved:
        resolved_df = pd.DataFrame(resolved)
        st.dataframe(
            resolved_df,
            column_config={
                "order_id": st.column_config.TextColumn("Order ID"),
                "pending_month": st.column_config.TextColumn("Pending Month"),
                "resolution_month": st.column_config.TextColumn("Resolution Month"),
                "settlement_amount": st.column_config.NumberColumn("Amount", format="₹%.2f")
            },
            use_container_width=True
        )
    else:
        st.info("No settlements were resolved in this run.")
    
    # Actions
    st.subheader("Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Settlement Report"):
            # TODO: Implement export functionality
            st.info("Export functionality coming soon!")
    
    with col2:
        if st.button("Generate Resolution Analysis"):
            # TODO: Implement resolution analysis generation
            st.info("Resolution analysis generation coming soon!")

def display_orders_management():
    """Display enhanced orders management interface."""
    if st.session_state.analysis_df is None:
        st.warning("No analysis data available. Please upload and process files first.")
        return
    
    # Orders Overview Section
    st.subheader("Orders Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_orders = len(st.session_state.analysis_df)
        st.metric("Total Orders", total_orders)
    with col2:
        delivered_orders = len(st.session_state.analysis_df[
            st.session_state.analysis_df['status'] == 'Delivered - Settled'
        ])
        st.metric("Delivered Orders", delivered_orders)
    with col3:
        pending_orders = len(st.session_state.analysis_df[
            st.session_state.analysis_df['status'] == 'Delivered - Pending Settlement'
        ])
        st.metric("Pending Orders", pending_orders)
    with col4:
        returned_orders = len(st.session_state.analysis_df[
            st.session_state.analysis_df['status'].str.contains('Returned', na=False)
        ])
        st.metric("Returned Orders", returned_orders)
    
    # Filtering Options
    st.subheader("Filter Orders")
    col1, col2 = st.columns(2)
    
    with col1:
        # Date Range Filter
        if 'created_on' in st.session_state.analysis_df.columns:
            min_date = pd.to_datetime(st.session_state.analysis_df['created_on']).min()
            max_date = pd.to_datetime(st.session_state.analysis_df['created_on']).max()
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        # Status Filter
        status_options = st.session_state.analysis_df['status'].unique()
        selected_statuses = st.multiselect(
            "Order Status",
            options=status_options,
            default=status_options
        )
    
    with col2:
        # Brand Filter
        brand_options = st.session_state.analysis_df['brand'].unique()
        selected_brands = st.multiselect(
            "Brand",
            options=brand_options,
            default=brand_options
        )
        
        # SKU Filter
        if 'sku' in st.session_state.analysis_df.columns:
            sku = st.text_input("SKU (optional)")
    
    # Apply Filters
    filtered_df = st.session_state.analysis_df.copy()
    
    if 'created_on' in filtered_df.columns:
        filtered_df['created_on'] = pd.to_datetime(filtered_df['created_on'])
        filtered_df = filtered_df[
            (filtered_df['created_on'].dt.date >= date_range[0]) &
            (filtered_df['created_on'].dt.date <= date_range[1])
        ]
    
    filtered_df = filtered_df[
        (filtered_df['status'].isin(selected_statuses)) &
        (filtered_df['brand'].isin(selected_brands))
    ]
    
    if 'sku' in filtered_df.columns and sku:
        filtered_df = filtered_df[filtered_df['sku'].str.contains(sku, case=False, na=False)]
    
    # Order Analytics
    st.subheader("Order Analytics")
    
    # Status Distribution
    status_counts = filtered_df['status'].value_counts()
    fig_status = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Order Status Distribution"
    )
    st.plotly_chart(fig_status, use_container_width=True)
    
    # Brand-wise Distribution
    brand_counts = filtered_df['brand'].value_counts()
    fig_brand = px.bar(
        x=brand_counts.index,
        y=brand_counts.values,
        title="Orders by Brand"
    )
    st.plotly_chart(fig_brand, use_container_width=True)
    
    # Daily Order Trends
    if 'created_on' in filtered_df.columns:
        daily_orders = filtered_df.groupby(
            filtered_df['created_on'].dt.date
        ).size().reset_index(name='count')
        
        fig_trend = px.line(
            daily_orders,
            x='created_on',
            y='count',
            title="Daily Order Trends"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Order Details Table
    st.subheader("Order Details")
    
    # Column Selection
    available_columns = filtered_df.columns.tolist()
    selected_columns = st.multiselect(
        "Select Columns",
        options=available_columns,
        default=[
            'order_release_id', 'brand', 'status', 'created_on',
            'profit_loss', 'order_settlement', 'return_settlement'
        ]
    )
    
    # Display filtered table
    st.dataframe(
        filtered_df[selected_columns],
            column_config={
                "order_release_id": st.column_config.TextColumn("Order ID"),
            "brand": st.column_config.TextColumn("Brand"),
            "status": st.column_config.TextColumn("Status"),
            "created_on": st.column_config.DatetimeColumn("Created On"),
            "profit_loss": st.column_config.NumberColumn("Profit/Loss", format="₹%.2f"),
            "order_settlement": st.column_config.NumberColumn("Order Settlement", format="₹%.2f"),
            "return_settlement": st.column_config.NumberColumn("Return Settlement", format="₹%.2f")
            },
            use_container_width=True
        )
    
    # Actions
    st.subheader("Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export to CSV"):
            # TODO: Implement export functionality
            st.info("Export functionality coming soon!")
    
    with col2:
        if st.button("Save Filter Preset"):
            # TODO: Implement filter preset saving
            st.info("Filter preset saving coming soon!")

def display_returns_analysis():
    """Display enhanced returns analysis interface."""
    if st.session_state.analysis_df is None:
        st.warning("No analysis data available. Please upload and process files first.")
        return
    
    # Returns Overview Section
    st.subheader("Returns Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_returns = len(st.session_state.analysis_df[
            st.session_state.analysis_df['status'].str.contains('Returned', na=False)
        ])
        st.metric("Total Returns", total_returns)
    with col2:
        return_rate = (total_returns / len(st.session_state.analysis_df)) * 100
        st.metric("Return Rate", f"{return_rate:.2f}%")
    with col3:
        total_return_value = st.session_state.analysis_df[
            st.session_state.analysis_df['status'].str.contains('Returned', na=False)
        ]['return_settlement'].sum()
        st.metric("Total Return Value", f"₹{total_return_value:,.2f}")
    with col4:
        avg_return_value = total_return_value / total_returns if total_returns > 0 else 0
        st.metric("Average Return Value", f"₹{avg_return_value:,.2f}")
    
    # Filtering Options
    st.subheader("Filter Returns")
    col1, col2 = st.columns(2)
    
    with col1:
        # Date Range Filter
        if 'created_on' in st.session_state.analysis_df.columns:
            min_date = pd.to_datetime(st.session_state.analysis_df['created_on']).min()
            max_date = pd.to_datetime(st.session_state.analysis_df['created_on']).max()
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        # Return Type Filter
        return_types = st.session_state.analysis_df[
            st.session_state.analysis_df['status'].str.contains('Returned', na=False)
        ]['status'].unique()
        selected_types = st.multiselect(
            "Return Type",
            options=return_types,
            default=return_types
        )
    
    with col2:
        # Brand Filter
        brand_options = st.session_state.analysis_df['brand'].unique()
        selected_brands = st.multiselect(
            "Brand",
            options=brand_options,
            default=brand_options
        )
        
        # SKU Filter
        if 'sku' in st.session_state.analysis_df.columns:
            sku = st.text_input("SKU (optional)")
    
    # Apply Filters
    filtered_df = st.session_state.analysis_df[
        st.session_state.analysis_df['status'].str.contains('Returned', na=False)
    ].copy()
    
    if 'created_on' in filtered_df.columns:
        filtered_df['created_on'] = pd.to_datetime(filtered_df['created_on'])
        filtered_df = filtered_df[
            (filtered_df['created_on'].dt.date >= date_range[0]) &
            (filtered_df['created_on'].dt.date <= date_range[1])
        ]
    
    filtered_df = filtered_df[
        (filtered_df['status'].isin(selected_types)) &
        (filtered_df['brand'].isin(selected_brands))
    ]
    
    if 'sku' in filtered_df.columns and sku:
        filtered_df = filtered_df[filtered_df['sku'].str.contains(sku, case=False, na=False)]
    
    # Returns Analytics
    st.subheader("Returns Analytics")
    
    # Return Type Distribution
    return_counts = filtered_df['status'].value_counts()
    fig_type = px.pie(
        values=return_counts.values,
        names=return_counts.index,
        title="Return Type Distribution"
    )
    st.plotly_chart(fig_type, use_container_width=True)
    
    # Brand-wise Return Rate
    brand_returns = filtered_df.groupby('brand').agg({
        'order_release_id': 'count',
        'return_settlement': 'sum'
    }).reset_index()
    
    brand_returns['return_rate'] = (
        brand_returns['order_release_id'] / 
        st.session_state.analysis_df.groupby('brand')['order_release_id'].count() * 100
    )
    
    fig_brand = px.bar(
        brand_returns,
        x='brand',
        y=['return_rate', 'return_settlement'],
        title="Brand-wise Return Analysis",
        barmode='group'
    )
    st.plotly_chart(fig_brand, use_container_width=True)
    
    # Daily Return Trends
    if 'created_on' in filtered_df.columns:
        daily_returns = filtered_df.groupby(
            filtered_df['created_on'].dt.date
        ).agg({
            'order_release_id': 'count',
            'return_settlement': 'sum'
        }).reset_index()
        
        fig_trend = px.line(
            daily_returns,
            x='created_on',
            y=['order_release_id', 'return_settlement'],
            title="Daily Return Trends"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Return Patterns
    st.subheader("Return Patterns")
    
    # High Return Rate Items
    if 'sku' in filtered_df.columns:
        sku_returns = filtered_df.groupby('sku').agg({
            'order_release_id': 'count',
            'return_settlement': 'sum'
        }).reset_index()
        
        sku_returns['return_rate'] = (
            sku_returns['order_release_id'] / 
            st.session_state.analysis_df.groupby('sku')['order_release_id'].count() * 100
        )
        
        high_return_skus = sku_returns[sku_returns['return_rate'] > 10].sort_values(
            'return_rate', ascending=False
        )
        
        if not high_return_skus.empty:
            st.write("High Return Rate Items (>10%)")
    st.dataframe(
        high_return_skus,
        column_config={
            "sku": st.column_config.TextColumn("SKU"),
            "order_release_id": st.column_config.NumberColumn("Return Count"),
            "return_settlement": st.column_config.NumberColumn("Return Value", format="₹%.2f"),
            "return_rate": st.column_config.NumberColumn("Return Rate", format="%.2f%%")
        },
        use_container_width=True
    )
    
    # Return Details Table
    st.subheader("Return Details")
    
    # Column Selection
    available_columns = filtered_df.columns.tolist()
    selected_columns = st.multiselect(
        "Select Columns",
        options=available_columns,
        default=[
            'order_release_id', 'brand', 'status', 'created_on',
            'return_settlement', 'profit_loss'
        ]
    )
    
    # Display filtered table
    st.dataframe(
        filtered_df[selected_columns],
        column_config={
            "order_release_id": st.column_config.TextColumn("Order ID"),
            "brand": st.column_config.TextColumn("Brand"),
            "status": st.column_config.TextColumn("Status"),
            "created_on": st.column_config.DatetimeColumn("Created On"),
            "return_settlement": st.column_config.NumberColumn("Return Value", format="₹%.2f"),
            "profit_loss": st.column_config.NumberColumn("Profit/Loss", format="₹%.2f")
        },
        use_container_width=True
    )

    # Actions
    st.subheader("Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export to CSV"):
            # TODO: Implement export functionality
            st.info("Export functionality coming soon!")
    
    with col2:
        if st.button("Generate Return Report"):
            # TODO: Implement report generation
            st.info("Report generation coming soon!")

def display_reports_analytics():
    """Display the Reports & Analytics tab."""
    if st.session_state.analysis_df is None:
        st.warning("No analysis data available. Please upload and process files first.")
        return
    
    st.subheader("Reports & Analytics")
    
    # Report Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Select Report Type",
            ["Reconciliation Report", "Financial Summary", "Data Quality Report"]
        )
    
    with col2:
        report_format = st.selectbox(
            "Select Format",
            ["PDF", "CSV", "Excel"]
        )
    
    # Date Range Selection
    if 'created_on' in st.session_state.analysis_df.columns:
        min_date = pd.to_datetime(st.session_state.analysis_df['created_on']).min()
        max_date = pd.to_datetime(st.session_state.analysis_df['created_on']).max()
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data based on selected date range
        filtered_df = st.session_state.analysis_df[
            (pd.to_datetime(st.session_state.analysis_df['created_on']).dt.date >= date_range[0]) &
            (pd.to_datetime(st.session_state.analysis_df['created_on']).dt.date <= date_range[1])
        ]
    else:
        filtered_df = st.session_state.analysis_df
    
    # Generate Report Button
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            try:
                if report_type == "Reconciliation Report":
                    pdf_path = generate_pdf_report(st.session_state.summary, filtered_df, 'reconciliation')
                elif report_type == "Financial Summary":
                    pdf_path = generate_pdf_report(st.session_state.summary, filtered_df, 'financial')
                else:
                    pdf_path = generate_pdf_report(st.session_state.summary, filtered_df, 'data_quality')
                
                # Provide download button
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="Download Report",
                        data=file,
                        file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                
                st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")

def display_data_quality():
    """Display the Data Quality tab."""
    if st.session_state.analysis_df is None:
        st.warning("No analysis data available. Please upload and process files first.")
        return
    
    st.subheader("Data Quality Monitoring")
    
    # Calculate data quality metrics
    metrics = calculate_data_quality_metrics(
        st.session_state.orders_df,
        st.session_state.returns_df,
        st.session_state.settlement_df
    )
    
    # Data Completeness Section
    st.subheader("Data Completeness")
    
    for df_name, completeness in metrics['completeness'].items():
        st.write(f"**{df_name.title()} Data Completeness**")
        
        # Create a progress bar for each field
        for field, percentage in completeness.items():
            st.progress(percentage / 100, text=f"{field}: {percentage:.2f}%")
    
    # Orphaned Records Section
    st.subheader("Orphaned Records")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Orphaned Returns",
            metrics['orphaned_records']['returns'],
            help="Returns without matching orders"
        )
    
    with col2:
        st.metric(
            "Orphaned Settlements",
            metrics['orphaned_records']['settlements'],
            help="Settlements without matching orders"
        )
    
    # Validation Errors Section
    st.subheader("Validation Errors")
    
    if metrics['validation_errors']:
        error_df = pd.DataFrame(metrics['validation_errors'])
        st.dataframe(
            error_df,
            column_config={
                "file": st.column_config.TextColumn("File"),
                "field": st.column_config.TextColumn("Field"),
                "message": st.column_config.TextColumn("Error Message")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No validation errors found!")
    
    # Data Quality Score
    st.subheader("Overall Data Quality Score")
    
    # Calculate overall score based on completeness and validation errors
    completeness_scores = [
        sum(completeness.values()) / len(completeness)
        for completeness in metrics['completeness'].values()
    ]
    
    avg_completeness = sum(completeness_scores) / len(completeness_scores)
    validation_penalty = len(metrics['validation_errors']) * 5  # 5 points per validation error
    orphaned_penalty = sum(metrics['orphaned_records'].values()) * 2  # 2 points per orphaned record
    
    quality_score = max(0, min(100, avg_completeness - validation_penalty - orphaned_penalty))
    
    st.progress(quality_score / 100, text=f"Data Quality Score: {quality_score:.2f}%")
    
    # Recommendations
    st.subheader("Recommendations")
    
    recommendations = []
    
    # Completeness recommendations
    for df_name, completeness in metrics['completeness'].items():
        low_completeness_fields = [
            field for field, percentage in completeness.items()
            if percentage < 90
        ]
        if low_completeness_fields:
            recommendations.append(
                f"Improve data completeness for {df_name} fields: {', '.join(low_completeness_fields)}"
            )
    
    # Orphaned records recommendations
    if metrics['orphaned_records']['returns'] > 0:
        recommendations.append(
            f"Investigate {metrics['orphaned_records']['returns']} orphaned returns"
        )
    if metrics['orphaned_records']['settlements'] > 0:
        recommendations.append(
            f"Investigate {metrics['orphaned_records']['settlements']} orphaned settlements"
        )
    
    # Validation error recommendations
    if metrics['validation_errors']:
        recommendations.append(
            f"Address {len(metrics['validation_errors'])} validation errors"
        )
    
    if recommendations:
        for rec in recommendations:
            st.warning(rec)
    else:
        st.success("No immediate recommendations. Data quality is good!")

def handle_file_upload():
    """Handle file upload and processing."""
    try:
        # File upload section
        st.subheader("Upload Files")
        
        # Month and Year Selection
        col1, col2 = st.columns(2)
        with col1:
            selected_month = st.selectbox(
                "Select Month",
                options=[
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]
            )
        with col2:
            current_year = datetime.now().year
            selected_year = st.selectbox(
                "Select Year",
                options=range(current_year - 2, current_year + 1),
                index=2  # Default to current year
            )
        
        # Convert month name to number
        month_map = {
            "January": "01", "February": "02", "March": "03", "April": "04",
            "May": "05", "June": "06", "July": "07", "August": "08",
            "September": "09", "October": "10", "November": "11", "December": "12"
        }
        month_number = month_map[selected_month]
        
        # Check for existing data
        st.write("### Existing Data")
        existing_data = check_existing_data(selected_year, month_number)
        if existing_data:
            st.info(f"Data already exists for {selected_month} {selected_year}")
            display_existing_data_summary(existing_data)
            if not st.checkbox("I want to update the existing data"):
                st.warning("Please check the box above if you want to update the existing data")
                return
        
        # File upload section
        st.write("### Upload Files")
        st.info(f"Please upload files for {selected_month} {selected_year}")
        
        # Create columns for file uploads
        col1, col2, col3 = st.columns(3)
        
        with col1:
            orders_file = st.file_uploader(
                "Upload Orders File",
                type=["csv", "xlsx"],
                key="orders_uploader"
            )
        
        with col2:
            returns_file = st.file_uploader(
                "Upload Returns File",
                type=["csv", "xlsx"],
                key="returns_uploader"
            )
        
        with col3:
            settlement_file = st.file_uploader(
                "Upload Settlement File",
                type=["csv", "xlsx"],
                key="settlement_uploader"
            )
        
        # Process files if all are uploaded
        if orders_file and returns_file and settlement_file:
            try:
                # Create temporary files with our naming convention
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                
                # Save uploaded files with our naming convention
                orders_path = temp_dir / f"orders-{selected_year}-{month_number}.csv"
                returns_path = temp_dir / f"returns-{selected_year}-{month_number}.csv"
                settlement_path = temp_dir / f"settlement-{selected_year}-{month_number}.csv"
                
                # Save uploaded files
                with open(orders_path, "wb") as f:
                    f.write(orders_file.getvalue())
                with open(returns_path, "wb") as f:
                    f.write(returns_file.getvalue())
                with open(settlement_path, "wb") as f:
                    f.write(settlement_file.getvalue())
                
                # Clear caches before processing new data
                clear_caches()
                
                # Process each file
                orders_success, orders_result = process_orders_file(orders_path)
                returns_success, returns_result = process_returns_file(returns_path)
                settlement_success, settlement_result = process_settlement_file(settlement_path)
                
                # Clean up temporary files
                orders_path.unlink()
                returns_path.unlink()
                settlement_path.unlink()
                temp_dir.rmdir()
                
                # Store validation results
                st.session_state.validation_results = {
                    'orders': [orders_result],
                    'returns': [returns_result],
                    'settlement': [settlement_result]
                }
                
                # Display validation results
                display_validation_results(st.session_state.validation_results)
                
                # Check if there are any validation errors
                has_errors = not (orders_success and returns_success and settlement_success)
                
                if not has_errors:
                    try:
                        # Load master files with caching
                        orders_df = load_master_file(ORDERS_MASTER)
                        returns_df = load_master_file(RETURNS_MASTER)
                        settlement_df = load_master_file(SETTLEMENT_MASTER)
                        
                        # Store DataFrames in session state
                        st.session_state.orders_df = orders_df
                        st.session_state.returns_df = returns_df
                        st.session_state.settlement_df = settlement_df
                        
                        # Analyze orders with historical context
                        analysis_df, summary = analyze_orders_cached_v3(
                            os.path.getmtime(ORDERS_MASTER),
                            os.path.getmtime(RETURNS_MASTER),
                            os.path.getmtime(SETTLEMENT_MASTER)
                        )
                        
                        # Store results in session state
                        st.session_state.analysis_df = analysis_df
                        st.session_state.summary = summary
                        
                        # Generate visualizations
                        st.session_state.visualizations = generate_visualizations(analysis_df, summary)
                        
                        # Identify anomalies
                        st.session_state.anomalies_df = identify_anomalies(analysis_df)
                        
                        # Display success message
                        st.success(f"Files processed successfully for {selected_month} {selected_year}!")
                        
                        # Display summary metrics
                        st.subheader("Processing Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Orders", summary['total_orders'])
                        with col2:
                            st.metric("Total Returns", summary['total_returns'])
                        with col3:
                            st.metric("Total Settlements", summary['total_settlements'])
                        with col4:
                            st.metric("Anomalies", len(st.session_state.anomalies_df))
                    except Exception as e:
                        logger.error(f"Error processing data: {str(e)}")
                        st.error(f"Error processing data: {str(e)}")
                        st.exception(e)
                else:
                    st.error("Please fix validation errors before proceeding.")
            
            except Exception as e:
                logger.error(f"Error processing files: {str(e)}")
                st.error(f"Error processing files: {str(e)}")
                st.exception(e)
    except Exception as e:
        logger.error(f"Error in handle_file_upload: {str(e)}")
        st.error(f"Error in file upload: {str(e)}")
        st.exception(e)

def check_existing_data(year: str, month: str) -> Dict[str, Any]:
    """
    Check if data exists for the selected month and year.
    
    Args:
        year: Selected year
        month: Selected month number
    
    Returns:
        Dictionary containing existing data information
    """
    try:
        existing_data = {}
        
        # Check orders
        if os.path.exists(ORDERS_MASTER):
            orders_df = load_master_file(ORDERS_MASTER)
            if 'created_on' in orders_df.columns:
                # Convert to datetime robustly, coercing errors
                orders_df['created_on'] = pd.to_datetime(orders_df['created_on'], errors='coerce')
                # Filter out rows where conversion failed before filtering by date
                orders_df = orders_df.dropna(subset=['created_on'])
                if not orders_df.empty:
                    month_data = orders_df[orders_df['created_on'].dt.strftime('%Y-%m') == f"{year}-{month}"]
                    if not month_data.empty:
                        existing_data['orders'] = {
                            'count': len(month_data),
                            'total_value': month_data['order_value'].sum() if 'order_value' in month_data.columns else 0,
                            'last_updated': month_data['created_on'].max()
                        }
            else:
                logger.warning("'created_on' column not found in orders_master.csv for checking existing data.")
        
        # Check returns
        if os.path.exists(RETURNS_MASTER):
            returns_df = load_master_file(RETURNS_MASTER)
            if 'created_on' in returns_df.columns:
                returns_df['created_on'] = pd.to_datetime(returns_df['created_on'], errors='coerce')
                returns_df = returns_df.dropna(subset=['created_on'])
                if not returns_df.empty:
                    month_data = returns_df[returns_df['created_on'].dt.strftime('%Y-%m') == f"{year}-{month}"]
                    if not month_data.empty:
                        existing_data['returns'] = {
                            'count': len(month_data),
                            'total_value': month_data['return_value'].sum() if 'return_value' in month_data.columns else 0,
                            'last_updated': month_data['created_on'].max()
                        }
            else:
                logger.warning("'created_on' column not found in returns_master.csv for checking existing data.")
        
        # Check settlements
        if os.path.exists(SETTLEMENT_MASTER):
            settlement_df = load_master_file(SETTLEMENT_MASTER)
            if 'created_on' in settlement_df.columns:
                settlement_df['created_on'] = pd.to_datetime(settlement_df['created_on'], errors='coerce')
                settlement_df = settlement_df.dropna(subset=['created_on'])
                if not settlement_df.empty:
                    month_data = settlement_df[settlement_df['created_on'].dt.strftime('%Y-%m') == f"{year}-{month}"]
                    if not month_data.empty:
                        existing_data['settlement'] = {
                            'count': len(month_data),
                            'total_value': month_data['settlement_amount'].sum() if 'settlement_amount' in month_data.columns else 0,
                            'last_updated': month_data['created_on'].max()
                        }
            else:
                logger.warning("'created_on' column not found in settlement_master.csv for checking existing data.")
        
        return existing_data
    except Exception as e:
        logger.error(f"Error checking existing data: {str(e)}")
        return {}

def display_existing_data_summary(existing_data: Dict[str, Any]):
    """Display summary of existing data."""
    try:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'orders' in existing_data:
                st.metric(
                    "Existing Orders",
                    existing_data['orders']['count'],
                    f"Last updated: {existing_data['orders']['last_updated'].strftime('%Y-%m-%d')}"
                )
        
        with col2:
            if 'returns' in existing_data:
                st.metric(
                    "Existing Returns",
                    existing_data['returns']['count'],
                    f"Last updated: {existing_data['returns']['last_updated'].strftime('%Y-%m-%d')}"
                )
        
        with col3:
            if 'settlement' in existing_data:
                st.metric(
                    "Existing Settlements",
                    existing_data['settlement']['count'],
                    f"Last updated: {existing_data['settlement']['last_updated'].strftime('%Y-%m-%d')}"
                )
    except Exception as e:
        logger.error(f"Error displaying existing data summary: {str(e)}")

def remove_duplicates(new_df: pd.DataFrame, historical_df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """
    Remove duplicate records from new data based on ID column.
    
    Args:
        new_df: New data DataFrame
        historical_df: Historical data DataFrame
        id_column: Column to use for duplicate detection
    
    Returns:
        DataFrame with duplicates removed
    """
    try:
        # Get IDs from historical data
        historical_ids = set(historical_df[id_column])
        
        # Filter out records that already exist
        new_df = new_df[~new_df[id_column].isin(historical_ids)]
        
        return new_df
    except Exception as e:
        logger.error(f"Error removing duplicates: {str(e)}")
        return new_df

def main():
    """Main function to run the Streamlit application."""
    st.title("Order Reconciliation Dashboard")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "File Upload",
        "Dashboard",
        "Orders Management",
        "Returns Analysis",
        "Settlement Tracking",
        "Reports & Analytics",
        "Data Quality"
    ])
    
    with tab1:
        handle_file_upload()
    
    with tab2:
        display_dashboard()
    
    with tab3:
        display_orders_management()
    
    with tab4:
        display_returns_analysis()
    
    with tab5:
        display_settlements_management()
    
    with tab6:
        display_reports_analytics()
    
    with tab7:
        display_data_quality()

if __name__ == "__main__":
    ensure_directories_exist()
    main() 