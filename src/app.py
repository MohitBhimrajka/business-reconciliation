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

from utils import (
    ensure_directories_exist, read_file, ANALYSIS_OUTPUT, REPORT_OUTPUT,
    VISUALIZATION_DIR, ANOMALIES_OUTPUT, ORDERS_MASTER, RETURNS_MASTER,
    SETTLEMENT_MASTER, ORDERS_PATTERN, RETURNS_PATTERN, SETTLEMENT_PATTERN
)
from ingestion import process_orders_file, process_returns_file, process_settlement_file
from analysis import analyze_orders, get_order_analysis_summary
from reporting import identify_anomalies

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

def load_existing_data():
    """Load existing data from files."""
    try:
        if os.path.exists(ANALYSIS_OUTPUT):
            st.session_state.analysis_df = read_file(ANALYSIS_OUTPUT)
            st.session_state.summary = get_order_analysis_summary(st.session_state.analysis_df)
        if os.path.exists(ANOMALIES_OUTPUT):
            st.session_state.anomalies_df = read_file(ANOMALIES_OUTPUT)
    except Exception as e:
        st.error(f"Error loading existing data: {e}")

def process_uploaded_files():
    """Process newly uploaded files and update analysis."""
    try:
        # Process each uploaded file
        for file_info in st.session_state.uploaded_files:
            file_path = file_info['path']
            file_type = file_info['type']
            
            if file_type == 'orders':
                process_orders_file(file_path)
            elif file_type == 'returns':
                process_returns_file(file_path)
            elif file_type == 'settlement':
                process_settlement_file(file_path)
        
        # Load master files
        orders_df = read_file(ORDERS_MASTER)
        returns_df = read_file(RETURNS_MASTER)
        settlement_df = read_file(SETTLEMENT_MASTER)
        
        # Run analysis
        st.session_state.analysis_df = analyze_orders(orders_df, returns_df, settlement_df)
        st.session_state.summary = get_order_analysis_summary(st.session_state.analysis_df)
        
        # Identify anomalies
        st.session_state.anomalies_df = identify_anomalies(
            st.session_state.analysis_df, orders_df, returns_df, settlement_df
        )
        
        # Clear uploaded files
        st.session_state.uploaded_files = []
        
        st.success("Files processed successfully!")
    except Exception as e:
        st.error(f"Error processing files: {e}")

def display_dashboard():
    """Display the main dashboard with metrics and charts."""
    if st.session_state.summary is None:
        st.warning("No analysis data available. Please upload and process files first.")
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Orders", st.session_state.summary['total_orders'])
    with col2:
        st.metric("Net Profit/Loss", f"₹{st.session_state.summary['net_profit_loss']:,.2f}")
    with col3:
        st.metric("Settlement Rate", f"{st.session_state.summary['settlement_rate']:.2f}%")
    with col4:
        st.metric("Return Rate", f"{st.session_state.summary['return_rate']:.2f}%")
    
    # Display charts
    col1, col2 = st.columns(2)
    with col1:
        # Order Status Distribution
        status_counts = st.session_state.analysis_df['status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Order Status Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Profit/Loss Distribution
        profit_loss_data = st.session_state.analysis_df[
            st.session_state.analysis_df['profit_loss'].notna()
        ]
        fig = px.histogram(
            profit_loss_data,
            x='profit_loss',
            title="Profit/Loss Distribution",
            nbins=50
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    # Display status changes
    if 'status_changed_this_run' in st.session_state.analysis_df.columns:
        status_changes = st.session_state.analysis_df[
            st.session_state.analysis_df['status_changed_this_run']
        ]
        if not status_changes.empty:
            st.subheader("Recent Status Changes")
            st.dataframe(
                status_changes[['order_release_id', 'status', 'profit_loss']],
                use_container_width=True
            )

def display_detailed_analysis():
    """Display detailed order analysis in an interactive table."""
    if st.session_state.analysis_df is None:
        st.warning("No analysis data available. Please upload and process files first.")
        return
    
    st.dataframe(
        st.session_state.analysis_df,
        use_container_width=True
    )

def display_master_data():
    """Display master data files in interactive tables."""
    col1, col2, col3 = st.tabs(["Orders", "Returns", "Settlement"])
    
    with col1:
        if os.path.exists(ORDERS_MASTER):
            orders_df = read_file(ORDERS_MASTER)
            st.dataframe(orders_df, use_container_width=True)
        else:
            st.warning("No orders master data available.")
    
    with col2:
        if os.path.exists(RETURNS_MASTER):
            returns_df = read_file(RETURNS_MASTER)
            st.dataframe(returns_df, use_container_width=True)
        else:
            st.warning("No returns master data available.")
    
    with col3:
        if os.path.exists(SETTLEMENT_MASTER):
            settlement_df = read_file(SETTLEMENT_MASTER)
            st.dataframe(settlement_df, use_container_width=True)
        else:
            st.warning("No settlement master data available.")

def display_anomalies():
    """Display identified anomalies."""
    if st.session_state.anomalies_df is None:
        st.warning("No anomalies data available. Please upload and process files first.")
        return
    
    st.dataframe(
        st.session_state.anomalies_df,
        use_container_width=True
    )

def main():
    """Main application function."""
    st.title("Order Reconciliation Dashboard")
    
    # Load existing data
    load_existing_data()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "File Upload",
        "Dashboard",
        "Detailed Analysis",
        "Master Data",
        "Anomalies"
    ])
    
    with tab1:
        st.header("File Upload")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            file_type = st.selectbox(
                "File Type",
                ["orders", "returns", "settlement"]
            )
        
        with col2:
            month = st.selectbox(
                "Month",
                [f"{i:02d}" for i in range(1, 13)]
            )
        
        with col3:
            year = st.selectbox(
                "Year",
                [2023, 2024, 2025]
            )
        
        uploaded_file = st.file_uploader(
            "Upload File",
            type=["csv", "xlsx"],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Save file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Add to uploaded files
            st.session_state.uploaded_files.append({
                'path': temp_path,
                'type': file_type,
                'month': month,
                'year': year
            })
            
            st.success(f"File uploaded successfully: {uploaded_file.name}")
        
        if st.session_state.uploaded_files:
            st.subheader("Uploaded Files")
            for file_info in st.session_state.uploaded_files:
                st.write(f"- {file_info['type']} ({file_info['month']}/{file_info['year']})")
            
            if st.button("Process Uploaded Files"):
                process_uploaded_files()
    
    with tab2:
        st.header("Dashboard")
        display_dashboard()
    
    with tab3:
        st.header("Detailed Analysis")
        display_detailed_analysis()
    
    with tab4:
        st.header("Master Data")
        display_master_data()
    
    with tab5:
        st.header("Anomalies")
        display_anomalies()

if __name__ == "__main__":
    ensure_directories_exist()
    main() 