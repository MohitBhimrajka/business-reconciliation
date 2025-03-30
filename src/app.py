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
    SETTLEMENT_MASTER, ORDERS_PATTERN, RETURNS_PATTERN, SETTLEMENT_PATTERN,
    validate_file_columns, get_file_identifier, format_currency, format_percentage,
    DATA_DIR
)
from ingestion import process_orders_file, process_returns_file, process_settlement_file
from analysis import analyze_orders, get_order_analysis_summary
from reporting import identify_anomalies, generate_visualizations

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

def load_existing_data():
    """Load existing data from files."""
    try:
        if os.path.exists(ANALYSIS_OUTPUT):
            st.session_state.analysis_df = read_file(ANALYSIS_OUTPUT)
            st.session_state.summary = get_order_analysis_summary(st.session_state.analysis_df)
            st.session_state.visualizations = generate_visualizations(
                st.session_state.analysis_df, st.session_state.summary
            )
        if os.path.exists(ANOMALIES_OUTPUT):
            st.session_state.anomalies_df = read_file(ANOMALIES_OUTPUT)
    except Exception as e:
        st.error(f"Error loading existing data: {e}")

def process_uploaded_files():
    """Process newly uploaded files and update analysis."""
    try:
        # Process each new file
        for file_path in st.session_state.newly_added_files:
            file_type = file_path.split('-')[0]  # Extract type from filename
            if file_type == 'orders':
                process_orders_file(DATA_DIR / file_path)
            elif file_type == 'returns':
                process_returns_file(DATA_DIR / file_path)
            elif file_type == 'settlement':
                process_settlement_file(DATA_DIR / file_path)
        
        # Load master files
        orders_df = read_file(ORDERS_MASTER)
        returns_df = read_file(RETURNS_MASTER)
        settlement_df = read_file(SETTLEMENT_MASTER)
        
        # Load previous analysis
        previous_analysis_df = None
        if os.path.exists(ANALYSIS_OUTPUT):
            previous_analysis_df = read_file(ANALYSIS_OUTPUT)
        
        # Run analysis
        st.session_state.analysis_df = analyze_orders(
            orders_df, returns_df, settlement_df, previous_analysis_df
        )
        st.session_state.summary = get_order_analysis_summary(st.session_state.analysis_df)
        
        # Generate visualizations
        st.session_state.visualizations = generate_visualizations(
            st.session_state.analysis_df, st.session_state.summary
        )
        
        # Identify anomalies
        st.session_state.anomalies_df = identify_anomalies(
            st.session_state.analysis_df, orders_df, returns_df, settlement_df
        )
        
        # Clear newly added files
        st.session_state.newly_added_files = []
        
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
        st.metric("Net Profit/Loss", format_currency(st.session_state.summary['net_profit_loss']))
    with col3:
        st.metric("Settlement Rate", format_percentage(st.session_state.summary['settlement_rate']))
    with col4:
        st.metric("Return Rate", format_percentage(st.session_state.summary['return_rate']))
    
    # Display charts
    if st.session_state.visualizations:
        # Order Status Distribution
        st.plotly_chart(
            st.session_state.visualizations['status_distribution'],
            use_container_width=True
        )
        
        # Profit/Loss Distribution
        st.plotly_chart(
            st.session_state.visualizations['profit_loss_distribution'],
            use_container_width=True
        )
        
        # Monthly Trends
        if 'monthly_orders_trend' in st.session_state.visualizations:
            st.subheader("Monthly Trends")
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    st.session_state.visualizations['monthly_orders_trend'],
                    use_container_width=True
                )
                st.plotly_chart(
                    st.session_state.visualizations['monthly_profit_loss_trend'],
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    st.session_state.visualizations['monthly_settlement_rate_trend'],
                    use_container_width=True
                )
        
        # Settlement Changes
        if 'settlement_changes' in st.session_state.visualizations:
            st.plotly_chart(
                st.session_state.visualizations['settlement_changes'],
                use_container_width=True
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

def display_settlement_updates():
    """Display settlement tracking information."""
    if st.session_state.analysis_df is None:
        st.warning("No analysis data available. Please upload and process files first.")
        return
    
    # Display settlement metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Orders Settled This Run",
            st.session_state.summary['settlement_changes']
        )
    with col2:
        st.metric(
            "Orders Newly Pending",
            st.session_state.summary['pending_changes']
        )
    
    # Display recently settled orders
    st.subheader("Recently Settled Orders")
    settled_orders = st.session_state.analysis_df[
        (st.session_state.analysis_df['status_changed_this_run']) &
        (st.session_state.analysis_df['status'] == 'Completed - Settled')
    ]
    
    if not settled_orders.empty:
        st.dataframe(
            settled_orders,
            column_config={
                "order_release_id": st.column_config.TextColumn("Order ID"),
                "profit_loss": st.column_config.NumberColumn("Profit/Loss", format="₹%.2f"),
                "settlement_update_run_timestamp": st.column_config.DatetimeColumn("Settlement Date")
            },
            use_container_width=True
        )
    else:
        st.info("No orders were settled in this run.")
    
    # Display pending settlement orders
    st.subheader("Pending Settlement Orders")
    pending_orders = st.session_state.analysis_df[
        st.session_state.analysis_df['status'] == 'Completed - Pending Settlement'
    ]
    
    if not pending_orders.empty:
        st.dataframe(
            pending_orders,
            column_config={
                "order_release_id": st.column_config.TextColumn("Order ID"),
                "final_amount": st.column_config.NumberColumn("Order Amount", format="₹%.2f")
            },
            use_container_width=True
        )
    else:
        st.info("No orders are pending settlement.")

def display_anomalies():
    """Display identified anomalies."""
    if st.session_state.anomalies_df is None:
        st.warning("No anomalies data available. Please upload and process files first.")
        return
    
    st.dataframe(
        st.session_state.anomalies_df,
        column_config={
            "type": st.column_config.TextColumn("Anomaly Type"),
            "order_release_id": st.column_config.TextColumn("Order ID"),
            "details": st.column_config.TextColumn("Details")
        },
        use_container_width=True
    )

def handle_file_upload(uploaded_file, file_type: str, month: str, year: str):
    """Handle file upload and validation."""
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        # Validate columns
        if not validate_file_columns(df, file_type):
            st.error(f"Invalid columns in {file_type} file. Please check the file format.")
            return False
        
        # Generate standard filename
        standard_filename = get_file_identifier(file_type, month, year)
        target_path = DATA_DIR / standard_filename
        
        # Check if file exists
        if target_path.exists():
            existing_df = read_file(target_path)
            
            # Compare files
            st.warning(f"File {standard_filename} already exists.")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Existing file:")
                st.write(f"- Rows: {len(existing_df)}")
                st.write(f"- Columns: {', '.join(existing_df.columns)}")
            with col2:
                st.write("Uploaded file:")
                st.write(f"- Rows: {len(df)}")
                st.write(f"- Columns: {', '.join(df.columns)}")
            
            # Overwrite confirmation
            if not st.button("Overwrite Existing File"):
                return False
        
        # Save file
        df.to_csv(target_path, index=False)
        st.success(f"File saved as {standard_filename}")
        
        # Add to newly added files
        st.session_state.newly_added_files.append(standard_filename)
        return True
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return False

def main():
    """Main application function."""
    st.title("Order Reconciliation Dashboard")
    
    # Load existing data
    load_existing_data()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "File Upload",
        "Dashboard",
        "Detailed Analysis",
        "Master Data",
        "Settlement Updates",
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
            if handle_file_upload(uploaded_file, file_type, month, year):
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
        st.header("Settlement Updates")
        display_settlement_updates()
    
    with tab6:
        st.header("Anomalies")
        display_anomalies()

if __name__ == "__main__":
    ensure_directories_exist()
    main() 