"""
Order analysis module for reconciliation application.
"""
import pandas as pd
import logging
import numpy as np
from typing import Dict, Tuple, List, Optional
from datetime import datetime
from collections import defaultdict

from utils import ensure_directories_exist, read_file, ANALYSIS_OUTPUT

logger = logging.getLogger(__name__)

class SettlementTracker:
    """Enhanced settlement tracking with comprehensive analysis capabilities."""
    
    def __init__(self):
        self.settlement_history = {}
        self.resolution_patterns = {}
        self.brand_analysis = {}
    
    def track_settlements(
        self,
        current_df: pd.DataFrame,
        previous_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Track settlements with enhanced metrics and analysis.
        
        Args:
            current_df: Current analysis DataFrame
            previous_df: Previous analysis DataFrame for comparison
            
        Returns:
            Dictionary containing settlement metrics and analysis
        """
        # Get current month
        current_month = pd.Timestamp.now().strftime('%Y-%m')
        
        # Initialize metrics
        metrics = {
            'current_month': current_month,
            'new_pending_settlements': [],
            'newly_resolved_settlements': [],
            'settlement_resolution_rates': {},
            'brand_wise_metrics': {},
            'resolution_patterns': {},
            'settlement_history': {}
        }
        
        # Track new pending settlements
        current_pending = current_df[
            (current_df['status'] == 'Delivered - Pending Settlement')
        ]
        
        if previous_df is not None:
            previous_pending = previous_df[
                (previous_df['status'] == 'Delivered - Pending Settlement')
            ]
            previous_pending_ids = set(previous_pending['order_id'])
            
            # Find new pending settlements
            new_pending = current_pending[
                ~current_pending['order_id'].isin(previous_pending_ids)
            ]
            
            # Find resolved settlements
            resolved = previous_pending[
                ~previous_pending['order_id'].isin(set(current_pending['order_id']))
            ]
            
            # Add to metrics
            metrics['new_pending_settlements'] = [
                {
                    'order_id': row['order_id'],
                    'pending_month': current_month,
                    'settlement_amount': row['order_settlement']
                }
                for _, row in new_pending.iterrows()
            ]
            
            metrics['newly_resolved_settlements'] = [
                {
                    'order_id': row['order_id'],
                    'pending_month': row['settlement_month'],
                    'resolution_month': current_month,
                    'settlement_amount': row['order_settlement']
                }
                for _, row in resolved.iterrows()
            ]
        
        # Calculate resolution rates
        all_months = sorted(set(current_df['settlement_month'].dropna()))
        for month in all_months:
            month_pending = current_df[
                (current_df['settlement_month'] == month) &
                (current_df['status'] == 'Delivered - Pending Settlement')
            ]
            month_resolved = current_df[
                (current_df['settlement_month'] == month) &
                (current_df['status'] == 'Delivered - Settled')
            ]
            
            total = len(month_pending) + len(month_resolved)
            if total > 0:
                resolution_rate = (len(month_resolved) / total) * 100
                metrics['settlement_resolution_rates'][month] = resolution_rate
        
        # Brand-wise analysis
        for brand in current_df['brand'].unique():
            brand_df = current_df[current_df['brand'] == brand]
            brand_metrics = {
                'total_orders': len(brand_df),
                'pending_settlements': len(brand_df[
                    brand_df['status'] == 'Delivered - Pending Settlement'
                ]),
                'settled_orders': len(brand_df[
                    brand_df['status'] == 'Delivered - Settled'
                ]),
                'total_settlement_value': brand_df['order_settlement'].sum(),
                'pending_settlement_value': brand_df[
                    brand_df['status'] == 'Delivered - Pending Settlement'
                ]['order_settlement'].sum()
            }
            metrics['brand_wise_metrics'][brand] = brand_metrics
        
        # Resolution patterns
        if previous_df is not None:
            resolution_times = []
            for _, row in metrics['newly_resolved_settlements']:
                pending_date = pd.Timestamp(row['pending_month'] + '-01')
                resolution_date = pd.Timestamp(row['resolution_month'] + '-01')
                resolution_time = (resolution_date - pending_date).days
                resolution_times.append(resolution_time)
            
            if resolution_times:
                metrics['resolution_patterns'] = {
                    'average_resolution_time': sum(resolution_times) / len(resolution_times),
                    'min_resolution_time': min(resolution_times),
                    'max_resolution_time': max(resolution_times),
                    'resolution_time_distribution': {
                        '0-30_days': len([t for t in resolution_times if t <= 30]),
                        '31-60_days': len([t for t in resolution_times if 30 < t <= 60]),
                        '61-90_days': len([t for t in resolution_times if 60 < t <= 90]),
                        '90+_days': len([t for t in resolution_times if t > 90])
                    }
                }
        
        # Settlement history
        for _, row in current_df.iterrows():
            order_id = row['order_id']
            if order_id not in metrics['settlement_history']:
                metrics['settlement_history'][order_id] = {
                    'order_id': order_id,
                    'brand': row['brand'],
                    'status_changes': [],
                    'settlement_amount': row['order_settlement'],
                    'current_status': row['status']
                }
            
            if row['status'] != metrics['settlement_history'][order_id]['current_status']:
                metrics['settlement_history'][order_id]['status_changes'].append({
                    'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                    'from_status': metrics['settlement_history'][order_id]['current_status'],
                    'to_status': row['status']
                })
                metrics['settlement_history'][order_id]['current_status'] = row['status']
        
        return metrics
    
    def analyze_settlement_trends(self, metrics: Dict) -> Dict:
        """
        Analyze settlement trends and patterns.
        
        Args:
            metrics: Settlement tracking metrics
            
        Returns:
            Dictionary containing trend analysis
        """
        trends = {
            'monthly_trends': {},
            'brand_trends': {},
            'resolution_trends': {}
        }
        
        # Monthly trends
        for month, rate in metrics['settlement_resolution_rates'].items():
            trends['monthly_trends'][month] = {
                'resolution_rate': rate,
                'pending_count': len([
                    s for s in metrics['settlement_history'].values()
                    if s['current_status'] == 'Delivered - Pending Settlement'
                    and s['settlement_month'] == month
                ])
            }
        
        # Brand trends
        for brand, brand_metrics in metrics['brand_wise_metrics'].items():
            trends['brand_trends'][brand] = {
                'settlement_rate': (
                    brand_metrics['settled_orders'] / brand_metrics['total_orders'] * 100
                ) if brand_metrics['total_orders'] > 0 else 0,
                'pending_value_ratio': (
                    brand_metrics['pending_settlement_value'] / brand_metrics['total_settlement_value'] * 100
                ) if brand_metrics['total_settlement_value'] > 0 else 0
            }
        
        # Resolution trends
        if 'resolution_patterns' in metrics:
            trends['resolution_trends'] = {
                'average_resolution_time': metrics['resolution_patterns']['average_resolution_time'],
                'resolution_time_distribution': metrics['resolution_patterns']['resolution_time_distribution']
            }
        
        return trends
    
    def generate_settlement_report(
        self,
        metrics: Dict,
        trends: Dict
    ) -> Dict:
        """
        Generate comprehensive settlement report.
        
        Args:
            metrics: Settlement tracking metrics
            trends: Settlement trend analysis
            
        Returns:
            Dictionary containing comprehensive report
        """
        report = {
            'summary': {
                'current_month': metrics['current_month'],
                'total_pending_settlements': len([
                    s for s in metrics['settlement_history'].values()
                    if s['current_status'] == 'Delivered - Pending Settlement'
                ]),
                'total_settlement_value': sum(
                    s['settlement_amount']
                    for s in metrics['settlement_history'].values()
                ),
                'current_resolution_rate': metrics['settlement_resolution_rates'].get(
                    metrics['current_month'],
                    0
                )
            },
            'monthly_analysis': {
                month: {
                    'resolution_rate': data['resolution_rate'],
                    'pending_count': data['pending_count']
                }
                for month, data in trends['monthly_trends'].items()
            },
            'brand_analysis': {
                brand: {
                    'settlement_rate': data['settlement_rate'],
                    'pending_value_ratio': data['pending_value_ratio']
                }
                for brand, data in trends['brand_trends'].items()
            },
            'resolution_analysis': trends['resolution_trends'],
            'recent_changes': {
                'new_pending': metrics['new_pending_settlements'],
                'newly_resolved': metrics['newly_resolved_settlements']
            }
        }
        
        return report

class OrderAnalyzer:
    """Enhanced order analysis with comprehensive settlement tracking."""
    
    def __init__(self):
        self.status_mapping = {
            'C': 'Cancelled',
            'D': 'Delivered',
            'RTO': 'Returned to Origin'
        }
        self.settlement_tracker = SettlementTracker()
    
    def analyze_orders(
        self,
        orders_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        settlements_df: pd.DataFrame,
        previous_analysis_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze orders with enhanced settlement tracking.
        
        Args:
            orders_df: Orders DataFrame
            returns_df: Returns DataFrame
            settlements_df: Settlements DataFrame
            previous_analysis_df: Previous analysis DataFrame (if available)
            
        Returns:
            Tuple of (analysis_df, analysis_summary)
        """
        # Initialize analysis DataFrame
        analysis_df = orders_df.copy()
        
        # Add settlement tracking columns if not present
        for col in ['settlement_month', 'settlement_resolved_date', 'settlement_resolution_month']:
            if col not in analysis_df.columns:
                analysis_df[col] = None
        
        # Determine order status and calculate financials
        analysis_df = self.determine_order_status_and_financials(
            analysis_df, returns_df, settlements_df
        )
        
        # Track settlements
        settlement_metrics = self.settlement_tracker.track_settlements(
            analysis_df, previous_analysis_df
        )
        
        # Analyze settlement trends
        settlement_trends = self.settlement_tracker.analyze_settlement_trends(
            settlement_metrics
        )
        
        # Generate settlement report
        settlement_report = self.settlement_tracker.generate_settlement_report(
            settlement_metrics, settlement_trends
        )
        
        # Calculate core metrics
        core_metrics = self.calculate_core_metrics(analysis_df)
        
        # Generate analysis summary
        analysis_summary = {
            'total_orders': len(analysis_df),
            'net_profit_loss': analysis_df['profit_loss'].sum(),
            'settlement_rate': core_metrics['settlement_rate'],
            'return_rate': core_metrics['return_rate'],
            'settlement_metrics': settlement_metrics,
            'settlement_trends': settlement_trends,
            'settlement_report': settlement_report,
            'core_metrics': core_metrics
        }
        
        return analysis_df, analysis_summary
    
    def determine_order_status(
        self,
        order_data: Dict,
        returns_data: List[Dict],
        settlement_data: List[Dict]
    ) -> str:
        """
        Determine order status based on order, returns, and settlement data.
        
        Args:
            order_data: Order information
            returns_data: List of returns for the order
            settlement_data: List of settlements for the order
        
        Returns:
            Order status string
        """
        order_id = order_data['order_release_id']
        
        # Case 1: Cancelled before shipment
        if order_data['is_ship_rel'] == 0 and order_data['order_status'] == 'C':
            return "Cancelled - No Impact"
        
        # Case 2: Cancelled after shipment
        if order_data['is_ship_rel'] == 1 and order_data['order_status'] == 'C':
            return "Cancelled - After Shipment"
        
        # Case 3: Delivered orders
        if order_data['order_status'] == 'D':
            # Check for returns
            if returns_data:
                return_type = returns_data[0]['return_type']
                if return_type == 'return_refund':
                    return "Returned - Refunded"
                elif return_type == 'exchange':
                    return "Returned - Exchanged"
            
            # Check for settlement
            if settlement_data:
                return "Delivered - Settled"
            return "Delivered - Pending Settlement"
        
        # Case 4: RTO orders
        if order_data['order_status'] == 'RTO':
            return "RTO - Returned"
        
        return "Unknown Status"
    
    def calculate_profit_loss(
        self,
        order_data: Dict,
        returns_data: List[Dict],
        settlement_data: List[Dict]
    ) -> float:
        """
        Calculate profit/loss for an order.
        
        Args:
            order_data: Order information
            returns_data: List of returns for the order
            settlement_data: List of settlements for the order
        
        Returns:
            Profit/loss amount
        """
        # Case 1: Cancelled before shipment
        if order_data['is_ship_rel'] == 0 and order_data['order_status'] == 'C':
            return 0.0
        
        # Get settlement amount
        settlement_amount = 0.0
        if settlement_data:
            settlement_amount = settlement_data[0]['total_actual_settlement']
        
        # Case 2: Delivered orders (no returns)
        if not returns_data:
            return settlement_amount
        
        # Case 3: Returned/Exchanged orders
        return_amount = returns_data[0]['total_actual_settlement']
        return settlement_amount - return_amount
    
    def calculate_core_metrics(self, analysis_df: pd.DataFrame) -> Dict:
        """
        Calculate core metrics from analysis results.
        
        Args:
            analysis_df: Analysis results DataFrame
        
        Returns:
            Dictionary of core metrics
        """
        metrics = {}
        
        # 1. AOV (Average Order Value)
        metrics['aov'] = analysis_df['total_actual_settlement'].mean()
        
        # 2. Return Rate
        total_orders = len(analysis_df)
        returned_orders = len(analysis_df[analysis_df['status'].str.contains('Returned', na=False)])
        metrics['return_rate'] = (returned_orders / total_orders) * 100 if total_orders > 0 else 0
        
        # 3. Settlement Rate
        settled_orders = len(analysis_df[analysis_df['status'] == 'Delivered - Settled'])
        metrics['settlement_rate'] = (settled_orders / total_orders) * 100 if total_orders > 0 else 0
        
        # 4. Net Profit/Loss
        metrics['net_profit_loss'] = analysis_df['profit_loss'].sum()
        
        # 5. Commission Rate
        total_settlement = analysis_df['total_actual_settlement'].sum()
        total_commission = analysis_df['total_commission'].sum()
        metrics['commission_rate'] = (total_commission / total_settlement) * 100 if total_settlement > 0 else 0
        
        # 6. Logistics Cost Ratio
        total_logistics = analysis_df['total_logistics_deduction'].sum()
        metrics['logistics_cost_ratio'] = (total_logistics / total_settlement) * 100 if total_settlement > 0 else 0
        
        # 7. Tax Rate
        total_tax = analysis_df['tcs_amount'].sum() + analysis_df['tds_amount'].sum()
        metrics['tax_rate'] = (total_tax / total_settlement) * 100 if total_settlement > 0 else 0
        
        return metrics
    
    def determine_payment_status(
        self,
        expected: float,
        actual: float,
        pending: float
    ) -> str:
        """
        Determine payment status based on settlement amounts.
        
        Args:
            expected: Expected settlement amount
            actual: Actual settlement amount
            pending: Pending settlement amount
        
        Returns:
            Payment status string
        """
        if actual >= expected:
            return "Paid"
        elif actual > 0:
            return "Partial"
        return "Pending"
    
    def calculate_financial_breakdown(
        self,
        order_data: Dict,
        returns_data: List[Dict],
        settlement_data: List[Dict]
    ) -> Dict:
        """
        Calculate financial breakdown for an order.
        
        Args:
            order_data: Order information
            returns_data: List of returns for the order
            settlement_data: List of settlements for the order
        
        Returns:
            Dictionary of financial breakdown
        """
        breakdown = {}
        
        # Get settlement data
        if settlement_data:
            settlement = settlement_data[0]
            breakdown['net_revenue'] = settlement['total_actual_settlement']
            breakdown['commission_cost'] = settlement['total_commission']
            breakdown['logistics_cost'] = settlement['total_logistics_deduction']
            breakdown['tax_deductions'] = settlement['tcs_amount'] + settlement['tds_amount']
        
        # Calculate net profit/loss
        breakdown['net_profit_loss'] = self.calculate_profit_loss(
            order_data, returns_data, settlement_data
        )
        
        return breakdown

def analyze_orders(
    orders_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    settlement_df: pd.DataFrame,
    previous_analysis_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze orders and determine their status and financials.
    
    Args:
        orders_df: Orders DataFrame
        returns_df: Returns DataFrame
        settlement_df: Settlement DataFrame
        previous_analysis_df: Previous analysis results (if available)
    
    Returns:
        Tuple of (DataFrame with analysis results, settlement metrics)
    """
    if orders_df.empty:
        logger.warning("No orders data available for analysis")
        return pd.DataFrame(), {}
    
    # Create analyzer instance
    analyzer = OrderAnalyzer()
    
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
        order_id = order['order_release_id']
        
        # Get related data
        order_returns = returns_df[returns_df['order_release_id'] == order_id].to_dict('records')
        order_settlements = settlement_df[settlement_df['order_release_id'] == order_id].to_dict('records')
        
        # Determine status
        status = analyzer.determine_order_status(order, order_returns, order_settlements)
        
        # Calculate profit/loss
        profit_loss = analyzer.calculate_profit_loss(order, order_returns, order_settlements)
        
        # Calculate financial breakdown
        financials = analyzer.calculate_financial_breakdown(order, order_returns, order_settlements)
        
        # Track status changes
        previous_status = previous_status_map.get(order_id)
        status_changed = previous_status is not None and previous_status != status
        
        # Update analysis DataFrame
        analysis_df.at[idx, 'status'] = status
        analysis_df.at[idx, 'profit_loss'] = profit_loss
        analysis_df.at[idx, 'return_settlement'] = order_returns[0]['total_actual_settlement'] if order_returns else 0
        analysis_df.at[idx, 'order_settlement'] = order_settlements[0]['total_actual_settlement'] if order_settlements else 0
        analysis_df.at[idx, 'status_changed_this_run'] = status_changed
        analysis_df.at[idx, 'settlement_update_run_timestamp'] = datetime.now().isoformat() if status_changed else None
        
        # Add financial breakdown columns
        for key, value in financials.items():
            analysis_df.at[idx, key] = value
    
    # Track settlements
    settlement_metrics = analyzer.settlement_tracker.track_settlements(
        analysis_df, previous_analysis_df
    )
    
    return analysis_df, settlement_metrics

def get_order_analysis_summary(analysis_df: pd.DataFrame, settlement_metrics: Dict) -> Dict:
    """
    Generate summary statistics from order analysis.
    
    Args:
        analysis_df: Analysis results DataFrame
        settlement_metrics: Settlement tracking metrics
    
    Returns:
        Dictionary containing summary statistics
    """
    analyzer = OrderAnalyzer()
    
    # Calculate core metrics
    metrics = analyzer.calculate_core_metrics(analysis_df)
    
    # Get status counts
    status_counts = analysis_df['status'].value_counts()
    
    # Calculate settlement changes
    status_changes = analysis_df['status_changed_this_run'].sum()
    settlement_changes = len(analysis_df[
        (analysis_df['status_changed_this_run']) &
        (analysis_df['status'] == 'Delivered - Settled')
    ])
    pending_changes = len(analysis_df[
        (analysis_df['status_changed_this_run']) &
        (analysis_df['status'] == 'Delivered - Pending Settlement')
    ])
    
    # Calculate settlement amounts
    total_return_settlement = analysis_df['return_settlement'].sum()
    total_order_settlement = analysis_df['order_settlement'].sum()
    
    # Add settlement tracking metrics
    summary = {
        'total_orders': len(analysis_df),
        'net_profit_loss': metrics['net_profit_loss'],
        'settlement_rate': metrics['settlement_rate'],
        'return_rate': metrics['return_rate'],
        'total_return_settlement': total_return_settlement,
        'total_order_settlement': total_order_settlement,
        'status_changes': status_changes,
        'settlement_changes': settlement_changes,
        'pending_changes': pending_changes,
        'status_counts': status_counts.to_dict(),
        'core_metrics': metrics,
        'settlement_metrics': settlement_metrics
    }
    
    return summary 