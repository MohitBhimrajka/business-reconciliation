# Order Reconciliation Application

A Python application that processes monthly order, return, and settlement data from Excel/CSV files, consolidates this data, and performs order-level financial analysis to determine profit/loss and settlement status for each order.

## Features

- Data ingestion from multiple file formats (Excel, CSV)
- Continuous, incremental consolidation of data
- Order-level financial analysis
- Aggregate reporting and analytics
- Anomaly detection
- Visualization capabilities

## Requirements

- Python 3.6+
- pandas
- matplotlib
- seaborn

## Installation

1. Clone the repository or download the source code.
2. Install the required dependencies:

```bash
pip install pandas matplotlib seaborn
```

## Usage

### Directory Structure

Organize your data files in a directory named `reconciliation/data` with the following naming convention:

- Orders: `orders-MM-YYYY.xlsx` or `orders-MM-YYYY.csv`
- Returns: `returns-MM-YYYY.xlsx` or `returns-MM-YYYY.csv`
- Settlement: `settlement-MM-YYYY.xlsx` or `settlement-MM-YYYY.csv`

For example:
- `orders-01-2024.xlsx`
- `returns-01-2024.csv`
- `settlement-02-2024.xlsx`

### Running the Application

You can run the application using Python:

```bash
python -m reconciliation.src.main --data-dir reconciliation/data
```

To generate visualizations, add the `--visualize` flag:

```bash
python -m reconciliation.src.main --data-dir reconciliation/data --visualize
```

### Output

The application produces the following output:

1. Consolidated Files:
   - `reconciliation/output/orders.csv`
   - `reconciliation/output/returns.csv`
   - `reconciliation/output/settlement.csv`

2. Analysis Output:
   - `reconciliation/output/order_analysis_summary.csv` - Contains each order_release_id with its status and profit/loss
   - `reconciliation/output/aggregate_report.txt` - Summary report with key metrics and statistics
   - `reconciliation/output/anomalies.csv` - Identified anomalies in the data

3. Visualizations (if enabled):
   - `reconciliation/output/visualizations/` - Directory containing various visualizations

## Order Analysis Logic

The application analyzes each order to determine its status and financial outcome:

1. **Cancelled Order**:
   - Condition: `is_ship_rel == 0`
   - Status: "Cancelled"
   - Financial Outcome: Profit/Loss = 0

2. **Returned Order**:
   - Condition: `is_ship_rel == 1` AND `return_creation_date` has a value
   - Status: "Returned"
   - Financial Outcome: Loss = vendor_payout_amount - return_settlement_amount

3. **Completed Order (Shipped, Not Returned)**:
   - Condition: `is_ship_rel == 1` AND `return_creation_date` is empty/null
   
   a. **Settled**:
      - Condition: order_release_id found in settlement.csv
      - Status: "Completed - Settled"
      - Financial Outcome: Profit = vendor_payout_amount
   
   b. **Not Yet Settled**:
      - Condition: order_release_id NOT found in settlement.csv
      - Status: "Completed - Pending Settlement"
      - Financial Outcome: Profit/Loss = Not Yet Determined

## Key Metrics

The application calculates the following key metrics:

- Total count of orders by status
- Total profit from settled orders
- Total loss from returned orders
- Net overall profit/loss
- Settlement rate for shipped orders
- Return rate
- Average profit per settled order
- Average loss per returned order 