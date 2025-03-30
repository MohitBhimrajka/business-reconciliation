# Order Reconciliation Application

A Streamlit-based web application for processing and analyzing order, return, and settlement data. The application provides an interactive interface for file uploads, data analysis, and visualization of key metrics.

## Features

- **File Upload and Processing**
  - Upload orders, returns, and settlement files
  - Support for CSV and Excel file formats
  - Automatic validation of file contents
  - Standardized file naming convention

- **Interactive Dashboard**
  - Key metrics display (Total Orders, Net Profit/Loss, Settlement Rate, Return Rate)
  - Interactive charts and visualizations
  - Real-time status updates
  - Order status distribution
  - Profit/Loss distribution

- **Detailed Analysis**
  - Comprehensive order analysis
  - Status tracking and changes
  - Financial calculations
  - Settlement tracking

- **Master Data Management**
  - Consolidated view of orders, returns, and settlements
  - Deduplication of records
  - Historical data tracking

- **Anomaly Detection**
  - Identification of data inconsistencies
  - Negative profit/loss tracking
  - Missing settlement data
  - Return/Settlement conflicts

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd reconciliation
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run src/app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the application:
   - Upload files using the File Upload tab
   - View metrics and charts in the Dashboard tab
   - Analyze detailed data in the Detailed Analysis tab
   - Check master data in the Master Data tab
   - Review anomalies in the Anomalies tab

## File Structure

```
reconciliation/
├── data/               # Input data files
├── output/            # Generated output files
│   ├── master/       # Consolidated master files
│   ├── analysis/     # Analysis results
│   ├── reports/      # Generated reports
│   └── visualizations/ # Interactive charts
├── src/              # Source code
│   ├── app.py        # Streamlit application
│   ├── ingestion.py  # Data ingestion module
│   ├── analysis.py   # Analysis module
│   ├── reporting.py  # Reporting module
│   └── utils.py      # Utility functions
└── requirements.txt  # Python dependencies
```

## File Naming Convention

Input files should follow the naming convention:
- Orders: `orders-MM-YYYY.csv` or `orders-MM-YYYY.xlsx`
- Returns: `returns-MM-YYYY.csv` or `returns-MM-YYYY.xlsx`
- Settlement: `settlement-MM-YYYY.csv` or `settlement-MM-YYYY.xlsx`

Where:
- MM: Month (01-12)
- YYYY: Year (e.g., 2024)

## Data Requirements

### Orders File
Required columns:
- `order_release_id`
- `order_status`
- `final_amount`
- `total_mrp`

### Returns File
Required columns:
- `order_release_id`
- `return_amount`

### Settlement File
Required columns:
- `order_release_id`
- `settlement_amount`

## Analysis Features

### Order Status Tracking
- Cancelled
- Returned
- Completed - Settled
- Completed - Pending Settlement

### Financial Calculations
- Profit/Loss per order
- Return settlement amounts
- Order settlement amounts
- Net profit/loss

### Anomaly Detection
- Negative profit/loss orders
- Missing settlement data
- Return/Settlement conflicts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 