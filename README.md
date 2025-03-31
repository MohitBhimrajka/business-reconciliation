# Business Reconciliation Dashboard

A comprehensive dashboard for reconciling business data, built with Streamlit. This application helps in analyzing and reconciling orders, returns, and settlements data from various sources.

## Features

- **Data Upload and Processing**
  - Upload orders, returns, and settlement files
  - Automatic data validation and cleaning
  - Support for CSV and Excel files
  - Historical data preservation

- **Analysis and Visualization**
  - Interactive charts and graphs
  - Trend analysis
  - Anomaly detection
  - Performance metrics

- **Reporting**
  - PDF report generation
  - Multiple report types (Reconciliation, Financial Summary, Data Quality)
  - Customizable report templates
  - Automated report scheduling

- **Data Quality Monitoring**
  - Real-time data validation
  - Error tracking and reporting
  - Data completeness checks
  - Schema validation

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for version control)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/business-reconciliation.git
   cd business-reconciliation
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create necessary directories:
   ```bash
   mkdir -p data output logs reconciliation
   ```

## Usage

1. Run the application:
   ```bash
   ./run.sh
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8501
   ```

3. Upload your data files:
   - Orders file (CSV/Excel)
   - Returns file (CSV/Excel)
   - Settlement file (CSV/Excel)

4. Select the month and year for analysis

5. View the analysis results and generate reports

## Project Structure

```
business-reconciliation/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── schemas.py          # Data schema definitions
│   ├── validation.py       # Data validation logic
│   └── reporting.py        # Report generation
├── data/                   # Data files directory
├── output/                 # Generated reports and outputs
├── logs/                   # Application logs
├── reconciliation/         # Reconciliation results
├── requirements.txt        # Python dependencies
├── run.sh                 # Application launcher
└── README.md              # Project documentation
```

## Data Schema

The application expects three main types of data files:

1. **Orders Data**
   - Order details
   - Customer information
   - Product information
   - Transaction details

2. **Returns Data**
   - Return details
   - Customer information
   - Product information
   - Settlement information

3. **Settlement Data**
   - Settlement details
   - Payment information
   - Commission details
   - Logistics information

## Error Handling

The application includes comprehensive error handling for:
- File upload errors
- Data validation errors
- Processing errors
- Report generation errors

All errors are logged and displayed to the user with clear messages and resolution steps.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Acknowledgments

- Streamlit team for the amazing framework
- Pandas team for data processing capabilities
- All contributors and users of this project 